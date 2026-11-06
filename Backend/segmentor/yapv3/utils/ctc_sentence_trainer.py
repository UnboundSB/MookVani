"""
yapv3.utils.ctc_sentence_trainer — Conformer + CTC fine-tuning on real sentences.

Trains YAPpingConformer (YAP front-end + Conformer + CTC word head) on real
sentence tensors matched from data_builder/sentences.csv.  The model finds
gesture segments itself (segments=None) and CTC aligns the entity stream to
the known word sequence — no frame-level word supervision needed.

Anti-collapse measures:
  * Emission deficit barrier — gradient survives even when model emits nothing
  * Blank probability cap — prevents spamming non-blank tokens
  * Frozen backbone for first N epochs — only Conformer + head train initially
  * Gradient clipping + NaN safeguard (P0-2)
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from yap_conformer import (YAPConfig, YAPpingConformer, ConformerConfig,
                           sentence_ctc_loss)
from .ctc_sentence_data import (RealSentenceCTCDataset, collate_ctc_batch,
                                build_real_sentence_dataset)


# ===========================================================================
# Anti-collapse CTC loss
# ===========================================================================
class AntiCollapseCTCLoss(nn.Module):
    """
    Wraps nn.CTCLoss with anti-collapse terms:

      1. Plain CTC loss (zero_infinity=False, finite samples only) — a model that
         never emits a blank makes alignment impossible, loss = +inf, and we do
         NOT silently replace it with 0 (that rewards collapse).
      2. Emission deficit barrier: counts frames where non-blank >= blank (the
         comparison the greedy decoder makes) and applies log(required/achieved),
         so the gradient survives in the fully collapsed regime.
      3. Blank probability cap: upper bound on mean blank prob to stop the model
         from spamming non-blank tokens to satisfy term 2.
    """

    def __init__(
        self,
        blank_id: int = 0,
        emission_w: float = 1.0,
        blank_w: float = 0.5,
        blank_target: float = 0.85,
        min_frames_per_sign: int = 4,
        spike_w: float = 0.0,
        entropy_w: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.blank_id = blank_id
        self.emission_w = emission_w
        self.blank_w = blank_w
        self.blank_target = blank_target
        self.min_frames_per_sign = min_frames_per_sign
        self.spike_w = spike_w
        self.entropy_w = entropy_w
        self.register_buffer("class_weights", class_weights)

    def forward(self, log_probs_tbc: torch.Tensor,
                targets: torch.Tensor,
                input_lengths: torch.Tensor,
                target_lengths: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        log_probs_tbc: (T, B, V+1) log-probabilities (CTC convention: time first)
        targets:       (sum_L,) flattened int targets (0 = blank)
        input_lengths: (B,) valid frames per sequence
        target_lengths: (B,) number of target tokens per sequence
        Returns (loss, info_dict).
        """
        # ---- Plain CTC (finite samples only) ----
        per_sample = F.ctc_loss(
            log_probs_tbc, targets, input_lengths, target_lengths,
            blank=self.blank_id, reduction="none", zero_infinity=False
        )
        finite = torch.isfinite(per_sample)
        if finite.any():
            ctc = per_sample[finite].mean()
        else:
            ctc = per_sample.sum() * 0.0

        # ---- Emission deficit barrier ----
        probs = log_probs_tbc.exp()                       # (T, B, V+1)
        nonblank = probs[:, :, 1:].sum(-1)                # (T, B) P(non-blank)
        blank_p = probs[:, :, self.blank_id]              # (T, B) P(blank)
        emit_frames = (nonblank >= blank_p).float()       # frames where decoder would emit
        required = target_lengths.float().to(log_probs_tbc.device)  # (B,)
        achieved = emit_frames.sum(0).clamp_min(1)        # (B,)
        deficit = (required / achieved.clamp_min(1)).clamp_max(10.0)
        emission = deficit.log().mean()

        # ---- Blank cap ----
        mean_blank = blank_p.mean()
        blank_penalty = torch.relu(mean_blank - self.blank_target) ** 2

        loss = ctc + self.emission_w * emission + self.blank_w * blank_penalty

        info = {
            "ctc": ctc.detach().item(),
            "emission": emission.detach().item(),
            "blank_penalty": blank_penalty.detach().item(),
            "mean_blank": mean_blank.detach().item(),
            "emit_rate": emit_frames.mean().detach().item(),
        }
        return loss, info


# ===========================================================================
# Metrics (greedy CTC decode + WER)
# ===========================================================================
@torch.no_grad()
def greedy_decode(log_probs_tbc: torch.Tensor, input_lengths: torch.Tensor,
                  blank_id: int, min_dur: int = 1):
    """Greedy CTC collapse per batch item. Returns list of token-id lists."""
    B = log_probs_tbc.shape[1]
    out = []
    am = log_probs_tbc.argmax(-1).cpu()  # (T, B)
    for b in range(B):
        L = int(input_lengths[b].item())
        row = am[:L, b].tolist()
        # collapse runs, drop blanks, drop runs shorter than min_dur
        collapsed = []
        for k, g in __import__("itertools").groupby(row):
            run = list(g)
            if k != blank_id and len(run) >= min_dur:
                collapsed.append(k)
        out.append(collapsed)
    return out


@torch.no_grad()
def wer_score(refs, hyps):
    """Corpus WER via Levenshtein. refs/hyps are lists of token-id lists."""
    S = D = I = N = 0
    for r, h in zip(refs, hyps):
        n, m = len(r), len(h)
        d = np.zeros((n + 1, m + 1), dtype=np.int32)
        d[:, 0] = np.arange(n + 1); d[0, :] = np.arange(m + 1)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                d[i, j] = min(d[i - 1, j - 1] + (r[i - 1] != h[j - 1]),
                              d[i - 1, j] + 1, d[i, j - 1] + 1)
        i, j, s, dd, ii = n, m, 0, 0, 0
        while i > 0 or j > 0:
            if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (r[i - 1] != h[j - 1]):
                s += int(r[i - 1] != h[j - 1]); i -= 1; j -= 1
            elif i > 0 and d[i, j] == d[i - 1, j] + 1:
                dd += 1; i -= 1
            else:
                ii += 1; j -= 1
        S += s; D += dd; I += ii; N += max(1, n)
    return (S + D + I) / max(1, N), S, D, I, N


# ===========================================================================
# Training loop
# ===========================================================================
def train_epoch(model, criterion, optimizer, loader, device, epoch, total_epochs,
                freeze_backbone: bool, boundary_w: float = 1.0, grad_clip: float = 1.0):
    """One training epoch. Returns (avg_loss, info_avg).

    boundary_w: weight for the boundary detection loss (end head BCE
    against frame-level boundary targets from sentences.csv).
    """
    model.train()
    if freeze_backbone:
        model.yap.eval()
        for p in model.yap.parameters():
            p.requires_grad = False
    else:
        model.yap.train()

    loss_sum = 0.0
    info_sum = {"ctc": 0.0, "emission": 0.0, "blank_penalty": 0.0,
                "mean_blank": 0.0, "emit_rate": 0.0, "boundary": 0.0}
    n_batches = 0

    pbar = tqdm(loader, desc=f"CTC-sent {epoch + 1:03d}/{total_epochs:03d} [train]",
                leave=False)
    for pose, lengths, targets, target_lengths, boundary in pbar:
        B, T, D = pose.shape
        J = 53
        # pose: (B, T, 163) -> slice coords (159) -> (B, T, J, 3)
        pose = pose.to(device)[:, :, :159].view(B, T, J, 3)
        conf = torch.ones(B, T, J, device=device)
        valid = torch.ones(B, T, J, dtype=torch.bool, device=device)
        lengths = lengths.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        boundary = boundary.to(device)                      # (B, T) float

        optimizer.zero_grad(set_to_none=True)
        out = model(pose, conf, valid,
                    torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0)
                    .expand(B, -1) / 30.0,
                    lengths=lengths)
        logits = out["word_logits"]                       # (B, K, V+1)
        token_mask = out["token_mask"]                    # (B, K) bool
        log_probs_tbc = logits.log_softmax(-1).transpose(0, 1)  # (K, B, V+1)

        # CTC input length = number of entities per sequence (not frames)
        entity_lengths = token_mask.sum(1).long().to(device)

        # NaN safeguard (P0-2)
        if not torch.isfinite(log_probs_tbc).all():
            raise FloatingPointError("Nonfinite log_probs in CTC forward — stopping.")

        # ---- CTC loss ----
        ctc_loss, info = criterion(log_probs_tbc, targets, entity_lengths, target_lengths)

        # ---- Boundary loss (end head BCE vs frame-level targets) ----
        yap_out = out.get("yap", out)
        end_logits = yap_out.get("end", None)               # (B, T)
        if end_logits is not None:
            # Mask to valid frames only
            pmask = torch.zeros(B, T, device=device)
            for b in range(B):
                n = int(lengths[b].item())
                pmask[b, :n] = 1.0
            end_bce = F.binary_cross_entropy_with_logits(
                end_logits, boundary, reduction="none")
            boundary_loss = (end_bce * pmask).sum() / pmask.sum().clamp_min(1)
        else:
            boundary_loss = torch.tensor(0.0, device=device)

        # ---- Combined loss ----
        loss = ctc_loss + boundary_w * boundary_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite combined loss — stopping.")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        loss_sum += loss.item()
        info_sum["ctc"] += info["ctc"]
        info_sum["emission"] += info["emission"]
        info_sum["blank_penalty"] += info["blank_penalty"]
        info_sum["mean_blank"] += info["mean_blank"]
        info_sum["emit_rate"] += info["emit_rate"]
        info_sum["boundary"] += boundary_loss.detach().item()
        n_batches += 1
        pbar.set_postfix({k: f"{v / max(1, n_batches):.4f}" for k, v in info_sum.items()})

    pbar.close()
    avg = {k: info_sum[k] / max(n_batches, 1) for k in info_sum}
    return loss_sum / max(n_batches, 1), avg


@torch.no_grad()
def validate(model, criterion, loader, device, classes, blank_id: int,
             min_dur: int = 1, boundary_w: float = 1.0):
    """One validation epoch. Returns (avg_loss, info_avg, wer, details).

    Also computes boundary loss against frame-level targets from sentences.csv.
    """
    model.eval()
    loss_sum = 0.0
    info_sum = {"ctc": 0.0, "emission": 0.0, "blank_penalty": 0.0,
                "mean_blank": 0.0, "emit_rate": 0.0, "boundary": 0.0}
    n_batches = 0
    refs, hyps = [], []

    for pose, lengths, targets, target_lengths, boundary in loader:
        B, T, D = pose.shape
        J = 53
        pose = pose.to(device)[:, :, :159].view(B, T, J, 3)
        conf = torch.ones(B, T, J, device=device)
        valid = torch.ones(B, T, J, dtype=torch.bool, device=device)
        lengths = lengths.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        boundary = boundary.to(device)

        out = model(pose, conf, valid,
                    torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0)
                    .expand(B, -1) / 30.0,
                    lengths=lengths)
        logits = out["word_logits"]
        token_mask = out["token_mask"]
        log_probs_tbc = logits.log_softmax(-1).transpose(0, 1)

        # CTC input length = number of entities per sequence
        entity_lengths = token_mask.sum(1).long().to(device)

        # ---- CTC loss ----
        ctc_loss, info = criterion(log_probs_tbc, targets, entity_lengths, target_lengths)

        # ---- Boundary loss ----
        yap_out = out.get("yap", out)
        end_logits = yap_out.get("end", None)
        if end_logits is not None:
            pmask = torch.zeros(B, T, device=device)
            for b in range(B):
                n = int(lengths[b].item())
                pmask[b, :n] = 1.0
            end_bce = F.binary_cross_entropy_with_logits(
                end_logits, boundary, reduction="none")
            boundary_loss = (end_bce * pmask).sum() / pmask.sum().clamp_min(1)
        else:
            boundary_loss = torch.tensor(0.0, device=device)

        loss = ctc_loss + boundary_w * boundary_loss
        loss_sum += loss.item()
        info_sum["ctc"] += info["ctc"]
        info_sum["emission"] += info["emission"]
        info_sum["blank_penalty"] += info["blank_penalty"]
        info_sum["mean_blank"] += info["mean_blank"]
        info_sum["emit_rate"] += info["emit_rate"]
        info_sum["boundary"] += boundary_loss.detach().item()
        n_batches += 1

        # Decode predictions (CTC input length = entity count, not frames)
        pred_tokens = greedy_decode(log_probs_tbc, entity_lengths, blank_id, min_dur=min_dur)
        # Build reference token lists from target_lengths
        offset = 0
        for b in range(pose.shape[0]):
            L = int(target_lengths[b].item())
            refs.append(targets[offset:offset + L].tolist())
            offset += L
        hyps.extend(pred_tokens)

    avg = {k: info_sum[k] / max(n_batches, 1) for k in info_sum}
    wer, S, D, I, N = wer_score(refs, hyps)
    return (loss_sum / max(n_batches, 1), avg, wer, {"S": S, "D": D, "I": I, "N": N,
                                                     "refs": refs, "hyps": hyps})


# ===========================================================================
# Main training entry
# ===========================================================================
def run_ctc_sentences(
    sentences_csv: Path | str,
    index_json: Path | str,
    classes_path: Path | str,
    out_dir: Path | str = "models/yap_ctc_sentences",
    epochs: int = 20,
    batch_size: int = 4,
    train_sentences: Optional[int] = None,
    max_len: int = 900,
    fps: float = 30.0,
    lr: float = 3e-4,
    freeze_epochs: int = 5,
    seed: int = 42,
    d_model: int = 256,
    heads: int = 4,
    layers: int = 4,
    init: Optional[Path | str] = None,
    cpu: bool = False,
    val_split: float = 0.1,
    emission_w: float = 1.0,
    blank_w: float = 0.5,
    blank_target: float = 0.85,
    boundary_w: float = 1.0,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    restart_every: Optional[int] = None,
    escape_patience: int = 0,
    escape_scale: float = 0.02,
    segmentation_csv: Optional[Path | str] = None,
):
    """Run Conformer + CTC fine-tuning on real sentences from sentences.csv.

    boundary_w: weight for the boundary detection loss (end head BCE against
    frame-level boundary targets generated from sentences.csv word positions).

    Pretrainer-aligned options (when restart_every is set):
      weight_decay, grad_clip, restart_every (cosine warm restart interval),
      escape_patience, escape_scale (plateau escape).

    If segmentation_csv is provided, human-annotated boundary marks from
    segmentation_clean.csv are used as validation targets instead of
    Gaussian-bump approximations — this tells you if the model actually
    learned to segment real sentences.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    print(f"[DEVICE] {device}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(classes_path, encoding="utf-8") as f:
        classes = json.load(f)
    num_classes = len(classes)
    blank_id = num_classes  # ConformerConfig reserves class 0 = blank, words 1..num_words

    # ---- datasets ----
    train_ds, val_ds = build_real_sentence_dataset(
        sentences_csv, index_json, classes_path,
        count=train_sentences, max_len=max_len, fps=fps, seed=seed,
        training=True, val_split=val_split
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_ctc_batch, num_workers=0)
    val_loader = (DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_ctc_batch, num_workers=0)
                  if val_ds is not None else None)

    # ---- Segmentation validation set (human-annotated boundary marks) ----
    seg_val_loader = None
    if segmentation_csv is not None:
        seg_path = Path(segmentation_csv)
        if seg_path.exists():
            seg_marks = load_segmentation_marks(seg_path, index_json)
            seg_items = build_segmentation_boundary_items(
                seg_marks, classes, max_len=max_len, fps=fps
            )
            if seg_items:
                seg_val_loader = DataLoader(
                    seg_items, batch_size=batch_size, shuffle=False,
                    collate_fn=collate_segmentation_batch, num_workers=0
                )
                print(f"[SEG] Loaded {len(seg_items)} segmentation validation items "
                      f"from {seg_path.name}")
            else:
                print(f"[SEG] No usable segmentation items from {seg_path.name}")
        else:
            print(f"[SEG] Segmentation CSV not found: {seg_path}")

    # ---- model ----
    ycfg = YAPConfig.corpus(d_model=128, n_mem_layers=3, summary_channels=32,
                            hippo_order=16, entity_dim=256)
    ccfg = ConformerConfig(num_words=num_classes, d_model=d_model, heads=heads,
                           layers=layers)
    model = YAPpingConformer(ycfg, ccfg).to(device)

    if init is not None:
        path = Path(init)
        if not path.exists():
            raise FileNotFoundError(f"Init checkpoint not found: {path}")
        # Load YAP weights only (Conformer + head are randomly initialized)
        from .model import load_state_from_checkpoint
        state = load_state_from_checkpoint(path)
        own = model.yap.state_dict()
        compatible = {k: v for k, v in state.items()
                      if k in own and own[k].shape == v.shape}
        if not compatible:
            raise RuntimeError(f"ZERO compatible YAP tensors in init checkpoint {path}")
        missing, unexpected = model.yap.load_state_dict(compatible, strict=False)
        print(f"[INIT] loaded {len(compatible)}/{len(own)} YAP tensors from {path}; "
              f"missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print("[INIT] no init — YAP backbone starts random")

    # ---- optimizer: Conformer + head only (YAP frozen for first freeze_epochs) ----
    trainable = list(model.ent_proj.parameters()) + list(model.meta_proj.parameters()) + \
                list(model.blocks.parameters()) + list(model.head.parameters()) + \
                [model.bos]
    if not freeze_epochs:
        # Unfreeze YAP too after the freeze period
        trainable += list(model.yap.parameters())

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    for g in optimizer.param_groups:
        g["base_lr"] = g["lr"]

    # Scheduler: cosine with warm restarts (pretrainer style) or plain cosine
    if restart_every and restart_every > 0:
        sys_path_added = False
        backend_utils = Path(__file__).resolve().parent.parent.parent.parent / "utils"
        if str(backend_utils.parent) not in sys.path:
            sys.path.insert(0, str(backend_utils.parent))
            sys_path_added = True
        from training_utils import build_warm_restart_scheduler
        scheduler, _ = build_warm_restart_scheduler(
            optimizer, total_epochs=epochs, restart_every=restart_every,
            eta_min=lr * 0.01
        )
        print(f"[SCHEDULER] cosine warm restart every {restart_every} epochs")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        print(f"[SCHEDULER] cosine annealing over {epochs} epochs")

    criterion = AntiCollapseCTCLoss(
        blank_id=blank_id,
        emission_w=emission_w,
        blank_w=blank_w,
        blank_target=blank_target,
    ).to(device)

    # Plateau escape import (pretrainer.py style)
    if escape_patience and escape_patience > 0:
        backend_utils = Path(__file__).resolve().parent.parent.parent.parent / "utils"
        if str(backend_utils.parent) not in sys.path:
            sys.path.insert(0, str(backend_utils.parent))
        from training_utils import escape_from_plateau

    best = math.inf
    history = []
    epochs_without_improvement = 0
    escapes = 0

    for epoch in range(epochs):
        train_ds.epoch = epoch
        if val_ds is not None:
            val_ds.epoch = epoch

        # Unfreeze YAP backbone after freeze_epochs
        freeze_backbone = epoch < freeze_epochs
        if epoch == freeze_epochs and freeze_epochs > 0:
            print(f"\n[UNFREEZE] YAP backbone trainable from epoch {epoch + 1}")
            trainable += list(model.yap.parameters())
            for g in optimizer.param_groups:
                g["params"] = trainable

        tr_loss, tr_info = train_epoch(model, criterion, optimizer, train_loader,
                                       device, epoch, epochs, freeze_backbone,
                                       boundary_w=boundary_w, grad_clip=grad_clip)

        # Use segmentation validation set (human marks) if available,
        # otherwise fall back to Gaussian-bump val set
        val_loader_active = seg_val_loader if seg_val_loader is not None else val_loader
        if val_loader_active is not None:
            v_loss, v_info, v_wer, v_details = validate(model, criterion, val_loader_active,
                                                        device, classes, blank_id,
                                                        boundary_w=boundary_w)
        else:
            v_loss, v_info, v_wer, v_details = tr_loss, tr_info, 0.0, {}

        history.append({"epoch": epoch + 1, "train": tr_loss, "val": v_loss,
                        **{f"train_{k}": tr_info[k] for k in tr_info},
                        **{f"val_{k}": v_info[k] for k in v_info},
                        "wer": v_wer,
                        **{f"wer_{k}": v_details.get(k, 0) for k in ("S", "D", "I", "N")}})

        improved = v_loss < best
        if improved:
            best = v_loss
            epochs_without_improvement = 0
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch + 1,
                "history": history,
                "classes": classes,
            }, out / "yap_ctc_sentences_best.pth")
        else:
            epochs_without_improvement += 1

        # --- plateau escape (pretrainer.py style) ---
        if escape_patience and escape_patience > 0 and epochs_without_improvement >= escape_patience:
            escapes += 1
            print(f"  [ESCAPE] No val improvement for {epochs_without_improvement} epochs "
                  f"(val loss {v_loss:.4f}) - kicking the model out of the plateau.", flush=True)
            esc_info = escape_from_plateau(model, optimizer, scale=escape_scale)
            print(f"  [ESCAPE] {esc_info}", flush=True)
            epochs_without_improvement = 0
            swa_state, swa_n = None, 0

        print("\n" + "=" * 76)
        print(f"  YAP CTC SENTENCES | EPOCH {epoch + 1:03d}/{epochs:03d} | "
              f"LR {scheduler.get_last_lr()[0]:.2e} | {'frozen' if freeze_backbone else 'unfrozen'}")
        print("-" * 76)
        print(f'  {"Loss":<20} {"Train":>10} {"Val":>10}')
        print("-" * 76)
        print(f'  {"CTC":<20} {tr_info["ctc"]:>10.4f} {v_info["ctc"]:>10.4f}')
        print(f'  {"Emission":<20} {tr_info["emission"]:>10.4f} {v_info["emission"]:>10.4f}')
        print(f'  {"BlankPenalty":<20} {tr_info["blank_penalty"]:>10.4f} {v_info["blank_penalty"]:>10.4f}')
        print(f'  {"TOTAL":<20} {tr_loss:>10.4f} {v_loss:>10.4f}')
        print("-" * 76)
        print(f'  Mean blank (train/val): {tr_info["mean_blank"]:.4f} / {v_info["mean_blank"]:.4f}')
        print(f'  Emit rate (train/val):  {tr_info["emit_rate"]:.4f} / {v_info["emit_rate"]:.4f}')
        print(f'  Boundary loss (tr/val): {tr_info["boundary"]:.4f} / {v_info["boundary"]:.4f}')
        if val_loader is not None:
            print(f'  Val WER: {v_wer * 100:.1f}%  (S{v_details.get("S",0)} '
                  f'D{v_details.get("D",0)} I{v_details.get("I",0)}/N{v_details.get("N",0)})')
        print(f'  Best val: {best:.4f}' + (' [new checkpoint saved]' if improved else ''))
        print("=" * 76 + "\n", flush=True)

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "best": best,
            "history": history,
        }, out / "yap_ctc_sentences_resume.pth")
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        scheduler.step()

    torch.save(model.state_dict(), out / "yap_ctc_sentences_final.pth")
    print(f"[DONE] Best val loss = {best:.4f}")
    return best, history
