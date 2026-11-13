"""
Stage 2B — YAPpingConformer synthetic segmentation + word classification.

Trains YAPpingConformer (YAP front-end + Conformer + word classification head)
on synthetic stitched sentences with ground-truth segments and word labels.

Key fix for the "hands down = sentence end" problem:
  * gap_tolerance_s raised to 0.5s — brief hand drops mid-sign do NOT end the
    segment; only a sustained absence counts as a boundary.
  * min_seg_s keeps segments long enough to be real signs, not flickers.
  * Conformer CE loss uses ground-truth segments (teacher forcing), so the
    model learns word-level context from the Conformer, not just boundary
    detection from the YAP presence/end heads.
  * sentence_ctc_loss available for Stage 3 (predicted segments, no alignment).

Training order (run these in order):
  Stage 1: python Backend/segmentor/train_noise_filter.py
           -> segmentor/models/yap_stage1/yap_noise_robust.pth
  Stage 2A: python Backend/segmentor/train_segmentor.py --overwrite
            -> segmentor/models/yap_stage2/yap_phaseA_best.pth  (segmentation only)
  Stage 2B: python Backend/segmentor/train_segmentor_phaseB.py --overwrite
            -> segmentor/models/yap_stage2b/yap_phaseB_best.pth  (segmentation + word CE)
  Stage 3:  python Backend/training/train_real_sentences.py --init <conformer_ctc_ckpt>
            -> real-sentence fine-tune

Usage (from D:\MookVani\Backend):
    .\venv\Scripts\python.exe segmentor\train_segmentor_phaseB.py --overwrite --epochs 30
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BACKEND))

from yap_conformer import (  # noqa: E402
    YAP, YAPConfig, YAPpingConformer, ConformerConfig,
    make_segments, presence_targets, end_targets,
    focal_presence_loss, layer_conf_loss, soft_count_loss,
    min_gap_penalty, dip_prior_loss, sentence_ctc_loss,
    Segment,
)
from utils.checkpoint_utils import load_flexible, extract_state_dict  # noqa: E402
from utils.pose_augmentation import AugmentConfig, augment_pose_sequence  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_CLASSES = BACKEND / "models" / "word_vocabulary" / "classes.json"
DEFAULT_STAGE1_CKPT = HERE / "models" / "yap_stage1" / "yap_noise_robust.pth"
DEFAULT_OUT_DIR = HERE / "models" / "yap_stage2b"
DEFAULT_DATA_SRC = HERE / "data"  # segmentor/data/<word>/<take>/word.pt


# ---------------------------------------------------------------------------
# Phase B dataset: synthetic sentences with WORD LABELS per segment
# ---------------------------------------------------------------------------
def discover_corpus(root: Path, classes: list[str], max_per_class: int, seed: int):
    """{class_idx: [clip_path, ...]} from segmentor/data/<word>/<take>/word.pt"""
    rng = random.Random(seed)
    result = {}
    for cid, name in enumerate(classes):
        folder = root / name
        if not folder.is_dir():
            continue
        paths = [p for p in folder.rglob("*.pt") if p.name != "processed.pt"]
        rng.shuffle(paths)
        good = []
        for path in paths:
            try:
                obj = torch.load(path, map_location="cpu", weights_only=True)
                if isinstance(obj, torch.Tensor) and obj.numel() % 163 == 0 and obj.numel() >= 163 * 4:
                    good.append(path)
            except (RuntimeError, ValueError, OSError, TypeError):
                continue
            if len(good) == max_per_class:
                break
        if good:
            result[cid] = good
    return result


def split_sources(corpus, seed, train_fraction=0.8):
    rng = random.Random(seed)
    train, val = {}, {}
    for key, sources in corpus.items():
        items = sources.copy()
        rng.shuffle(items)
        if len(items) < 2:
            train[key] = items
            continue
        n = min(len(items) - 1, max(1, int(len(items) * train_fraction)))
        train[key], val[key] = items[:n], items[n:]
    return train, val


def read_xyz(path: Path, joints: int):
    clip = torch.load(path, map_location="cpu", weights_only=True).float()
    clip = clip.reshape(-1, 163)
    if joints * 3 > 163:
        raise ValueError("Only the 53-joint corpus layout is supported")
    return clip[:, :joints * 3].reshape(-1, joints, 3).numpy().copy()


def stretch(clip, rng):
    length = max(4, round(len(clip) * rng.uniform(0.8, 1.25)))
    pos = np.linspace(0, len(clip) - 1, length)
    low = pos.astype(np.int64)
    high = np.minimum(low + 1, len(clip) - 1)
    a = (pos - low).astype(np.float32)[:, None, None]
    return (clip[low] * (1 - a) + clip[high] * a).astype(np.float32)


def make_sentence_with_labels(corpus, joints, rng, fps, max_words, max_len):
    """
    Like make_sentence but returns word labels alongside spans.

    Returns:
        raw      (T, joints, 3) float32 pose array
        spans    [(start_t, end_t, class_idx), ...]  one per word in signing order
    """
    keys = list(corpus)
    n_words = rng.randint(1, max_words)
    frames = []
    spans = []
    cursor = 0
    lead = rng.randint(10, 25)
    frames.append(np.zeros((lead, joints, 3), np.float32))
    cursor += lead
    for k in range(n_words):
        cid = rng.choice(keys)
        path = rng.choice(corpus[cid])
        clip = stretch(read_xyz(path, joints), rng)
        if k:
            gap = rng.choice((0, 0, rng.randint(2, 7), rng.randint(10, 25)))
            if gap:
                frames.append(np.zeros((gap, joints, 3), np.float32))
                cursor += gap
        start_t = cursor / fps
        frames.append(clip)
        cursor += len(clip)
        end_t = cursor / fps
        spans.append((start_t, end_t, cid))
    raw = np.concatenate(frames, axis=0)
    if raw.shape[0] > max_len:
        raise ValueError("sentence too long for --max-len")
    return raw, spans


class SentenceSetB(torch.utils.data.Dataset):
    """Synthetic sentences with word labels per segment (Phase B)."""

    def __init__(self, corpus, joints, count, fps, max_words, max_len, seed, training):
        self.corpus = corpus
        self.joints = joints
        self.count = count
        self.fps = fps
        self.max_words = max_words
        self.max_len = max_len
        self.seed = seed
        self.training = training
        self.epoch = 0

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        rng = random.Random(self.seed + index * 1000003 + (self.epoch * 1000000007 if self.training else 0))
        for _ in range(20):
            try:
                raw, spans = make_sentence_with_labels(
                    self.corpus, self.joints, rng, self.fps, self.max_words, self.max_len
                )
                break
            except ValueError:
                continue
        else:
            raise ValueError("Could not generate a sentence; increase --max-len")

        pose = torch.from_numpy(raw)
        T = len(pose)
        valid = torch.isfinite(pose).all(-1) & (pose.abs().sum(-1) > 0)
        pose = torch.nan_to_num(pose, nan=0.0, posinf=0.0, neginf=0.0)
        conf = valid.float()
        if self.training:
            pose += torch.randn_like(pose) * 0.002 * valid.unsqueeze(-1)
            if rng.random() < 0.25 and T >= 8:
                s = rng.randrange(T - 4)
                e = min(T, s + rng.randint(2, 6))
                part = slice(0, 21) if rng.random() < 0.5 else slice(21, 42)
                valid[s:e, part] = False
                conf[s:e, part] = 0
                pose[s:e, part] = 0

        times = torch.arange(T, dtype=torch.float32) / self.fps

        # Build per-frame targets from spans (frame indices for Segment construction)
        presence_t = torch.zeros(T, dtype=torch.float32)
        end_t = torch.zeros(T, dtype=torch.float32)
        word_labels = torch.zeros(len(spans), dtype=torch.long)  # class idx per segment
        start_indices = []
        end_indices = []
        for i, (s_t, e_t, cid) in enumerate(spans):
            s_idx = max(0, int(round(s_t * self.fps)))
            e_idx = min(T, int(round(e_t * self.fps)))
            if e_idx <= s_idx:
                continue
            presence_t[s_idx:e_idx] = 1.0
            # end prob peak at the last frame of the segment
            end_t[e_idx - 1] = 1.0
            word_labels[i] = cid
            start_indices.append(s_idx)
            end_indices.append(e_idx)

        return pose, conf, valid, times, presence_t, end_t, word_labels, start_indices, end_indices


def collate_b(batch):
    poses, confs, valids, times, presences, ends, word_labels_list, start_indices_list, end_indices_list = zip(*batch)
    B = len(poses)
    T = max(p.shape[0] for p in poses)
    J = poses[0].shape[1]

    pose_b = torch.zeros(B, T, J, 3)
    conf_b = torch.zeros(B, T, J)
    valid_b = torch.zeros(B, T, J, dtype=torch.bool)
    t_b = torch.zeros(B, T)
    mask_b = torch.zeros(B, T, dtype=torch.bool)

    pres_b = torch.zeros(B, T)
    end_b = torch.zeros(B, T)

    # Build Segment objects from frame indices (teacher forcing)
    all_segments = []  # List[List[Segment]] per batch item
    counts = []
    for i in range(B):
        segs = []
        starts = start_indices_list[i]
        ends_i = end_indices_list[i]
        for s_idx, e_idx in zip(starts, ends_i):
            if e_idx <= s_idx:
                continue
            segs.append(Segment(s_idx, e_idx, s_idx / 30.0, e_idx / 30.0, "given", 0.0))
        all_segments.append(segs)
        counts.append(len(segs))

    K = min(max(max(counts) if counts else 1, 1), 12)  # cap at max_entities default

    word_targets_b = torch.zeros(B, K, dtype=torch.long)
    word_mask_b = torch.zeros(B, K, dtype=torch.bool)

    for i, (segs, wl) in enumerate(zip(all_segments, word_labels_list)):
        for j, s in enumerate(segs):
            if j < K:
                word_targets_b[i, j] = wl[j] if j < len(wl) else 0
                word_mask_b[i, j] = True

    for i, (p, c, v, t_, pr, en) in enumerate(zip(
            poses, confs, valids, times, presences, ends)):
        n = p.shape[0]
        pose_b[i, :n] = p
        conf_b[i, :n] = c
        valid_b[i, :n] = v
        t_b[i, :n] = t_
        mask_b[i, :n] = True
        pres_b[i, :n] = pr[:n]
        end_b[i, :n] = en[:n]

    return (pose_b, conf_b, valid_b, t_b, mask_b,
            pres_b, end_b, word_targets_b, word_mask_b, all_segments, K)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def yap_losses_b(output, presence_t, end_t, mask):
    pres = focal_presence_loss(output["presence"], presence_t, mask)
    end_raw = F.binary_cross_entropy_with_logits(
        output["end"], end_t, reduction="none",
        pos_weight=presence_t.new_tensor(10.0))
    end = (end_raw * mask).sum() / mask.sum().clamp_min(1)
    layer = layer_conf_loss(output["layer_conf"], presence_t, mask)
    # n_words = number of segments per sample = number of end peaks
    n_words = end_t.sum(dim=1).round().long()
    count = soft_count_loss(output["end"], n_words, mask)
    total = pres + end + 0.5 * layer + count
    return total, {"presence": pres.detach(), "end": end.detach(),
                   "layer": layer.detach(), "count": count.detach()}


def conformer_ce_loss_b(word_logits, token_mask, word_targets, word_mask):
    """Cross-entropy over real entity slots (teacher-forced segments)."""
    B, K, V1 = word_logits.shape
    flat_logits = word_logits.reshape(-1, V1)
    flat_targets = word_targets.reshape(-1)
    flat_mask = word_mask.reshape(-1)
    ce = F.cross_entropy(flat_logits, flat_targets, reduction="none").view(B, K)
    return (ce * token_mask * word_mask).sum() / (token_mask * word_mask).sum().clamp_min(1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[INFO] Device: {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.classes, "r", encoding="utf-8") as f:
        classes = json.load(f)
    if isinstance(classes, dict):
        classes = classes["classes"]
    num_classes = len(classes)
    print(f"[INFO] {num_classes} word classes")

    corpus = discover_corpus(DEFAULT_DATA_SRC, classes, args.max_per_class, args.seed)
    train_sources, val_sources = split_sources(corpus, args.seed)

    # ---- YAP config: FIXED for "hands down = sentence end" ----------------
    cfg = YAPConfig(
        parts=YAPConfig.corpus().parts,          # J = 53 corpus layout
        body_ref=YAPConfig.corpus().body_ref,
        d_model=args.d_model,
        n_mem_layers=args.n_mem_layers,
        summary_channels=args.summary_channels,
        hippo_order=args.hippo_order,
        entity_dim=args.entity_dim,
        # --- boundaries: hands down does NOT instantly end a segment ---
        on_thr=0.6,
        off_thr=0.4,
        on_hold_s=0.06,
        off_hold_s=0.20,          # presence must stay low this long to end
        gap_tolerance_s=0.50,     # was 0.25 — brief hand drops OK mid-sign
        cut_thr=0.7,
        min_seg_s=0.20,
        max_seg_s=4.0,
        still_speed_thr=0.05,
        still_flush_s=3.5,
        overlap_fraction=0.5,
        preroll_min_s=0.10,
        preroll_max_s=0.50,
    )
    ccfg = ConformerConfig(
        num_words=num_classes,
        d_model=args.conformer_d_model,
        heads=args.conformer_heads,
        layers=args.conformer_layers,
        conv_kernel=args.conformer_conv_kernel,
        dropout=args.conformer_dropout,
        max_entities=args.max_entities,
    )
    model = YAPpingConformer(cfg, ccfg).to(device)

    # ---- init from Stage 1 ------------------------------------------------
    if args.init and args.init.exists():
        sd = torch.load(args.init, map_location="cpu", weights_only=True)
        sd = extract_state_dict(sd)
        # Stage 1 checkpoint saves YAP directly (keys like "core.cleaner.x").
        # YAPpingConformer wraps as self.yap, so keys need "yap." prefix.
        yap_sd = {}
        for k, v in sd.items():
            new_k = k if k.startswith("yap.") else "yap." + k
            if new_k in model.state_dict():
                yap_sd[new_k] = v
        missing, unexpected = load_flexible(model, yap_sd, label="STAGE1->PHASEB",
                                            verbose=True, strict=False)
        print(f"[INIT] Loaded {len(yap_sd) - len(missing)}/{len(yap_sd)} Stage-1 tensors "
              f"into YAP core; Conformer+head start random.")

    # ---- datasets ---------------------------------------------------------
    train_ds = SentenceSetB(train_sources, args.joints, args.train_sentences,
                            args.fps, args.max_words, args.max_len, args.seed, True)
    val_ds = SentenceSetB(val_sources, args.joints, args.val_sentences,
                          args.fps, args.max_words, args.max_len, args.seed + 7, False)

    tr = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=collate_b, num_workers=0)
    va = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=collate_b, num_workers=0)

    # ---- optimizer --------------------------------------------------------
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "yap_phaseB_best.pth").exists() and not args.overwrite:
        raise FileExistsError("Output checkpoint exists. Use --out-dir or --overwrite")

    best = math.inf
    history = []

    for epoch in range(args.epochs):
        train_ds.epoch = epoch
        model.train()
        train_sum = 0.0
        train_ce = 0.0
        for batch in tqdm(tr, desc=f"PhaseB {epoch + 1}/{args.epochs}", leave=False):
            (pose, conf, valid, t_, mask,
             pres_t, end_t, word_targets, word_mask, all_segments, K) = batch
            pose, conf, valid, t_, mask = (x.to(device) for x in
                                           (pose, conf, valid, t_, mask))
            pres_t, end_t = pres_t.to(device), end_t.to(device)
            word_targets, word_mask = word_targets.to(device), word_mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            out_dict = model(pose, conf, valid, t_, segments=all_segments)
            yap_loss, yap_parts = yap_losses_b(out_dict["yap"], pres_t, end_t, mask)
            ce_loss = conformer_ce_loss_b(out_dict["word_logits"],
                                          out_dict["token_mask"],
                                          word_targets, word_mask)
            loss = yap_loss + args.ce_weight * ce_loss

            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite Phase-B loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            train_sum += loss.item()
            train_ce += ce_loss.item()

        model.eval()
        val_sum = 0.0
        val_ce = 0.0
        with torch.no_grad():
            for batch in va:
                (pose, conf, valid, t_, mask,
                 pres_t, end_t, word_targets, word_mask, all_segments, K) = batch
                pose, conf, valid, t_, mask = (x.to(device) for x in
                                               (pose, conf, valid, t_, mask))
                pres_t, end_t = pres_t.to(device), end_t.to(device)
                word_targets, word_mask = word_targets.to(device), word_mask.to(device)

                out_dict = model(pose, conf, valid, t_, segments=all_segments)
                yap_loss, yap_parts = yap_losses_b(out_dict["yap"], pres_t, end_t, mask)
                ce_loss = conformer_ce_loss_b(out_dict["word_logits"],
                                              out_dict["token_mask"],
                                              word_targets, word_mask)
                loss = yap_loss + args.ce_weight * ce_loss
                val_sum += loss.item()
                val_ce += ce_loss.item()

        tr_loss = train_sum / len(tr)
        v_loss = val_sum / len(va)
        tr_ce = train_ce / len(tr)
        v_ce = val_ce / len(va)

        history.append({
            "epoch": epoch + 1,
            "train": tr_loss, "val": v_loss,
            "train_ce": tr_ce, "val_ce": v_ce,
            **{f"val_{k}": yap_parts[k] for k in yap_parts},
        })

        print(f"\n[PhaseB Ep {epoch + 1:03d}/{args.epochs:03d}] "
              f"train {tr_loss:.4f} | val {v_loss:.4f} | "
              f"train_CE {tr_ce:.4f} | val_CE {v_ce:.4f} | "
              f"presence {yap_parts['presence']:.4f} end {yap_parts['end']:.4f}")

        if v_loss < best:
            best = v_loss
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch + 1,
                "history": history,
                "cfg": {
                    "d_model": cfg.d_model, "n_mem_layers": cfg.n_mem_layers,
                    "summary_channels": cfg.summary_channels,
                    "hippo_order": cfg.hippo_order, "entity_dim": cfg.entity_dim,
                    "gap_tolerance_s": cfg.gap_tolerance_s,
                    "off_hold_s": cfg.off_hold_s,
                    "min_seg_s": cfg.min_seg_s,
                },
                "ccfg": {
                    "num_words": ccfg.num_words, "d_model": ccfg.d_model,
                    "heads": ccfg.heads, "layers": ccfg.layers,
                    "conv_kernel": ccfg.conv_kernel, "dropout": ccfg.dropout,
                    "max_entities": ccfg.max_entities,
                },
                "classes": classes,
            }, out / "yap_phaseB_best.pth")
            print(f"  * NEW BEST val {best:.4f}")

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "best": best,
            "history": history,
        }, out / "yap_phaseB_resume.pth")
        scheduler.step()

    torch.save(model.state_dict(), out / "yap_phaseB_final.pth")
    print(f"\n[DONE] Best val loss={best:.4f} -> {out / 'yap_phaseB_best.pth'}")
    print("[NOTE] Phase B trains segmentation + Conformer word CE on ground-truth segments.")
    print("       Hands-down no longer instantly ends a segment (gap_tolerance_s=0.50).")


def arguments():
    p = argparse.ArgumentParser(description="Stage 2B: YAPpingConformer synthetic segmentation + word CE")
    p.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    p.add_argument("--init", type=Path, default=DEFAULT_STAGE1_CKPT,
                   help="Stage-1 yap_noise_robust.pth to initialize YAP core from")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--train-sentences", type=int, default=1600)
    p.add_argument("--val-sentences", type=int, default=200)
    p.add_argument("--max-per-class", type=int, default=22)
    p.add_argument("--max-words", type=int, default=5)
    p.add_argument("--max-len", type=int, default=900)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-mem-layers", type=int, default=3)
    p.add_argument("--summary-channels", type=int, default=32)
    p.add_argument("--hippo-order", type=int, default=16)
    p.add_argument("--entity-dim", type=int, default=256)
    p.add_argument("--conformer-d-model", type=int, default=256)
    p.add_argument("--conformer-heads", type=int, default=4)
    p.add_argument("--conformer-layers", type=int, default=4)
    p.add_argument("--conformer-conv-kernel", type=int, default=7)
    p.add_argument("--conformer-dropout", type=float, default=0.1)
    p.add_argument("--max-entities", type=int, default=12)
    p.add_argument("--ce-weight", type=float, default=1.0,
                   help="Weight of Conformer word CE loss vs YAP losses")
    p.add_argument("--joints", type=int, default=53)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(arguments())
