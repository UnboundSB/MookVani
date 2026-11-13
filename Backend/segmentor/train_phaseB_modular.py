"""
train_phaseB_modular.py — Stage 2B training: YAPpingConformer on synthetic sentences.

Modular: imports data loading/splitting/sentence-formation from data_utils.py.
Trains YAP core + Conformer + word CE head with ground-truth segments (teacher forcing).

Key anti-leakage guarantees:
  * Each class capped at 20 clips (16 train / 4 val)
  * A clip is NEVER in both train and val (split_train_val guarantees this)
  * Validation sentences are built from val-only clips; unseen words/frames at eval time

Usage (from D:\MookVani\Backend):
    .\venv\Scripts\python.exe segmentor\train_phaseB_modular.py --overwrite --epochs 20
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

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
    presence_targets, end_targets, focal_presence_loss, layer_conf_loss,
    soft_count_loss, min_gap_penalty, dip_prior_loss,
)
from utils.checkpoint_utils import load_flexible, extract_state_dict  # noqa: E402

from data_utils import (  # noqa: E402
    load_classes, discover_clips, split_train_val,
    SentenceDataset, collate,
    SEGMENTOR_DATA as DEFAULT_DATA_SRC, CLASSES_JSON,
    MAX_CLIPS_PER_CLASS, TRAIN_CLIPS_PER_CLASS, VAL_CLIPS_PER_CLASS,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_CLASSES = CLASSES_JSON
DEFAULT_STAGE1_CKPT = HERE / "models" / "yap_stage1" / "yap_noise_robust.pth"
DEFAULT_OUT_DIR = HERE / "models" / "yap_stage2b"


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
    n_words = end_t.sum(dim=1).round().long()
    count = soft_count_loss(output["end"], n_words, mask)
    total = pres + end + 0.5 * layer + count
    return total, {
        "presence": pres.detach(), "end": end.detach(),
        "layer": layer.detach(), "count": count.detach(),
    }


def conformer_ce_loss_b(word_logits, token_mask, word_targets, word_mask):
    """Cross-entropy over real entity slots (teacher-forced segments)."""
    B, K, V1 = word_logits.shape
    flat_logits = word_logits.reshape(-1, V1)
    flat_targets = word_targets.reshape(-1)
    flat_mask = (token_mask * word_mask).reshape(-1)
    ce = F.cross_entropy(flat_logits, flat_targets, reduction="none").view(B, K)
    return (ce * flat_mask.view(B, K)).sum() / flat_mask.view(B, K).sum().clamp_min(1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[INFO] Device: {device}")

    random_seed = args.seed
    import random, numpy as np
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    classes = load_classes(args.classes)
    num_classes = len(classes)
    print(f"[INFO] {num_classes} word classes")

    # ---- discover + split (NO LEAKAGE: 16 train / 4 val per class, cap 20) ----
    corpus = discover_clips(
        data_root=DEFAULT_DATA_SRC,
        classes=classes,
        max_per_class=MAX_CLIPS_PER_CLASS,
        seed=random_seed,
    )
    train_corpus, val_corpus = split_train_val(
        corpus,
        train_per_class=TRAIN_CLIPS_PER_CLASS,
        val_per_class=VAL_CLIPS_PER_CLASS,
        seed=random_seed,
    )

    n_train_clips = sum(len(v) for v in train_corpus.values())
    n_val_clips = sum(len(v) for v in val_corpus.values())
    print(f"[SPLIT] Train: {n_train_clips} clips | Val: {n_val_clips} clips | "
          f"Classes: {len(train_corpus)} (cap {MAX_CLIPS_PER_CLASS}, "
          f"{TRAIN_CLIPS_PER_CLASS} train / {VAL_CLIPS_PER_CLASS} val each)")

    # ---- datasets ----------------------------------------------------------
    train_ds = SentenceDataset(
        train_corpus, joints=args.joints, count=args.train_sentences,
        fps=args.fps, max_words=args.max_words, max_len=args.max_len,
        seed=random_seed, training=True, apply_skip=args.apply_skip,
    )
    val_ds = SentenceDataset(
        val_corpus, joints=args.joints, count=args.val_sentences,
        fps=args.fps, max_words=args.max_words, max_len=args.max_len,
        seed=random_seed + 7, training=False, apply_skip=args.apply_skip,
    )

    tr = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=collate, num_workers=0)
    va = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=collate, num_workers=0)

    # ---- model -------------------------------------------------------------
    cfg = YAPConfig(
        parts=YAPConfig.corpus().parts,
        body_ref=YAPConfig.corpus().body_ref,
        d_model=args.d_model,
        n_mem_layers=args.n_mem_layers,
        summary_channels=args.summary_channels,
        hippo_order=args.hippo_order,
        entity_dim=args.entity_dim,
        # Anti "hands-down = sentence end": brief hand drops don't cut segments
        gap_tolerance_s=args.gap_tolerance,
        off_hold_s=0.20,
        min_seg_s=0.20,
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

    # ---- init from Stage 1 --------------------------------------------------
    if args.init and args.init.exists():
        sd = torch.load(args.init, map_location="cpu", weights_only=True)
        sd = extract_state_dict(sd)
        yap_sd = {}
        for k, v in sd.items():
            new_k = k if k.startswith("yap.") else "yap." + k
            if new_k in model.state_dict():
                yap_sd[new_k] = v
        missing, unexpected = load_flexible(model, yap_sd, label="STAGE1->PHASEB",
                                            verbose=True, strict=False)
        matched = len(yap_sd) - len(missing)
        print(f"[INIT] Loaded {matched}/{len(yap_sd)} Stage-1 tensors into YAP core; "
              f"Conformer+head start random.")

    # ---- optimizer ----------------------------------------------------------
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
                "classes": classes,
                "gap_tolerance_s": cfg.gap_tolerance_s,
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
    print("[NOTE] Trains on synthetic sentences only; validates on synthetic sentences "
          "(separate clips). No data leakage between train/val splits.")


def arguments():
    p = argparse.ArgumentParser(description="Stage 2B: YAPpingConformer synthetic segmentation + word CE")
    p.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    p.add_argument("--init", type=Path, default=DEFAULT_STAGE1_CKPT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--train-sentences", type=int, default=400)
    p.add_argument("--val-sentences", type=int, default=80)
    p.add_argument("--max-per-class", type=int, default=MAX_CLIPS_PER_CLASS)
    p.add_argument("--max-words", type=int, default=5)
    p.add_argument("--max-len", type=int, default=300)
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
    p.add_argument("--ce-weight", type=float, default=1.0)
    p.add_argument("--joints", type=int, default=53)
    p.add_argument("--gap-tolerance", type=float, default=0.50,
                   help="Seconds hands must stay down before segment ends (anti hands-down bug)")
    p.add_argument("--apply-skip", action="store_true", default=True,
                   help="Apply 5-9 frame skip after first major movement")
    p.add_argument("--no-skip", action="store_true", help="Disable frame skip")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = arguments()
    if args.no_skip:
        args.apply_skip = False
    run(args)
