#!/usr/bin/env python3
"""
train_phase3.py — YAP gesture-segmentation pretraining (Phase 3).

Stitches 2-5 random words from classes.json into synthetic sentences,
trains YAP presence + end heads (everything else frozen), validates with
boundary F1 / IoU / count MAE using the production SegmentTracker.

Usage (from D:\MookVani\Backend):
    python segmentor/yapv3/train_phase3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure yap_conformer.py and yapv3/utils are importable
SEGMENTOR_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = SEGMENTOR_DIR.parent
sys.path.insert(0, str(SEGMENTOR_DIR))
sys.path.insert(0, str(BACKEND_DIR))

from yapv3.utils import (AugmentationConfig, run_phase3)
from yapv3.utils.data import build_sentence_dataset  # noqa: F401  (re-exported)


def arguments():
    import argparse
    p = argparse.ArgumentParser(description="YAP Phase-3 gesture segmentation pretraining")
    p.add_argument("--data-root", type=Path, default=BACKEND_DIR / "data")
    p.add_argument("--classes", type=Path,
                   default=BACKEND_DIR / "models" / "word_vocabulary" / "classes.json")
    p.add_argument("--out-dir", type=Path,
                   default=SEGMENTOR_DIR / "models" / "yap_phase3")
    p.add_argument("--init", type=Path, default=None,
                   help="Pretrained YAP checkpoint (yap_stage1/yap_noise_robust.pth etc.)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-sentences", type=int, default=4000)
    p.add_argument("--val-sentences", type=int, default=400)
    p.add_argument("--max-per-class", type=int, default=None,
                   help="Max clips per class to load (None = all)")
    p.add_argument("--max-len", type=int, default=900)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-mem-layers", type=int, default=3)
    p.add_argument("--summary-channels", type=int, default=32)
    p.add_argument("--hippo-order", type=int, default=16)
    p.add_argument("--entity-dim", type=int, default=256)
    p.add_argument("--cpu", action="store_true")
    # Augmentation toggles
    p.add_argument("--no-mirror", action="store_true",
                   help="Disable left-right mirror augmentation")
    p.add_argument("--no-shift", action="store_true", help="Disable x/y shift augmentation")
    p.add_argument("--no-tilt", action="store_true", help="Disable z-tilt augmentation")
    p.add_argument("--no-time-warp", action="store_true",
                   help="Disable Gaussian accel/decel time-warp")
    p.add_argument("--no-hold-last", action="store_true",
                   help="Disable hold-last-valid for zero coords")
    return p.parse_args()


def main():
    args = arguments()

    aug = AugmentationConfig(
        mirror=not args.no_mirror,
        shift=not args.no_shift,
        tilt=not args.no_tilt,
        time_warp=not args.no_time_warp,
        hold_last_valid=not args.no_hold_last,
        seed=args.seed,
    )

    best, history = run_phase3(
        data_root=args.data_root,
        classes_path=args.classes,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_sentences=args.train_sentences,
        val_sentences=args.val_sentences,
        max_per_class=args.max_per_class,
        max_len=args.max_len,
        fps=args.fps,
        lr=args.lr,
        seed=args.seed,
        d_model=args.d_model,
        n_mem_layers=args.n_mem_layers,
        summary_channels=args.summary_channels,
        hippo_order=args.hippo_order,
        entity_dim=args.entity_dim,
        init=args.init,
        cpu=args.cpu,
        aug=aug,
    )
    print(f"\n[DONE] Best val loss: {best:.4f}. Checkpoint in {args.out_dir}")


if __name__ == "__main__":
    main()
