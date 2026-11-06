#!/usr/bin/env python3
"""
train_ctc_sentences.py — YAPpingConformer Conformer + CTC fine-tuning on real sentences.

Reads sentences.csv, matches sentences to tensors in tensors_sentence_level/index.json,
and trains YAPpingConformer with CTC loss on real sentence videos so it can segment
properly at the sentence level.

Usage (from D:\MookVani\Backend):
    python segmentor/yapv3/train_ctc_sentences.py
"""
from __future__ import annotations

import sys
from pathlib import Path

SEGMENTOR_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = SEGMENTOR_DIR.parent
sys.path.insert(0, str(SEGMENTOR_DIR))
sys.path.insert(0, str(BACKEND_DIR))

from yapv3.utils import run_ctc_sentences  # noqa: E402


def arguments():
    import argparse
    p = argparse.ArgumentParser(
        description="YAPpingConformer Conformer + CTC fine-tuning on real sentences")
    p.add_argument("--sentences-csv", type=Path,
                   default=SEGMENTOR_DIR / "data_builder" / "sentences.csv")
    p.add_argument("--index-json", type=Path,
                   default=BACKEND_DIR / "data" / "tensors_sentence_level" / "index.json")
    p.add_argument("--classes", type=Path,
                   default=BACKEND_DIR / "models" / "word_vocabulary" / "classes.json")
    p.add_argument("--out-dir", type=Path,
                   default=SEGMENTOR_DIR / "models" / "yap_ctc_sentences")
    p.add_argument("--init", type=Path, default=None,
                   help="Pretrained YAP checkpoint (yap_stage1/yap_noise_robust.pth etc.)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--train-sentences", type=int, default=None,
                   help="Max sentences to use (None = all matched)")
    p.add_argument("--max-len", type=int, default=900)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--freeze-epochs", type=int, default=5,
                   help="Epochs to freeze YAP backbone (Conformer+head only)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--emission-w", type=float, default=1.0)
    p.add_argument("--blank-w", type=float, default=0.5)
    p.add_argument("--blank-target", type=float, default=0.85)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main():
    args = arguments()
    best, history = run_ctc_sentences(
        sentences_csv=args.sentences_csv,
        index_json=args.index_json,
        classes_path=args.classes,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_sentences=args.train_sentences,
        max_len=args.max_len,
        fps=args.fps,
        lr=args.lr,
        freeze_epochs=args.freeze_epochs,
        seed=args.seed,
        d_model=args.d_model,
        heads=args.heads,
        layers=args.layers,
        init=args.init,
        cpu=args.cpu,
        val_split=args.val_split,
        emission_w=args.emission_w,
        blank_w=args.blank_w,
        blank_target=args.blank_target,
    )
    print(f"\n[DONE] Best val loss: {best:.4f}. Checkpoints in {args.out_dir}")


if __name__ == "__main__":
    main()
