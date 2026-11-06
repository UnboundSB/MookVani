"""
CLI entry point for word-level CTC training.

Usage (from D:\MookVani\Backend):
    python segmentor\train_word_ctc.py --epochs 50 --batch_size 8 --max_per_class 20
    python segmentor\train_word_ctc.py --eval-only --checkpoint models/word_ctc/best_model.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from segmentor.word_ctc.train import train_word_ctc
from segmentor.word_ctc.eval import evaluate_word_ctc


def main():
    parser = argparse.ArgumentParser(description="Word-level CTC training")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_per_class", type=int, default=20,
                        help="max samples per class (cap to prevent overfitting)")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", default=None, help="cuda | cpu | mps")
    parser.add_argument("--out_dir", default=None, help="checkpoint output dir")
    parser.add_argument("--eval-only", action="store_true", help="only run evaluation")
    parser.add_argument("--checkpoint", default=None, help="checkpoint path for eval")
    args = parser.parse_args()

    if args.eval_only:
        evaluate_word_ctc(checkpoint=args.checkpoint, batch_size=args.batch_size, device=args.device)
    else:
        train_word_ctc(
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            max_per_class=args.max_per_class, patience=args.patience,
            device=args.device, out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
