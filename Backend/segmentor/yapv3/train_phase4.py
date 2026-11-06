#!/usr/bin/env python3
"""
train_phase4.py — YAPpingConformer Conformer + CTC training on REAL sentences
with boundary detection loss.

Fine-tunes YAPpingConformer (YAP front-end + Conformer + CTC word head) on real
sentence videos matched from data_builder/sentences.csv to
tensors_sentence_level/index.json.

KEY FEATURE: Boundary detection loss.
  * Reads sentences.csv for the sentence list
  * Matches each sentence to its tensor in tensors_sentence_level/index.json
  * Generates frame-level boundary targets from word positions (Gaussian bumps
    at each word end, plus final sentence end)
  * Trains with CTC loss (word sequence alignment) + boundary loss (end head
    BCE against frame-level targets)
  * The boundary loss penalizes the model for missing real sentence boundaries

Uses AntiCollapseCTCLoss to prevent the model from collapsing to all-blank.

Pipeline position: Stage 3 / Phase-4 — real-sentence fine-tuning after phase-3
gesture-segmentation pretraining (presence + end heads).

Usage (from D:\MookVani\Backend):
    python segmentor/yapv3/train_phase4.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

SEGMENTOR_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = SEGMENTOR_DIR.parent
sys.path.insert(0, str(SEGMENTOR_DIR))
sys.path.insert(0, str(BACKEND_DIR))

from yapv3.utils import run_ctc_sentences  # noqa: E402


def arguments():
    import argparse
    p = argparse.ArgumentParser(
        description="YAPpingConformer Conformer + CTC training on real sentences "
                    "with boundary detection loss")
    p.add_argument("--sentences-csv", type=Path,
                   default=SEGMENTOR_DIR / "data_builder" / "sentences.csv")
    p.add_argument("--index-json", type=Path,
                   default=BACKEND_DIR / "data" / "tensors_sentence_level" / "index.json")
    p.add_argument("--classes", type=Path,
                   default=BACKEND_DIR / "models" / "word_vocabulary" / "classes.json")
    p.add_argument("--out-dir", type=Path,
                   default=SEGMENTOR_DIR / "models" / "yap_phase4")
    p.add_argument("--init", type=Path, default=None,
                   help="Pretrained YAP checkpoint (phase-3 best / stage-1 / stage-2). "
                        "If None, YAP backbone starts random.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-sentences", type=int, default=None,
                   help="Max sentences to use (None = all matched from sentences.csv)")
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--max-len", type=int, default=900)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--freeze-epochs", type=int, default=5,
                   help="Epochs to freeze YAP backbone (Conformer+head only)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--emission-w", type=float, default=1.0)
    p.add_argument("--blank-w", type=float, default=0.5)
    p.add_argument("--blank-target", type=float, default=0.85)
    p.add_argument("--boundary-w", type=float, default=1.0,
                   help="Weight for the boundary detection loss (end head BCE "
                        "against frame-level targets from sentences.csv)")
    p.add_argument("--segmentation-csv", type=Path,
                   default=SEGMENTOR_DIR / "data_builder" / "segmentation_clean.csv",
                   help="Human-annotated boundary marks CSV for validation. "
                        "Uses real frame_index marks as ground truth.")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def run_phase4(
    sentences_csv: Path | str,
    index_json: Path | str,
    classes_path: Path | str,
    out_dir: Path | str = "models/yap_phase4",
    epochs: int = 50,
    batch_size: int = 128,
    train_sentences: Optional[int] = None,
    val_split: float = 0.1,
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
    emission_w: float = 1.0,
    blank_w: float = 0.5,
    blank_target: float = 0.85,
    boundary_w: float = 1.0,
    segmentation_csv: Optional[Path | str] = None,
):
    """
    Run Conformer + CTC fine-tuning on real sentences from sentences.csv.

    Trains with combined loss: CTC (word sequence) + boundary_w * boundary BCE
    (end head vs frame-level targets from sentences.csv word positions).
    """
    return run_ctc_sentences(
        sentences_csv=sentences_csv,
        index_json=index_json,
        classes_path=classes_path,
        out_dir=out_dir,
        epochs=epochs,
        batch_size=batch_size,
        train_sentences=train_sentences,
        max_len=max_len,
        fps=fps,
        lr=lr,
        freeze_epochs=freeze_epochs,
        seed=seed,
        d_model=d_model,
        heads=heads,
        layers=layers,
        init=init,
        cpu=cpu,
        val_split=val_split,
        emission_w=emission_w,
        blank_w=blank_w,
        blank_target=blank_target,
        boundary_w=boundary_w,
        segmentation_csv=segmentation_csv,
    )


def main():
    args = arguments()
    best, history = run_phase4(
        sentences_csv=args.sentences_csv,
        index_json=args.index_json,
        classes_path=args.classes,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_sentences=args.train_sentences,
        val_split=args.val_split,
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
        emission_w=args.emission_w,
        blank_w=args.blank_w,
        blank_target=args.blank_target,
        boundary_w=args.boundary_w,
        segmentation_csv=args.segmentation_csv,
    )
    print(f"\n[DONE] Phase-4 best val loss: {best:.4f}. Checkpoints in {args.out_dir}")


if __name__ == "__main__":
    main()
