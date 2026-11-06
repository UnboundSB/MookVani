"""
Word-level CTC training package for YAP segmentor vocabulary (228 classes).

Files:
    dataset.py   - load .pt word tensors, map folder->class index, cap at 20 per class
    model.py     - Conformer-CTC over 163-dim input, 229 outputs (blank + 228 words)
    loss.py      - anti-collapse CTC loss (blank floor + emission barrier + blank cap)
    train.py     - training loop with early stopping
    eval.py      - top-1 / top-5 accuracy on held-out split
"""

from .dataset import WordCTCDataset, collate_fn
from .model import ConformerCTC
from .loss import AntiCollapseCTCLoss
from .train import train_word_ctc
from .eval import evaluate_word_ctc

__all__ = [
    "WordCTCDataset", "collate_fn",
    "ConformerCTC", "AntiCollapseCTCLoss",
    "train_word_ctc", "evaluate_word_ctc",
]
