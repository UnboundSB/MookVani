"""
Dataset for word-level CTC training.

Loads (T, 163) .pt tensors from:
    Backend/data/tensors_include50_word_level/<class>/<video>.pt
    Backend/data/tensors_isl_csltr_word_level/<class>/<video>.pt

Maps folder name -> class index in classes.json (0..227).
Caps at MAX_PER_CLASS=20 samples per class to prevent overfitting.
Splits per class into train / val.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
CLASSES_PATH = BACKEND_DIR / "models" / "word_vocabulary" / "classes.json"
WORD_CORPORA = [
    BACKEND_DIR / "data" / "tensors_include50_word_level",
    BACKEND_DIR / "data" / "tensors_isl_csltr_word_level",
]

MAX_PER_CLASS = 20
VAL_RATIO = 0.2
SEED = 42


def load_classes() -> list:
    with open(CLASSES_PATH, encoding="utf-8") as f:
        return json.load(f)


class WordCTCDataset(Dataset):
    """
    Samples: (tensor (T,163) float32, class_idx int 0..227).
    class_idx is the CTC *word* label (1..228); blank=0 handled by loss.
    """

    def __init__(self, split: str = "train", max_per_class: int = MAX_PER_CLASS,
                 val_ratio: float = VAL_RATIO, seed: int = SEED):
        classes = load_classes()
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for i, c in enumerate(classes)}
        self.num_classes = len(classes)

        # gather samples per class
        per_class: dict[int, list[Path]] = {i: [] for i in range(self.num_classes)}
        for root in WORD_CORPORA:
            if not root.is_dir():
                continue
            for class_dir in root.iterdir():
                if not class_dir.is_dir():
                    continue
                ci = self.class_to_idx.get(class_dir.name)
                if ci is None:
                    continue
                pts = sorted(class_dir.rglob("*.pt"))
                random.seed(seed)
                random.shuffle(pts)
                pts = pts[:max_per_class]
                per_class[ci].extend(pts)

        # build train / val lists per class
        self.samples: list[Tuple[Path, int]] = []
        self.split = split
        for ci, paths in per_class.items():
            random.seed(seed)
            random.shuffle(paths)
            n_val = max(1, int(len(paths) * val_ratio)) if len(paths) > 1 else 0
            if split == "train":
                chosen = paths[n_val:]
            else:
                chosen = paths[:n_val]
            for p in chosen:
                self.samples.append((p, ci))

        print(f"[WORD_CTC] {split}: {len(self.samples)} samples, "
              f"{self.num_classes} classes, cap={max_per_class}/class")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, class_idx = self.samples[idx]
        x = torch.load(path, weights_only=True).float()  # (T, 163)
        return x, class_idx


def collate_fn(batch):
    """Pad tensors to max T in batch. Returns (padded (B,T,163), lengths (B), labels (B))."""
    tensors = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    lengths = torch.tensor([t.shape[0] for t in tensors], dtype=torch.long)
    T_max = int(lengths.max())
    padded = torch.zeros(len(tensors), T_max, 163, dtype=torch.float32)
    for i, t in enumerate(tensors):
        padded[i, : t.shape[0]] = t
    return padded, lengths, labels
