"""
yapv3.utils.synthetic_sentence_data — synthetic sentence data loader for
Conformer + CTC pretraining.

Behavior (per your spec):
  1. The FIRST word is loaded in full.
     Every SUBSEQUENT word: find the first frame where motion occurs
     (non-zero hand coords), then skip FORWARD 5-7 frames from there — so
     the hands never reset to rest between words, mimicking real continuous
     signing where the hands go straight into the next sign.
  2. The COMPLETE synthetic sentence is then randomly accelerated /
     decelerated with a Gaussian time-warp so the model learns to segment
     relatively (not at a fixed pace).
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_word_clip(filepath: Path) -> torch.Tensor:
    """Load a word clip from .pt or .npy file."""
    if filepath.suffix.lower() == '.npy':
        arr = np.load(filepath)
        return torch.from_numpy(arr.astype(np.float32))
    else:
        return torch.load(filepath, map_location='cpu', weights_only=True).float()


def _first_motion_frame(coords: np.ndarray) -> int:
    """
    Return the index of the first frame where hand/body motion occurs.

    coords: (T, 159) — 53 joints x 3 coords.  Motion = any joint's coords
    deviate from the resting pose by more than a tiny epsilon.
    """
    T = coords.shape[0]
    if T == 0:
        return 0
    # Use frame-to-frame absolute difference as motion signal
    diff = np.abs(np.diff(coords, axis=0)).sum(axis=1)  # (T-1,)
    motion_frames = np.where(diff > 0.02)[0] + 1  # +1 because diff shifts by 1
    if len(motion_frames) == 0:
        return 0
    return int(motion_frames[0])


def _gaussian_time_warp(coords: np.ndarray, rng: random.Random,
                        sigma_s: float = 0.6, amp: float = 0.35,
                        fps: float = 30.0) -> np.ndarray:
    """
    Warp time with a Gaussian velocity bump applied to the WHOLE sentence.

    Speeds up then slows down (or vice versa) at a random point in the
    sentence, so the model learns to segment gestures regardless of pacing.
    Returns coords resampled onto the original frame grid.
    """
    T = coords.shape[0]
    t = np.arange(T, dtype=np.float64) / fps
    centre = rng.uniform(0.25, 0.75) * T / fps   # where the bump sits
    vel = 1.0 + amp * np.exp(-0.5 * ((t - centre) / sigma_s) ** 2)
    vel = np.clip(vel, 0.5, 2.0)
    warp = np.cumsum(vel) / fps
    warp = warp / warp[-1] * (T / fps)            # normalise to same duration
    out = np.empty_like(coords)
    for c in range(coords.shape[1]):
        out[:, c] = np.interp(t, warp, coords[:, c])
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SyntheticSentenceDataset(Dataset):
    """
    Builds synthetic sentences from per-class word clips.

    Each sample is a concatenation of N random words where:
      * word 0 is used in full
      * words 1..N-1 are cut at (first_motion_frame + random 5-7 frames)
        so the hands flow continuously from one sign into the next
      * the whole sentence is then Gaussian time-warped (accel/decel)

    Returns:
        pose:     (T, 163) float32 — raw 163-dim corpus tensor
        length:   int — valid frame count
        word_idx: (L,) int64 — 0-based class indices (CTC needs 1-based +1)
    """

    def __init__(self, data_root: Path | str, classes: list[str],
                 samples_per_epoch: int = 1200, is_train: bool = True,
                 max_samples_per_class: int = 15, n_words_range: tuple[int, int] = (1, 10),
                 skip_frames_range: tuple[int, int] = (5, 8),
                 fps: float = 30.0, seed: int = 42,
                 time_warp: bool = True, time_warp_prob: float = 0.8,
                 warp_sigma_s: float = 0.6, warp_amp: float = 0.35):
        self.data_root = Path(data_root)
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.samples_per_epoch = samples_per_epoch
        self.is_train = is_train
        self.n_words_range = n_words_range
        self.skip_frames_range = skip_frames_range
        self.fps = fps
        self.seed = seed
        self.time_warp = time_warp
        self.time_warp_prob = time_warp_prob
        self.warp_sigma_s = warp_sigma_s
        self.warp_amp = warp_amp
        self.epoch = 0

        # ---- Discover per-class word clips ----
        self.class_files: dict[str, list[Path]] = {}
        valid_classes = []
        for cls in classes:
            cls_dir = self.data_root / cls
            if cls_dir.is_dir():
                # Support both .pt and .npy word clips
                files = sorted(list(cls_dir.glob("*.pt")) + list(cls_dir.glob("*.npy")))
                if max_samples_per_class and len(files) > max_samples_per_class:
                    files = files[:max_samples_per_class]
                if files:
                    self.class_files[cls] = files
                    valid_classes.append(cls)
        self.valid_classes = valid_classes
        if not self.valid_classes:
            raise RuntimeError(f"No word clips found under {data_root} for classes {classes[:5]}...")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
        # Per-epoch RNG so each epoch builds different sentences
        rng = random.Random(self.seed + index * 1000003 +
                            (self.epoch * 1000000007 if self.is_train else 0))

        # 1. Pick n words (distinct, no repetition)
        n = rng.randint(*self.n_words_range)
        n_available = min(n, len(self.valid_classes))
        chosen_words = rng.sample(self.valid_classes, n_available)

        pieces = []
        labels = []

        for i, word in enumerate(chosen_words):
            filepath = rng.choice(self.class_files[word])
            x = _load_word_clip(filepath)
            T = x.shape[0]
            coords = x[:, :159].numpy().astype(np.float32)  # (T, 159)
            flags = x[:, 159:]                              # (T, 4)

            if i == 0:
                # First word: use in full
                piece_coords = coords
                piece_flags = flags
            else:
                # Subsequent words: find first motion frame, skip 5-7 forward
                first_motion = _first_motion_frame(coords)
                skip = rng.randint(*self.skip_frames_range)
                start = first_motion + skip
                if start >= T:
                    # Word too short — fall back to using the last frame
                    start = max(0, T - 1)
                piece_coords = coords[start:]
                piece_flags = flags[start:]

            pieces.append((piece_coords, piece_flags))
            labels.append(self.class_to_idx[word])

        # Concatenate all pieces (no rest frames between words)
        if pieces:
            all_coords = np.concatenate([p[0] for p in pieces], axis=0)
            all_flags = np.concatenate([p[1] for p in pieces], axis=0)
        else:
            all_coords = np.zeros((1, 159), dtype=np.float32)
            all_flags = np.zeros((1, 4), dtype=np.float32)

        full_tensor = np.concatenate([all_coords, all_flags], axis=1)  # (T, 163)

        # ---- Gaussian time warp on the WHOLE sentence ----
        if self.is_train and self.time_warp and rng.random() < self.time_warp_prob:
            full_tensor = _gaussian_time_warp(full_tensor, rng,
                                              sigma_s=self.warp_sigma_s,
                                              amp=self.warp_amp,
                                              fps=self.fps)

        T_final = full_tensor.shape[0]
        word_idx_t = torch.tensor(labels, dtype=torch.long)
        return torch.from_numpy(full_tensor).float(), T_final, word_idx_t


def collate_synthetic_batch(batch):
    """Collate a list of SyntheticSentenceDataset items.

    Returns:
        pose:     (B, T, 163) float32
        lengths:  (B,) int64
        targets:  (sum_L,) int64 — flattened CTC targets (0-based class ids)
        target_lengths: (B,) int64
    """
    xs, lengths, labels = zip(*batch)
    max_len = max(lengths)
    batch_size = len(xs)
    D = xs[0].shape[-1]

    x_padded = torch.zeros(batch_size, max_len, D, dtype=torch.float32)
    for i, x in enumerate(xs):
        x_padded[i, :lengths[i], :] = x

    lengths_t = torch.tensor(lengths, dtype=torch.long)
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    targets = torch.cat(labels)

    return x_padded, lengths_t, targets, target_lengths


def build_synthetic_dataset(data_root: Path | str, classes_path: Path | str,
                            samples_per_epoch: int = 1200, is_train: bool = True,
                            max_samples_per_class: int = 15,
                            n_words_range: tuple[int, int] = (1, 10),
                            skip_frames_range: tuple[int, int] = (5, 8),
                            fps: float = 30.0, seed: int = 42,
                            time_warp: bool = True) -> SyntheticSentenceDataset:
    """Convenience: load classes + build SyntheticSentenceDataset."""
    with open(classes_path, encoding="utf-8") as f:
        classes = json.load(f)
    return SyntheticSentenceDataset(
        data_root, classes, samples_per_epoch=samples_per_epoch,
        is_train=is_train, max_samples_per_class=max_samples_per_class,
        n_words_range=n_words_range, skip_frames_range=skip_frames_range,
        fps=fps, seed=seed, time_warp=time_warp)


__all__ = [
    "SyntheticSentenceDataset",
    "collate_synthetic_batch",
    "build_synthetic_dataset",
    "_first_motion_frame",
    "_gaussian_time_warp",
    "_load_word_clip",
]
