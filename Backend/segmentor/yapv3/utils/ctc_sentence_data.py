"""
yapv3.utils.ctc_sentence_data — real-sentence dataset for Conformer + CTC fine-tuning.

Reads `data_builder/sentences.csv` to get the sentence list and video paths,
matches each sentence to its tensor in `data/tensors_sentence_level/index.json`
(by sentence text), and builds a (tensor, word_indices) dataset for CTC training.

Each tensor is a (T, 163) pose sequence. The CTC target is the in-scope word
indices in signing order (class 0 = blank, words are 1..num_words per ConformerConfig).
"""
from __future__ import annotations

import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
from utils.dataloader_sentence import remap_sentence_words  # noqa: E402


def _gaussian_bump(T: int, center: int, sigma: float = 3.0) -> np.ndarray:
    """Gaussian bump (T,) centered at frame `center`, integrates to ~1."""
    t = np.arange(T, dtype=np.float32)
    bump = np.exp(-0.5 * ((t - center) / sigma) ** 2)
    s = bump.sum()
    if s > 0:
        bump = bump / s
    return bump.astype(np.float32)


def _build_boundary_targets(T: int, n_words: int, fps: float = 30.0,
                            sigma_s: float = 0.1) -> np.ndarray:
    """
    Build frame-level boundary targets (T,) for a sentence with n_words words.

    Distributes n_words across T frames: word k ends at frame (k+1)*T/(n_words+1)
    for internal boundaries, plus a final end at the last frame.
    Each boundary is a Gaussian bump (sigma in seconds -> frames).
    """
    targets = np.zeros(T, dtype=np.float32)
    if n_words <= 0 or T <= 0:
        return targets

    sigma_frames = max(1.0, sigma_s * fps)

    # Internal boundaries: after each word except the last
    for k in range(1, n_words):
        center = int(k * T / (n_words + 1))
        center = max(0, min(T - 1, center))
        targets = np.maximum(targets, _gaussian_bump(T, center, sigma_frames))

    # Final end at the last frame
    targets = np.maximum(targets, _gaussian_bump(T, T - 1, sigma_frames))

    return targets


# ---------------------------------------------------------------------------
# Sentence index matching
# ---------------------------------------------------------------------------
def load_sentence_index(index_path: Path | str) -> list[dict]:
    """Load the sentence-level tensor index (list of dicts with tensor_path,
    sentence, in_scope_words, in_scope_indices, out_scope_words, frames)."""
    with open(index_path, encoding="utf-8") as f:
        return json.load(f)


def build_sentence_to_tensor(index: list[dict]) -> dict[str, dict]:
    """Map sentence text (lowercased, stripped) -> index entry.

    If multiple takes exist for the same sentence, the FIRST one is kept
    (deterministic).  All takes share the same sentence + word targets.
    """
    mapping: dict[str, dict] = {}
    for entry in index:
        key = entry.get("sentence", "").strip().lower()
        if key and key not in mapping:
            mapping[key] = entry
    return mapping


def load_sentences_csv(csv_path: Path | str) -> list[dict]:
    """Read sentences.csv -> list of {sentence_id, sentence, sentence_video, source, is_composed}."""
    rows = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def match_sentences_to_tensors(
    sentences_csv: Path | str,
    index_json: Path | str,
    max_sentences: Optional[int] = None,
    seed: int = 42,
) -> list[dict]:
    """Match sentences.csv entries to index.json tensor entries.

    Returns list of dicts with:
        sentence_id, sentence, sentence_video, source, is_composed,
        tensor_path, in_scope_words, in_scope_indices, out_scope_words, frames
    Only sentences that have a matching tensor entry are returned.
    """
    rows = load_sentences_csv(sentences_csv)
    index = load_sentence_index(index_json)
    index_by_sentence = build_sentence_to_tensor(index)

    matched = []
    for row in rows:
        sentence = row.get("sentence", "").strip()
        key = sentence.lower()
        if key not in index_by_sentence:
            continue
        entry = index_by_sentence[key]
        matched.append({
            "sentence_id": row.get("sentence_id", ""),
            "sentence": sentence,
            "sentence_video": row.get("sentence_video", ""),
            "source": row.get("source", ""),
            "is_composed": row.get("is_composed", "0"),
            "tensor_path": entry.get("tensor_path", ""),
            "in_scope_words": entry.get("in_scope_words", []),
            "in_scope_indices": entry.get("in_scope_indices", []),
            "out_scope_words": entry.get("out_scope_words", []),
            "frames": entry.get("frames", 0),
        })

    # Deterministic shuffle + optional cap
    rng = random.Random(seed)
    rng.shuffle(matched)
    if max_sentences is not None:
        matched = matched[:max_sentences]

    return matched


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class RealSentenceCTCDataset(Dataset):
    """Dataset of real sentence tensors with CTC word targets.

    Each item: (pose_tensor, length, word_indices_tensor)
      pose_tensor:  (T, 163) float32 — the raw 163-dim corpus tensor
      length:       int — number of valid frames
      word_indices: (L,) int64 — in-scope word class indices in signing order
                            (values in 1..num_words; 0 is reserved for blank)
    """

    def __init__(
        self,
        matched: list[dict],
        classes: list[str],
        max_len: int = 900,
        fps: float = 30.0,
        seed: int = 42,
        training: bool = True,
        drop_out_of_scope: bool = True,
    ):
        self.matched = matched
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}  # 0-based class index
        self.max_len = max_len
        self.fps = fps
        self.seed = seed
        self.training = training
        self.drop_out_of_scope = drop_out_of_scope
        self.epoch = 0
        self._cache: dict[int, tuple[torch.Tensor, int, torch.Tensor]] = {}

        # ---- Remap index.json indices (262-class vocab) -> classes.json indices (0-based) ----
        # remap_sentence_words expects items with 'in_scope_words' and 'out_scope_words'.
        remapped_items, _ = remap_sentence_words(
            matched, self.class_to_idx, max_out_scope=5, max_words=18,
            drop_sentence_on_unknown=False
        )
        # remap_sentence_words adds 'target_indices' (0-based class indices) to each item.

        # Pre-filter: drop sentences with no in-scope words (nothing to predict)
        self.valid_items = []
        for i, item in enumerate(remapped_items):
            indices = item.get("target_indices", [])
            if len(indices) == 0:
                continue
            # CTC needs 1-based indices (0 = blank)
            word_idx = [idx + 1 for idx in indices]
            self.valid_items.append((i, word_idx))

        if not self.valid_items:
            raise RuntimeError("No sentences with in-scope words found after filtering.")

    def __len__(self):
        return len(self.valid_items)

    def __getitem__(self, index):
        # Per-epoch RNG so augmentations differ each epoch (none for now, but ready)
        rng = random.Random(self.seed + index * 1000003 +
                            (self.epoch * 1000000007 if self.training else 0))

        if index in self._cache:
            return self._cache[index]

        item_idx, word_indices = self.valid_items[index]
        item = self.matched[item_idx]
        tensor_path = item["tensor_path"]

        # Load tensor
        seq = torch.load(tensor_path, map_location="cpu", weights_only=True).float()
        T = seq.shape[0]

        # Time-resample to max_len if too long (smooth linear interpolation)
        if T > self.max_len:
            seq = seq.T.unsqueeze(0)                      # (1, 163, T)
            seq = torch.nn.functional.interpolate(
                seq, size=self.max_len, mode="linear", align_corners=True
            ).squeeze(0).T                               # (max_len, 163)
            T = self.max_len

        # Drop-out hole filling: forward-fill coords that suddenly go zero
        seq = _fill_missing_temporal(seq)

        word_idx_t = torch.tensor(word_indices, dtype=torch.long)

        # ---- Frame-level boundary targets from sentence word count ----
        n_words = len(word_indices)
        boundary_np = _build_boundary_targets(T, n_words, fps=self.fps,
                                              sigma_s=0.1)
        boundary_t = torch.from_numpy(boundary_np)

        result = (seq, T, word_idx_t, boundary_t)
        self._cache[index] = result
        return result


def _fill_missing_temporal(seq: torch.Tensor) -> torch.Tensor:
    """Forward-fill coords that suddenly go zero (hold last valid value).

    seq: (T, 163) — first 159 are coords (53 joints x 3), last 4 are flags.
    """
    coords = seq[:, :159].view(-1, 53, 3)
    flags = seq[:, 159:]
    valid = (coords.abs().sum(dim=-1) > 0)  # (T, 53)
    out = coords.clone()
    T, J, _ = out.shape
    for j in range(J):
        last_valid = None
        for t in range(T):
            if valid[t, j]:
                last_valid = out[t, j].clone()
            elif last_valid is not None:
                out[t, j] = last_valid
    seq[:, :159] = out.view(-1, 159)
    return seq


def collate_ctc_batch(items):
    """Collate a list of RealSentenceCTCDataset items into a padded batch.

    Returns:
        pose:          (B, T, 163) float32
        lengths:       (B,) int64 — valid frame count per sequence
        targets:       (sum_L,) int64 — flattened CTC targets (1-based, 0 = blank)
        target_lengths: (B,) int64 — length of each target sequence
        boundary:      (B, T) float32 — frame-level boundary targets (1 at word ends)
    """
    B = len(items)
    T = max(item[1] for item in items)
    D = items[0][0].shape[1]

    pose = torch.zeros(B, T, D)
    lengths = torch.zeros(B, dtype=torch.long)
    target_flat = []
    target_lengths = []
    boundary = torch.zeros(B, T)

    for b, (seq, n, word_idx, boundary_t) in enumerate(items):
        pose[b, :n] = seq
        lengths[b] = n
        target_flat.extend(word_idx.tolist())
        target_lengths.append(len(word_idx))
        boundary[b, :n] = boundary_t[:n]

    return (
        pose,
        lengths,
        torch.tensor(target_flat, dtype=torch.long),
        torch.tensor(target_lengths, dtype=torch.long),
        boundary,
    )


def build_real_sentence_dataset(
    sentences_csv: Path | str,
    index_json: Path | str,
    classes_path: Path | str,
    count: Optional[int] = None,
    max_len: int = 900,
    fps: float = 30.0,
    seed: int = 42,
    training: bool = True,
    val_split: float = 0.1,
) -> tuple[RealSentenceCTCDataset, Optional[RealSentenceCTCDataset]]:
    """Build train (and optionally val) RealSentenceCTCDataset from sentences.csv.

    Splits by sentence (all takes of the same sentence stay on one side).
    """
    with open(classes_path, encoding="utf-8") as f:
        classes = json.load(f)

    matched = match_sentences_to_tensors(
        sentences_csv, index_json, max_sentences=count, seed=seed
    )
    if not matched:
        raise RuntimeError(
            f"No sentences matched between {sentences_csv} and {index_json}. "
            f"Check that sentence texts align."
        )

    # Group by sentence for split (all takes of same sentence on same side)
    by_sentence: dict[str, list[dict]] = {}
    for item in matched:
        by_sentence.setdefault(item["sentence"].strip().lower(), []).append(item)

    keys = sorted(by_sentence.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    n_val = max(1, int(round(len(keys) * val_split))) if len(keys) >= 2 else 0
    val_keys = set(keys[:n_val])
    train_keys = set(keys[n_val:])

    train_items = [it for k in sorted(train_keys) for it in by_sentence[k]]
    val_items = [it for k in sorted(val_keys) for it in by_sentence[k]] if val_keys else []

    train_ds = RealSentenceCTCDataset(
        train_items, classes, max_len=max_len, fps=fps, seed=seed, training=True
    )
    val_ds = (
        RealSentenceCTCDataset(val_items, classes, max_len=max_len, fps=fps,
                               seed=seed + 7, training=False)
        if val_items
        else None
    )

    print(f"[DATA] Real sentences: train={len(train_ds)} | val={len(val_ds) if val_ds else 0} "
          f"(sentences: train={len(train_keys)} | val={len(val_keys)})")
    print(f"[DATA] Vocabulary: {len(classes)} classes (0=blank, words=1..{len(classes)})")
    return train_ds, val_ds


def load_segmentation_marks(segmentation_csv: Path | str,
                            index_json: Path | str) -> list[dict]:
    """
    Load segmentation_clean.csv marks and match to tensors.

    Returns list of dicts with sentence, frame_index, tensor_path, etc.
    Only includes marks where frame_index < tensor_frames (usable for training).
    """
    segmentation_csv = Path(segmentation_csv)
    index_json = Path(index_json)

    with open(index_json, encoding="utf-8") as f:
        index = json.load(f)

    # Build sentence -> index entry mapping
    index_by_sentence: dict[str, dict] = {}
    for entry in index:
        s = entry.get("sentence", "").strip().lower()
        if s and s not in index_by_sentence:
            index_by_sentence[s] = entry

    marks = []
    with open(segmentation_csv, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("accepted", "0") != "1":
                continue
            sentence = row.get("sentence", "").strip()
            s_key = sentence.lower()
            if s_key not in index_by_sentence:
                continue

            entry = index_by_sentence[s_key]
            tensor_frames = entry.get("frames", 0)
            try:
                frame_index = int(row["frame_index"])
            except (ValueError, TypeError):
                continue

            # Only use marks within tensor bounds
            if frame_index >= tensor_frames or frame_index < 0:
                continue

            marks.append({
                "sentence": sentence,
                "sentence_hash": row.get("sentence_hash", ""),
                "sentence_id": row.get("sentence_id", ""),
                "sentence_video": row.get("sentence_video", ""),
                "frame_index": frame_index,
                "timestamp": float(row.get("timestamp", 0)) if row.get("timestamp") else 0.0,
                "tensor_path": entry.get("tensor_path", ""),
                "tensor_frames": tensor_frames,
                "in_scope_words": entry.get("in_scope_words", []),
                "in_scope_indices": entry.get("in_scope_indices", []),
            })

    return marks


def build_segmentation_boundary_items(
    marks: list[dict],
    classes: list[str],
    max_len: int = 900,
    fps: float = 30.0,
    sigma_s: float = 0.1,
) -> list[dict]:
    """
    Convert segmentation marks into training items with real boundary targets.

    Each item: dict with pose, length, word_indices, boundary, sentence, sentence_hash.
    boundary: (T,) float32 — gaussian-smoothed 1.0 at each marked boundary frame.
    """
    class_to_idx = {c: i for i, c in enumerate(classes)}
    items = []

    # Group marks by sentence_hash so multiple marks per sentence become multiple boundaries
    by_hash: dict[str, list[dict]] = {}
    for mark in marks:
        by_hash.setdefault(mark["sentence_hash"], []).append(mark)

    for sentence_hash, sentence_marks in by_hash.items():
        # Use the first mark's tensor info (all marks for same sentence share tensor)
        first_mark = sentence_marks[0]
        tensor_path = first_mark["tensor_path"]
        try:
            seq = torch.load(tensor_path, map_location="cpu", weights_only=True).float()
        except Exception:
            continue

        T = seq.shape[0]
        if T > max_len:
            seq = seq.T.unsqueeze(0)
            seq = torch.nn.functional.interpolate(
                seq, size=max_len, mode="linear", align_corners=True
            ).squeeze(0).T
            T = max_len

        seq = _fill_missing_temporal(seq)

        # Word indices (1-based for CTC, 0 = blank)
        indices = first_mark.get("in_scope_indices", [])
        word_idx = [idx + 1 for idx in indices if idx is not None]
        if len(word_idx) == 0:
            continue

        # Build boundary targets from ALL marks for this sentence
        boundary = np.zeros(T, dtype=np.float32)
        original_frames = first_mark["tensor_frames"]
        for mark in sentence_marks:
            fi = mark["frame_index"]
            # Rescale frame_index if tensor was truncated
            if original_frames > max_len:
                fi = int(fi * T / original_frames)
            fi = max(0, min(T - 1, fi))
            sigma_frames = max(1.0, sigma_s * fps)
            boundary = np.maximum(boundary, _gaussian_bump(T, fi, sigma_frames))

        word_idx_t = torch.tensor(word_idx, dtype=torch.long)
        boundary_t = torch.from_numpy(boundary)

        items.append({
            "pose": seq,
            "length": T,
            "word_indices": word_idx_t,
            "boundary": boundary_t,
            "sentence": first_mark["sentence"],
            "sentence_hash": sentence_hash,
            "n_marks": len(sentence_marks),
        })

    return items


def collate_segmentation_batch(items):
    """Collate segmentation batch items.

    Returns:
        pose:     (B, T, 163) float32
        lengths:  (B,) int64
        targets:  (sum_L,) int64 — flattened CTC targets
        target_lengths: (B,) int64
        boundary: (B, T) float32 — frame-level boundary targets
    """
    B = len(items)
    T = max(item["length"] for item in items)
    D = items[0]["pose"].shape[1]

    pose = torch.zeros(B, T, D)
    lengths = torch.zeros(B, dtype=torch.long)
    target_flat = []
    target_lengths = []
    boundary = torch.zeros(B, T)

    for b, item in enumerate(items):
        pose[b, :item["length"]] = item["pose"]
        lengths[b] = item["length"]
        target_flat.extend(item["word_indices"].tolist())
        target_lengths.append(len(item["word_indices"]))
        boundary[b, :item["length"]] = item["boundary"]

    return (
        pose,
        lengths,
        torch.tensor(target_flat, dtype=torch.long),
        torch.tensor(target_lengths, dtype=torch.long),
        boundary,
    )


__all__ = [
    "RealSentenceCTCDataset",
    "collate_ctc_batch",
    "build_real_sentence_dataset",
    "match_sentences_to_tensors",
    "load_segmentation_marks",
    "build_segmentation_boundary_items",
    "collate_segmentation_batch",
]
