"""
yapv3.utils.data — data pipeline for YAP gesture segmentation.

Responsibilities:
  * Augmentations: mirror (L-R), shift (x/y), tilt (z-rotation), Gaussian accel/decel (time-warp)
  * Zero-coord handling: forward-fill (hold last valid value) when a coord suddenly goes 0
  * Sentence stitching: pick 2-5 random words from classes.json, concatenate with pauses
  * Dataset + collate for the phase-3 trainer
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Layout constants (163-dim corpus tensor: 53 joints x 3 coords + 4 flags)
# ---------------------------------------------------------------------------
HAND_DIM = 63          # 21 joints x 3
BODY_START = 126       # 4 joints x 3 = 12
BODY_DIM = 12
FACE_START = 138       # 7 joints x 3 = 21
FACE_DIM = 21
FLAGS_START = 159      # 4 flags
FLAGS_DIM = 4
TOTAL_DIM = 163

LHAND_JOINTS = slice(0, 21)      # joints 0..20  (left hand)
RHAND_JOINTS = slice(21, 42)     # joints 21..41 (right hand)
BODY_JOINTS = slice(42, 46)      # joints 42..45 (shoulders + elbows)
FACE_JOINTS = slice(46, 53)      # joints 46..52 (face)


# ===========================================================================
# Augmentation config
# ===========================================================================
@dataclass
class AugmentationConfig:
    """Knobs for on-the-fly gesture augmentation."""

    mirror: bool = True            # randomly mirror left-right
    mirror_prob: float = 0.5

    shift: bool = True             # random x/y translation
    shift_xy_std: float = 0.04     # std of random shift (normalised coords)

    tilt: bool = True              # random in-plane rotation
    tilt_deg_std: float = 6.0      # std of rotation angle (degrees)

    time_warp: bool = True         # Gaussian accel/decel (speed up / slow down)
    time_warp_prob: float = 0.5
    warp_sigma_s: float = 0.6      # width of the Gaussian velocity bump (seconds)
    warp_amp: float = 0.35         # max relative speed change (+/-)

    # Zero-coord handling
    hold_last_valid: bool = True   # forward-fill missing coords instead of zeros

    # Seed for reproducibility
    seed: int = 42


# ===========================================================================
# Augmentation primitives
# ===========================================================================
def mirror_left_right(coords: np.ndarray, rng: random.Random) -> np.ndarray:
    """Mirror the pose left<->right.

    Flips the x-coordinate of every joint and swaps the left/right hand
    blocks so a mirrored signer still has a sensible pose.
    coords: (T, J, 3)  — last axis is (x, y, z)
    """
    if not rng.random() < 1.0:  # caller decides probability; kept for symmetry
        return coords
    out = coords.copy()
    # Flip x
    out[..., 0] = -out[..., 0]
    # Swap left and right hands
    lh = out[:, LHAND_JOINTS, :].copy()
    rh = out[:, RHAND_JOINTS, :].copy()
    out[:, LHAND_JOINTS, :] = rh
    out[:, RHAND_JOINTS, :] = lh
    # Body: shoulders are at joints 42,43 (left, right) — swap them
    out[:, 42, :], out[:, 43, :] = out[:, 43, :].copy(), out[:, 42, :].copy()
    out[:, 44, :], out[:, 45, :] = out[:, 45, :].copy(), out[:, 44, :].copy()
    return out


def shift_xy(coords: np.ndarray, rng: random.Random, std: float = 0.04) -> np.ndarray:
    """Random x/y translation so the signer need not be centred."""
    out = coords.copy()
    dx = rng.gauss(0.0, std)
    dy = rng.gauss(0.0, std)
    out[..., 0] += dx
    out[..., 1] += dy
    return out


def tilt_z(coords: np.ndarray, rng: random.Random, deg_std: float = 6.0) -> np.ndarray:
    """Small in-plane rotation around z (camera axis)."""
    out = coords.copy()
    deg = rng.gauss(0.0, deg_std)
    rad = math.radians(deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    x = out[..., 0].copy()
    y = out[..., 1].copy()
    out[..., 0] = cos_a * x - sin_a * y
    out[..., 1] = sin_a * x + cos_a * y
    return out


def gaussian_time_warp(coords: np.ndarray, rng: random.Random,
                       sigma_s: float = 0.6, amp: float = 0.35,
                       fps: float = 30.0) -> np.ndarray:
    """Warp time with a Gaussian velocity bump.

    Speeds up then slows down (or vice versa) so the model learns to
    segment gestures that are not uniformly paced.  Returns coords
    resampled onto the original time grid.
    """
    T = coords.shape[0]
    t = np.arange(T, dtype=np.float64) / fps
    centre = rng.uniform(0.25, 0.75) * T / fps   # where the bump sits
    # Gaussian velocity multiplier: 1 + amp * exp(-(t-centre)^2 / (2 sigma^2))
    vel = 1.0 + amp * np.exp(-0.5 * ((t - centre) / sigma_s) ** 2)
    vel = np.clip(vel, 0.5, 2.0)
    # Cumulative warp map: new_time = integral of vel
    warp = np.cumsum(vel) / fps
    warp = warp / warp[-1] * (T / fps)   # normalise to same total duration
    # Resample coords onto original grid
    out = np.empty_like(coords)
    for j in range(coords.shape[1]):
        for c in range(3):
            out[:, j, c] = np.interp(t, warp, coords[:, j, c])
    return out


def hold_last_valid(pose: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Forward-fill coords that suddenly go zero.

    pose:  (T, J, 3)
    valid: (T, J) bool — True where the landmark is trusted
    Returns pose with missing coords replaced by the last valid value.
    """
    if not isinstance(pose, torch.Tensor):
        pose = torch.from_numpy(pose)
    out = pose.clone()
    T, J, _ = out.shape
    for j in range(J):
        last_valid = None
        for t in range(T):
            if valid[t, j]:
                last_valid = out[t, j].clone()
            elif last_valid is not None:
                out[t, j] = last_valid
    return out


# ===========================================================================
# Sentence stitching
# ===========================================================================
def load_classes(classes_path: Path | str) -> list[str]:
    with open(classes_path, encoding="utf-8") as f:
        return json.load(f)


def find_word_clips(data_root: Path | str, classes: list[str],
                    max_per_class: Optional[int] = None) -> dict[int, list[Path]]:
    """Scan word tensor dirs for .pt clips, grouped by class id.

    Looks in both tensors_isl_csltr_word_level and yap_word_level.
    """
    data_root = Path(data_root)
    label_map = {c: i for i, c in enumerate(classes)}
    result: dict[int, list[Path]] = {i: [] for i in range(len(classes))}

    # Candidate roots (order matters: yap_word_level has all 229 classes)
    roots = [data_root / "yap_word_level", data_root / "tensors_isl_csltr_word_level"]
    for root in roots:
        if not root.is_dir():
            continue
        for cls_dir in root.iterdir():
            if not cls_dir.is_dir():
                continue
            lbl = cls_dir.name
            if lbl not in label_map:
                continue
            cid = label_map[lbl]
            for pt in sorted(cls_dir.glob("*.pt")):
                result[cid].append(pt)
            if max_per_class is not None:
                result[cid] = result[cid][:max_per_class]

    # Drop empty classes
    return {k: v for k, v in result.items() if v}


def _parse_clip(path: Path, fps: float = 30.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one .pt word clip and return (coords, conf, valid) numpy arrays.

    coords: (T, 53, 3) float32
    conf:   (T, 53) float32
    valid:  (T, 53) bool
    """
    clip = torch.load(path, map_location="cpu", weights_only=True).float().numpy()
    T = clip.shape[0]
    coords = clip[:, :159].reshape(T, 53, 3).astype(np.float32)   # drop flags
    flags = clip[:, 159:]                                          # (T, 4)

    # Build valid mask: joint is valid if any of its 3 coords is non-zero
    joint_valid = (np.abs(coords).sum(axis=-1) > 0)  # (T, 53)
    # Use first flag channel as overall confidence proxy if present
    conf_block = np.ones((T, 53), dtype=np.float32)
    if flags.shape[1] >= 1:
        conf_block = np.clip(flags[:, 0:1].repeat(53, axis=1), 0, 1).astype(np.float32)

    return coords, conf_block, joint_valid


def stitch_random_words(clips_by_class: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]],
                        classes: list[str],
                        rng: random.Random,
                        n_words_range: tuple[int, int] = (2, 5),
                        max_len: int = 900,
                        fps: float = 30.0,
                        gap_range_s: tuple[float, float] = (0.1, 0.5),
                        lead_tail_s: tuple[float, float] = (0.3, 0.8)):
    """Stitch 2-5 random word clips into one "sentence" tensor.

    clips_by_class maps class id -> list of preloaded (coords, conf, valid) tuples.

    Returns:
        pose:      (T, J, 3) float32
        conf:      (T, J) float32 — all 1.0 (we generate conf from valid)
        valid:     (T, J) bool
        spans:     list of (start_s, end_s) for each word (ground truth boundaries)
        labels:    list of class ids for each word
    """
    n_words = rng.randint(*n_words_range)
    available_cids = list(clips_by_class.keys())
    chosen_cids = rng.sample(available_cids, min(n_words, len(available_cids)))

    pose_parts = []
    conf_parts = []
    valid_parts = []
    spans = []
    labels = []
    cursor = 0.0

    # Lead-in silence
    lead = rng.uniform(*lead_tail_s)
    lead_frames = int(lead * fps)
    if lead_frames > 0:
        J = 53
        zero_block = np.zeros((lead_frames, J, 3), dtype=np.float32)
        pose_parts.append(zero_block)
        conf_parts.append(np.zeros((lead_frames, J), dtype=np.float32))
        valid_parts.append(np.zeros((lead_frames, J), dtype=bool))
        cursor += lead

    for cid in chosen_cids:
        coords, conf_block, joint_valid = rng.choice(clips_by_class[cid])
        T = coords.shape[0]

        start_s = cursor
        end_s = cursor + T / fps
        spans.append((start_s, end_s))
        labels.append(cid)

        pose_parts.append(coords.astype(np.float32))
        conf_parts.append(conf_block)
        valid_parts.append(joint_valid)

        cursor = end_s

        # Inter-word gap (silence)
        gap = rng.uniform(*gap_range_s)
        gap_frames = int(gap * fps)
        if gap_frames > 0:
            J = 53
            zero_block = np.zeros((gap_frames, J, 3), dtype=np.float32)
            pose_parts.append(zero_block)
            conf_parts.append(np.zeros((gap_frames, J), dtype=np.float32))
            valid_parts.append(np.zeros((gap_frames, J), dtype=bool))
            cursor += gap

    # Tail silence
    tail = rng.uniform(*lead_tail_s)
    tail_frames = int(tail * fps)
    if tail_frames > 0:
        J = 53
        zero_block = np.zeros((tail_frames, J, 3), dtype=np.float32)
        pose_parts.append(zero_block)
        conf_parts.append(np.zeros((tail_frames, J), dtype=np.float32))
        valid_parts.append(np.zeros((tail_frames, J), dtype=bool))
        cursor += tail

    pose = np.concatenate(pose_parts, axis=0)
    conf = np.concatenate(conf_parts, axis=0)
    valid = np.concatenate(valid_parts, axis=0)

    # Trim to max_len
    max_frames = int(max_len * fps)
    if pose.shape[0] > max_frames:
        pose = pose[:max_frames]
        conf = conf[:max_frames]
        valid = valid[:max_frames]
        # Only keep fully observed spans
        spans = [(s, e) for s, e in spans if e <= max_frames / fps]
        labels = labels[:len(spans)]

    times = np.arange(pose.shape[0], dtype=np.float32) / fps
    return pose, conf, valid, times, spans, labels


# ===========================================================================
# Dataset
# ===========================================================================
class SentenceClipDataset(Dataset):
    """Dataset of stitched word-sentences for gesture-segmentation pretraining.

    Each item is a synthetic "sentence" of 2-5 random words stitched together
    with silences, optionally augmented.  Zero-coord frames are forward-filled
    (hold-last-valid) so jitter from missing landmarks does not break learning.

    All word clips are preloaded into RAM at init time so that __getitem__
    stitches from memory (no per-item disk I/O during training).
    """

    def __init__(self, clips_by_class: dict[int, list[Path]], classes: list[str],
                 count: int, fps: float = 30.0, max_len: int = 900,
                 n_words_range: tuple[int, int] = (2, 5),
                 aug: Optional[AugmentationConfig] = None, seed: int = 42,
                 training: bool = True):
        self.classes = classes
        self.count = count
        self.fps = fps
        self.max_len = max_len
        self.n_words_range = n_words_range
        self.aug = aug or AugmentationConfig(seed=seed)
        self.seed = seed
        self.training = training
        self.epoch = 0

        # ---- Preload all word clips into RAM (cid -> list of (coords, conf, valid)) ----
        # hold_last_valid is applied once per clip here, not per sentence, for speed.
        self.clips_by_class: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
        for cid, paths in clips_by_class.items():
            loaded = []
            for p in paths:
                coords, conf, valid = _parse_clip(p, fps=fps)
                # Forward-fill missing coords once at preload time
                coords_t = torch.from_numpy(coords)
                valid_t = torch.from_numpy(valid)
                coords_t = hold_last_valid(coords_t, valid_t)
                coords = coords_t.numpy()
                loaded.append((coords, conf, valid))
            self.clips_by_class[cid] = loaded
        total_clips = sum(len(v) for v in self.clips_by_class.values())
        print(f"[DATA] Preloaded {total_clips} word clips into RAM across "
              f"{len(self.clips_by_class)} classes")

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        # Per-epoch RNG so augmentations differ each epoch
        rng = random.Random(self.seed + index * 1000003 +
                            (self.epoch * 1000000007 if self.training else 0))

        pose, conf, valid, times, spans, labels = stitch_random_words(
            self.clips_by_class, self.classes, rng,
            n_words_range=self.n_words_range,
            max_len=self.max_len, fps=self.fps)

        # ---- Augmentations (training only) ----
        if self.training:
            if self.aug.mirror and rng.random() < self.aug.mirror_prob:
                pose = mirror_left_right(pose, rng)
            if self.aug.shift:
                pose = shift_xy(pose, rng, std=self.aug.shift_xy_std)
            if self.aug.tilt:
                pose = tilt_z(pose, rng, deg_std=self.aug.tilt_deg_std)
            if self.aug.time_warp and rng.random() < self.aug.time_warp_prob:
                pose = gaussian_time_warp(pose, rng,
                                          sigma_s=self.aug.warp_sigma_s,
                                          amp=self.aug.warp_amp,
                                          fps=self.fps)

        # Recompute valid after augmentation (keep original validity)
        valid_t = torch.from_numpy(valid)
        pose_t = torch.from_numpy(pose)
        conf_t = torch.from_numpy(conf)
        times_t = torch.from_numpy(times)

        return pose_t, conf_t, valid_t, times_t, spans, labels


def collate_sentence_batch(items):
    """Collate a list of SentenceClipDataset items into a padded batch.

    Returns:
        pose:   (B, T, J, 3)
        conf:   (B, T, J)
        valid:  (B, T, J) bool
        times:  (B, T)
        mask:   (B, T) bool — True for real frames
        spans:  list per batch item of (start_s, end_s)
        labels: list per batch item of class ids
    """
    B = len(items)
    T = max(item[0].shape[0] for item in items)
    J = items[0][0].shape[1]

    pose = torch.zeros(B, T, J, 3)
    conf = torch.zeros(B, T, J)
    valid = torch.zeros(B, T, J, dtype=torch.bool)
    times = torch.zeros(B, T)
    mask = torch.zeros(B, T, dtype=torch.bool)
    spans = []
    labels = []

    for b, (p, c, v, tt, sp, lb) in enumerate(items):
        n = p.shape[0]
        pose[b, :n] = p
        conf[b, :n] = c
        valid[b, :n] = v
        times[b, :n] = tt
        times[b, n:] = tt[-1] if n > 0 else 0.0
        mask[b, :n] = True
        spans.append(sp)
        labels.append(lb)

    return pose, conf, valid, times, mask, spans, labels


def build_sentence_dataset(data_root: Path | str, classes_path: Path | str,
                           count: int, fps: float = 30.0, max_len: int = 900,
                           n_words_range: tuple[int, int] = (2, 5),
                           aug: Optional[AugmentationConfig] = None,
                           seed: int = 42, training: bool = True,
                           max_per_class: Optional[int] = None):
    """Convenience: load classes, find clips, build dataset."""
    classes = load_classes(classes_path)
    clips_by_class = find_word_clips(data_root, classes, max_per_class=max_per_class)
    if not clips_by_class:
        raise RuntimeError(f"No word clips found under {data_root}. "
                           f"Checked yap_word_level and tensors_isl_csltr_word_level.")
    return SentenceClipDataset(clips_by_class, classes, count, fps=fps,
                               max_len=max_len, n_words_range=n_words_range,
                               aug=aug, seed=seed, training=training)
