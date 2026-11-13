"""
data_utils.py — Data loading, splitting, and synthetic sentence formation.

Kept modular so the training script doesn't get crammed with data logic.

Responsibilities:
  * Scan segmentor/data/<word>/<take>/word.pt clips
  * Cap each class at MAX_CLIPS_PER_CLASS (20)
  * Split 16 train / 4 val per class (no leakage: a clip is NEVER in both)
  * Build synthetic sentences by stitching word clips with gaps
  * Frame-skip logic: skip 5-9 frames after the first major-movement frame
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SEGMENTOR_DATA = Path(__file__).resolve().parent / "data"          # segmentor/data/<word>/<take>/
CLASSES_JSON = Path(__file__).resolve().parent.parent / "models" / "word_vocabulary" / "classes.json"

MAX_CLIPS_PER_CLASS = 20
TRAIN_CLIPS_PER_CLASS = 16
VAL_CLIPS_PER_CLASS = 4
JOINT_COUNT = 53          # corpus layout: lh(21) + rh(21) + arms(4) + face(7)


@dataclass
class ClipInfo:
    word: str
    class_idx: int
    take: str
    path: Path


def load_classes(path: Path = CLASSES_JSON) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return list(payload["classes"])
    return list(payload)


def discover_clips(
    data_root: Path = SEGMENTOR_DATA,
    classes: Optional[list[str]] = None,
    max_per_class: int = MAX_CLIPS_PER_CLASS,
    seed: int = 42,
) -> dict[int, list[ClipInfo]]:
    """
    Scan segmentor/data/ and return {class_idx: [ClipInfo, ...]}.

    Each class is capped at max_per_class clips. Clips are shuffled deterministically
    so the train/val split below is reproducible.
    """
    if classes is None:
        classes = load_classes()
    class_to_idx = {c: i for i, c in enumerate(classes)}
    rng = random.Random(seed)

    result: dict[int, list[ClipInfo]] = {}
    for cid, name in enumerate(classes):
        folder = data_root / name
        if not folder.is_dir():
            continue
        takes = sorted(p for p in folder.iterdir() if p.is_dir())
        rng.shuffle(takes)
        clips: list[ClipInfo] = []
        for take_dir in takes:
            word_pt = take_dir / "word.pt"
            if not word_pt.exists():
                continue
            try:
                obj = torch.load(word_pt, map_location="cpu", weights_only=True)
                if isinstance(obj, torch.Tensor) and obj.numel() % 163 == 0 and obj.numel() >= 163 * 4:
                    clips.append(ClipInfo(word=name, class_idx=cid, take=take_dir.name, path=word_pt))
            except (RuntimeError, ValueError, OSError, TypeError):
                continue
            if len(clips) >= max_per_class:
                break
        if clips:
            result[cid] = clips
    return result


def split_train_val(
    corpus: dict[int, list[ClipInfo]],
    train_per_class: int = TRAIN_CLIPS_PER_CLASS,
    val_per_class: int = VAL_CLIPS_PER_CLASS,
    seed: int = 42,
) -> tuple[dict[int, list[ClipInfo]], dict[int, list[ClipInfo]]]:
    """
    Split each class's clips into train / val. A clip is NEVER in both.

    train_per_class=16, val_per_class=4  =>  20 per class (the cap).
    """
    rng = random.Random(seed)
    train_corpus: dict[int, list[ClipInfo]] = {}
    val_corpus: dict[int, list[ClipInfo]] = {}

    for cid, clips in corpus.items():
        shuffled = list(clips)
        rng.shuffle(shuffled)
        n_train = min(train_per_class, len(shuffled))
        n_val = min(val_per_class, len(shuffled) - n_train)
        train_corpus[cid] = shuffled[:n_train]
        val_corpus[cid] = shuffled[n_train : n_train + n_val]

    return train_corpus, val_corpus


# ---------------------------------------------------------------------------
# Frame-skip logic: find first major-movement frame, then skip 5-9 frames
# ---------------------------------------------------------------------------

def find_first_major_movement(
    pose: np.ndarray,
    joints: int = JOINT_COUNT,
    threshold: float = 0.04,
) -> int:
    """
    Find the first frame where the hands move "majorly" (signing actually starts).

    pose: (T, joints, 3) numpy array
    Returns: index of the first frame where hand displacement exceeds `threshold`
             relative to the first frame. Returns 0 if never found.
    """
    T = pose.shape[0]
    if T < 2:
        return 0

    # Use the two hands (joints 0-20 left, 21-41 right) as movement indicators
    lh = pose[:, 0:21, :]   # (T, 21, 3)
    rh = pose[:, 21:42, :]  # (T, 21, 3)

    # Reference = frame 0 (rest pose)
    lh_ref = lh[0:1, :, :]   # (1, 21, 3)
    rh_ref = rh[0:1, :, :]   # (1, 21, 3)

    # Per-frame mean displacement from reference
    lh_disp = np.linalg.norm(lh - lh_ref, axis=-1).mean(axis=1)  # (T,)
    rh_disp = np.linalg.norm(rh - rh_ref, axis=-1).mean(axis=1)  # (T,)
    combined = np.maximum(lh_disp, rh_disp)                      # (T,)

    # Find first frame exceeding threshold
    for t in range(T):
        if combined[t] > threshold:
            return t
    return 0


def apply_frame_skip(
    pose: np.ndarray,
    skip_min: int = 5,
    skip_max: int = 9,
    joints: int = JOINT_COUNT,
    movement_threshold: float = 0.04,
    rng: Optional[random.Random] = None,
) -> np.ndarray:
    """
    Skip 5-9 frames AFTER the first major-movement frame.

    This simulates the lead-in before a sign becomes fully visible.
    The skipped frames are dropped; the remaining clip is returned.

    pose: (T, joints, 3)
    Returns: (T - skip, joints, 3) or original if not enough frames.
    """
    if rng is None:
        rng = random.Random(42)

    T = pose.shape[0]
    if T < skip_max + 2:
        return pose

    first_move = find_first_major_movement(pose, joints, movement_threshold)

    # Skip 5-9 frames starting from first_move
    skip = rng.randint(skip_min, skip_max)
    start = first_move + skip
    if start >= T:
        # Not enough frames after skipping; return a small tail
        return pose[T // 2 :]

    return pose[start:]


# ---------------------------------------------------------------------------
# Synthetic sentence formation
# ---------------------------------------------------------------------------

def build_synthetic_sentence(
    corpus: dict[int, list[ClipInfo]],
    rng: random.Random,
    fps: float = 30.0,
    max_words: int = 5,
    max_len: int = 300,
    apply_skip: bool = True,
) -> tuple[np.ndarray, list[tuple[float, float, int]]]:
    """
    Stitch random word clips into one synthetic sentence.

    Returns:
        raw: (T, joints, 3) float32 pose array
        spans: [(start_t, end_t, class_idx), ...] one per word in signing order
    """
    keys = list(corpus.keys())
    n_words = rng.randint(1, max_words)
    frames = []
    spans = []
    cursor = 0

    # Lead-in rest frames
    lead = rng.randint(10, 25)
    frames.append(np.zeros((lead, JOINT_COUNT, 3), dtype=np.float32))
    cursor += lead

    for k in range(n_words):
        cid = rng.choice(keys)
        clip_infos = corpus[cid]
        clip_info = rng.choice(clip_infos)

        # Load clip
        obj = torch.load(clip_info.path, map_location="cpu", weights_only=True).float()
        clip = obj.reshape(-1, 163)[:, : JOINT_COUNT * 3].reshape(-1, JOINT_COUNT, 3).numpy().copy()

        # Optional frame skip: drop 5-9 frames after first major movement
        if apply_skip:
            clip = apply_frame_skip(clip, 5, 9, JOINT_COUNT, 0.04, rng)

        # Inter-word gap (real pauses between signs)
        if k:
            gap = rng.choice((0, 0, rng.randint(2, 7), rng.randint(10, 25)))
            if gap:
                frames.append(np.zeros((gap, JOINT_COUNT, 3), dtype=np.float32))
                cursor += gap

        start_t = cursor / fps
        frames.append(clip)
        cursor += len(clip)
        end_t = cursor / fps
        spans.append((start_t, end_t, cid))

    raw = np.concatenate(frames, axis=0)
    if raw.shape[0] > max_len:
        raise ValueError(f"Sentence too long ({raw.shape[0]} > {max_len}); increase --max-len")

    return raw, spans


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SentenceDataset:
    """
    Synthetic sentence dataset over a train or val corpus split.

    Each __getitem__ builds a new random sentence on the fly.
    """

    def __init__(
        self,
        corpus: dict[int, list[ClipInfo]],
        joints: int = JOINT_COUNT,
        count: int = 400,
        fps: float = 30.0,
        max_words: int = 5,
        max_len: int = 300,
        seed: int = 42,
        training: bool = True,
        apply_skip: bool = True,
    ):
        self.corpus = corpus
        self.joints = joints
        self.count = count
        self.fps = fps
        self.max_words = max_words
        self.max_len = max_len
        self.seed = seed
        self.training = training
        self.apply_skip = apply_skip
        self.epoch = 0

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int):
        rng = random.Random(
            self.seed + index * 1000003 + (self.epoch * 1000000007 if self.training else 0)
        )
        for _ in range(20):
            try:
                raw, spans = build_synthetic_sentence(
                    self.corpus, rng, self.fps, self.max_words, self.max_len,
                    apply_skip=self.apply_skip,
                )
                break
            except ValueError:
                continue
        else:
            raise ValueError("Could not generate a sentence; increase --max-len")

        pose = torch.from_numpy(raw).float()
        T = len(pose)
        valid = torch.isfinite(pose).all(-1) & (pose.abs().sum(-1) > 0)
        pose = torch.nan_to_num(pose, nan=0.0, posinf=0.0, neginf=0.0)
        conf = valid.float()

        times = torch.arange(T, dtype=torch.float32) / self.fps

        # Per-frame presence / end targets from spans
        presence_t = torch.zeros(T, dtype=torch.float32)
        end_t = torch.zeros(T, dtype=torch.float32)
        word_labels = torch.zeros(len(spans), dtype=torch.long)
        start_indices = []
        end_indices = []
        for i, (s_t, e_t, cid) in enumerate(spans):
            s_idx = max(0, int(round(s_t * self.fps)))
            e_idx = min(T, int(round(e_t * self.fps)))
            if e_idx <= s_idx:
                continue
            presence_t[s_idx:e_idx] = 1.0
            end_t[e_idx - 1] = 1.0
            word_labels[i] = cid
            start_indices.append(s_idx)
            end_indices.append(e_idx)

        return {
            "pose": pose,
            "conf": conf,
            "valid": valid,
            "times": times,
            "presence_t": presence_t,
            "end_t": end_t,
            "word_labels": word_labels,
            "start_indices": start_indices,
            "end_indices": end_indices,
        }


def collate(items: list[dict]):
    B = len(items)
    T = max(it["pose"].shape[0] for it in items)
    J = items[0]["pose"].shape[1]

    pose_b = torch.zeros(B, T, J, 3)
    conf_b = torch.zeros(B, T, J)
    valid_b = torch.zeros(B, T, J, dtype=torch.bool)
    t_b = torch.zeros(B, T)
    mask_b = torch.zeros(B, T, dtype=torch.bool)

    pres_b = torch.zeros(B, T)
    end_b = torch.zeros(B, T)

    # Build Segment objects (teacher forcing)
    from yap_conformer import Segment  # noqa: E402

    all_segments: list[list[Segment]] = []
    counts = []
    for i in range(B):
        segs = []
        for s_idx, e_idx in zip(items[i]["start_indices"], items[i]["end_indices"]):
            if e_idx <= s_idx:
                continue
            segs.append(Segment(s_idx, e_idx, s_idx / 30.0, e_idx / 30.0, "given", 0.0))
        all_segments.append(segs)
        counts.append(len(segs))

    K = min(max(max(counts) if counts else 1, 1), 12)
    word_targets_b = torch.zeros(B, K, dtype=torch.long)
    word_mask_b = torch.zeros(B, K, dtype=torch.bool)

    for i, (segs, wl) in enumerate(zip(all_segments, items)):
        for j, s in enumerate(segs):
            if j < K:
                word_targets_b[i, j] = wl["word_labels"][j] if j < len(wl["word_labels"]) else 0
                word_mask_b[i, j] = True

    for i, it in enumerate(items):
        n = it["pose"].shape[0]
        pose_b[i, :n] = it["pose"]
        conf_b[i, :n] = it["conf"]
        valid_b[i, :n] = it["valid"]
        t_b[i, :n] = it["times"]
        mask_b[i, :n] = True
        pres_b[i, :n] = it["presence_t"][:n]
        end_b[i, :n] = it["end_t"][:n]

    return (
        pose_b, conf_b, valid_b, t_b, mask_b,
        pres_b, end_b, word_targets_b, word_mask_b, all_segments, K,
    )
