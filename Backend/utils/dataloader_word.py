"""
Word-level dataloader for the ISL landmark corpus.
==================================================

Loads the **top-most frequent** word classes from the word-level video tensors
(``Backend/data/tensors_include50_word_level``, one directory per class, one
``.pt`` clip per file) with a hard per-class sample budget.

Selection rule
--------------
1. Every class directory is scanned and its clip count is measured.
2. A class qualifies when it owns at least ``min_samples_per_class`` clips
   (default **14**).  A class that falls below that tier is **eliminated from
   the loader** - the clips stay on disk untouched, because they are still
   used by the pretrainer and by the other models.
3. A qualifying class contributes at most ``max_samples_per_class`` clips
   (default **15**) - the "15 +- 1" budget.
4. The surviving classes are ordered by descending clip count (the "top most
   frequent" classes), ties broken alphabetically, so the class index is
   deterministic across runs and across machines.

The corpus statistics (262 classes / 4273 clips) are::

    >= 21 clips :  52 classes      >= 15 clips : 135 classes
    >= 20 clips : 120 classes      >= 14 clips : 228 classes
    >= 17 clips : 125 classes      >= 10 clips : 231 classes

so the default 14/15 rule keeps **228 of the 262 classes** and drops the 34
rarest ones (10, 9, 8 and 4 clips) from the loader.

The synthetic ``[NOTHING]`` idle class is appended as one extra output class
(the classification head therefore has ``len(classes) + 1`` logits), exactly
like the historical ``scripts/dataset.py`` behaviour.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:  # package-relative import (Backend on sys.path)
    from utils.pose_augmentation import (
        AugmentConfig,
        DEFAULT_AUGMENT,
        NECK_ANCHOR_WORD,
        augment_pose_sequence,
        as_sequence,
        fill_missing_temporal,
    )
except ImportError:  # pragma: no cover - direct execution fallback
    from pose_augmentation import (  # type: ignore
        AugmentConfig,
        DEFAULT_AUGMENT,
        NECK_ANCHOR_WORD,
        augment_pose_sequence,
        as_sequence,
        fill_missing_temporal,
    )

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WORD_LEVEL_DIR_DEFAULT = os.path.join(BACKEND_DIR, "data", "tensors_include50_word_level")

# The PUBLISHED vocabulary of the project.  Every trainer, every evaluation
# script and the live inference path read this file, so the CTC blank id, the
# head size and the sentence targets always refer to the SAME class list.
# It is generated from the corpus with the selection rule below.
CANONICAL_CLASSES_PATH = os.path.join(BACKEND_DIR, "models", "word_vocabulary", "classes.json")

# --- selection budget ------------------------------------------------------ #
# The TIER is what balances the corpus: a class below it is eliminated.  The
# corpus itself only spans 14..22 clips per class above the tier, so the CAP
# does not buy any balance (1.07x at 15 clips, 1.43x at 20 clips, 1.57x with no
# cap at all) - it only buys less data.  Measured:
#
#   cap 14 -> 3192 clips     cap 18 -> 3699 clips
#   cap 15 -> 3327 clips     cap 20 -> 3941 clips   (+18% over 15)
#   cap 16 -> 3452 clips     cap 22 -> 3994 clips   (+20% over 15)
#
# So the cap is set just below the maximum the corpus offers: 20 clips per
# class, which is 3941 clips over 228 classes with a 1.43x spread.
MIN_SAMPLES_PER_CLASS = 14     # a class below this tier is eliminated
MAX_SAMPLES_PER_CLASS = 20     # hard budget of clips loaded per class

EXCLUDED_CLASS_NAMES = ("NONE", "[NOTHING]", "NOTHING")

NOTHING_CLASS_NAME = "[NOTHING]"
DEFAULT_MAX_LEN = 300          # frames

# --------------------------------------------------------------------------- #
# The hole-filling mode - ONE knob for the whole project.
# --------------------------------------------------------------------------- #
# "none"        : keep the corpus convention (an undetected modality owns a
#                 ZERO block).  This is the default, because the backbone was
#                 PRETRAINED on this encoding: filling the holes would change
#                 the input distribution the pretrained weights expect.
# "hold"        : carry the last valid value of the landmark forward.
# "interpolate" : linear interpolation between the surrounding detections.
#
# The same value is used by the word loader AND by the word clips inside the
# sentence loader, so a model never sees the same clip encoded two ways.
FILL_MISSING_DEFAULT = "none"


# --------------------------------------------------------------------------- #
# Class discovery
# --------------------------------------------------------------------------- #
@dataclass
class WordClassInfo:
    """One class of the word corpus: its name, its clips and how many it owns."""

    name: str
    files: List[str] = field(default_factory=list)
    total_clips: int = 0        # clips on disk (before the budget is applied)

    @property
    def loaded_clips(self) -> int:
        return len(self.files)


def discover_word_classes(
    data_dir: str = WORD_LEVEL_DIR_DEFAULT,
    min_samples_per_class: int = MIN_SAMPLES_PER_CLASS,
    max_samples_per_class: int = MAX_SAMPLES_PER_CLASS,
    exclude: Sequence[str] = EXCLUDED_CLASS_NAMES,
) -> List[WordClassInfo]:
    """Scan the corpus and return the qualifying classes, most frequent first.

    Classes with fewer than ``min_samples_per_class`` clips are eliminated
    (they are NOT deleted from disk).  Every surviving class is capped at
    ``max_samples_per_class`` clips.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Word-level tensor directory not found: {data_dir}")

    excluded = {e.lower() for e in exclude}
    infos: List[WordClassInfo] = []

    for name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, name)
        if not os.path.isdir(class_dir):
            continue
        if name.lower() in excluded:
            continue

        clips = sorted(
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if f.lower().endswith((".pt", ".pth"))
        )
        if len(clips) < min_samples_per_class:
            # BELOW THE TIER -> eliminated from the loader (files stay on disk).
            continue

        infos.append(
            WordClassInfo(
                name=name,
                files=clips[:max_samples_per_class],
                total_clips=len(clips),
            )
        )

    # "top most frequent classes" -> descending clip count, alphabetical tie-break.
    infos.sort(key=lambda info: (-info.total_clips, info.name))
    return infos


def count_class_dirs(data_dir: str = WORD_LEVEL_DIR_DEFAULT,
                     exclude: Sequence[str] = EXCLUDED_CLASS_NAMES) -> int:
    """Number of class directories present on disk (before the tier filter)."""
    if not os.path.isdir(data_dir):
        return 0
    excluded = {e.lower() for e in exclude}
    return sum(
        1 for name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, name)) and name.lower() not in excluded
    )


def get_word_classes(
    data_dir: str = WORD_LEVEL_DIR_DEFAULT,
    min_samples_per_class: int = MIN_SAMPLES_PER_CLASS,
    max_samples_per_class: int = MAX_SAMPLES_PER_CLASS,
) -> List[str]:
    """The ordered list of class names the word loader will train on."""
    return [info.name for info in discover_word_classes(
        data_dir, min_samples_per_class, max_samples_per_class
    )]


def word_clip_index(
    data_dir: str = WORD_LEVEL_DIR_DEFAULT,
    min_samples_per_class: int = MIN_SAMPLES_PER_CLASS,
    max_samples_per_class: int = MAX_SAMPLES_PER_CLASS,
) -> Dict[str, List[str]]:
    """``class name -> list of clip paths`` for the selected classes.

    Used by the sentence dataloader, which trains on the very same word clips
    (as single-word "sentences") next to the continuous sentence tensors.
    """
    return {
        info.name: list(info.files)
        for info in discover_word_classes(
            data_dir, min_samples_per_class, max_samples_per_class
        )
    }


# --------------------------------------------------------------------------- #
# Train / val split
# --------------------------------------------------------------------------- #
def split_word_classes(
    infos: Sequence[WordClassInfo],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Deterministic per-class split into train and validation clips.

    The split is done on the FILE LIST (never on the tensor content), so no
    clip can appear in both partitions.  A class always keeps at least one
    validation clip and at least one training clip.
    """
    rng = random.Random(seed)
    train_items: List[Tuple[str, int]] = []
    val_items: List[Tuple[str, int]] = []

    for class_idx, info in enumerate(infos):
        files = list(info.files)
        rng.shuffle(files)
        n_val = int(round(len(files) * val_fraction))
        n_val = max(1, min(n_val, len(files) - 1)) if len(files) > 1 else 0
        val_items.extend((path, class_idx) for path in files[:n_val])
        train_items.extend((path, class_idx) for path in files[n_val:])

    return train_items, val_items


def load_split_tensors(
    items: Sequence[Tuple[str, int]],
    map_location: str = "cpu",
) -> Dict[int, List[torch.Tensor]]:
    """Load the clips of a split into memory: ``{class_index: [clip, ...]}``.

    Used by the phase trainers, which keep the whole (small) word corpus in RAM
    and apply their own temporal curriculum on top of it.
    """
    out: Dict[int, List[torch.Tensor]] = {}
    for path, label in items:
        try:
            clip = torch.load(path, map_location=map_location, weights_only=True).float()
        except Exception as exc:  # pragma: no cover - a corrupt file must not kill the run
            print(f"  [WARN] could not load {path}: {exc}")
            continue
        out.setdefault(label, []).append(clip)
    return out


# --------------------------------------------------------------------------- #
# The canonical split - ONE place, every phase reads it
# --------------------------------------------------------------------------- #
def canonical_split_items(
    data_dir: str = WORD_LEVEL_DIR_DEFAULT,
    min_samples_per_class: int = MIN_SAMPLES_PER_CLASS,
    max_samples_per_class: int = MAX_SAMPLES_PER_CLASS,
    path: Optional[str] = None,
    classes: Optional[Sequence[str]] = None,
    refresh: bool = False,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[str]]:
    """The train/val items of the CANONICAL manifest.

    Every phase of the pipeline calls this one function, so a clip that is
    training data in one phase is never validation data in another one.  The
    manifest is validated against the corpus: if the corpus changed (a class
    was re-extracted, the clip budget changed), the mismatch is reported
    instead of silently producing a different split.

    Returns ``(train_items, val_items, classes)``.
    """
    from .splits import (
        CANONICAL_SPLIT_PATH,
        load_canonical_split,
        refresh_canonical_split,
        split_fingerprint,
        split_items_from_manifest,
    )

    manifest_path = path or CANONICAL_SPLIT_PATH
    if refresh or not os.path.exists(manifest_path):
        manifest = refresh_canonical_split(
            data_dir, min_samples_per_class, max_samples_per_class, path=manifest_path)
    else:
        manifest = load_canonical_split(manifest_path)

    if classes is None:
        infos = discover_word_classes(
            data_dir, min_samples_per_class, max_samples_per_class
        )
        classes = [i.name for i in infos]

    class_to_idx = {c: i for i, c in enumerate(classes)}
    train_items, val_items, manifest_classes = split_items_from_manifest(
        manifest, class_to_idx)

    if set(manifest_classes) != set(classes):
        raise RuntimeError(
            f"[SPLIT] The canonical manifest holds {len(manifest_classes)} classes but the "
            f"corpus selection produced {len(classes)}.\n"
            f"        Regenerate it:  python scripts/save_phase4_split.py --canonical"
        )

    print(f"[SPLIT] Canonical manifest {split_fingerprint(manifest)[:12]}: "
          f"{len(train_items)} train / {len(val_items)} val clips, {len(classes)} classes.")
    return train_items, val_items, classes


def load_word_split_tensors(
    data_dir: str = WORD_LEVEL_DIR_DEFAULT,
    min_samples_per_class: int = MIN_SAMPLES_PER_CLASS,
    max_samples_per_class: int = MAX_SAMPLES_PER_CLASS,
    path: Optional[str] = None,
    refresh: bool = False,
) -> Tuple[Dict[int, List[torch.Tensor]], Dict[int, List[torch.Tensor]], List[str]]:
    """The canonical split, loaded into memory: ``(train_data, val_data, classes)``."""
    train_items, val_items, classes = canonical_split_items(
        data_dir, min_samples_per_class, max_samples_per_class, path=path, refresh=refresh)
    return (load_split_tensors(train_items),
            load_split_tensors(val_items),
            classes)


# --------------------------------------------------------------------------- #
# The synthetic [NOTHING] idle class
# --------------------------------------------------------------------------- #
def generate_nothing_sample(
    T: Optional[int] = None,
    neck: Sequence[float] = NECK_ANCHOR_WORD,
    rng: Optional[random.Random] = None,
) -> torch.Tensor:
    """Generate a realistic resting / idle / non-signing clip.

    The clip is expressed in the SAME geometric frame as the word corpus (the
    neck sits on the corpus median), otherwise the idle class would be a
    domain-shifted outlier that the classifier can recognise from the frame
    alone instead of from the pose.

    The face block is left empty and its presence flag cleared, because the
    face is effectively dead in the word corpus (present in 0.45% of frames).
    """
    rng = rng or random
    if T is None:
        T = rng.randint(25, 65)

    frames = np.zeros((T, 163), dtype=np.float32)
    nx, ny, nz = float(neck[0]), float(neck[1]), float(neck[2])

    # shoulders / elbows around the neck, one shoulder width apart
    base_arms = np.array([
        [nx - 0.50, ny + 0.20, nz],
        [nx + 0.50, ny + 0.20, nz],
        [nx - 0.55, ny + 0.65, nz],
        [nx + 0.55, ny + 0.65, nz],
    ], dtype=np.float32)

    t_phase = rng.uniform(0, 2 * np.pi)
    t_freq = rng.uniform(0.06, 0.12)
    sway = (0.012 * np.sin(np.arange(T) * t_freq + t_phase))[:, None, None].astype(np.float32)

    noise_arms = rng.gauss(0, 0) * np.ones((T, 4, 3), dtype=np.float32)
    noise_arms = np.random.normal(0, 0.006, (T, 4, 3)).astype(np.float32)

    arms_seq = base_arms[None, :, :] + sway + noise_arms
    frames[:, 126:138] = arms_seq.reshape(T, 12)
    # flags: hands absent, arms present, face absent
    frames[:, 159:163] = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    # In 20% of the idle samples the signer rests a hand on the lap.
    if rng.random() < 0.20:
        left = rng.random() < 0.5
        hand = np.random.normal(0, 0.04, (T, 21, 3)).astype(np.float32)
        hand += np.array([0.0, 0.40, 0.0], dtype=np.float32)
        if left:
            frames[:, 0:63] = hand.reshape(T, 63)
            frames[:, 159] = 1.0
        else:
            frames[:, 63:126] = hand.reshape(T, 63)
            frames[:, 160] = 1.0

    return torch.from_numpy(frames)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class ISLWordTensorDataset(Dataset):
    """Word-level dataset over the selected classes.

    Every item is ``(clip, length, label)`` where ``clip`` is a ``(T, 163)``
    MediaPipe landmark tensor and ``label`` is the class index inside the
    ordered class list of the loader.

    The augmentation set (mirror / tilt / bounded left-right shift / speed up
    or slow down / temporal shift / SpecAugment) is applied ON THE FLY and
    only on the training partition - the validation partition always sees the
    raw tensors.
    """

    def __init__(
        self,
        items: Sequence[Tuple[str, int]],
        class_to_idx: Dict[str, int],
        max_len: int = DEFAULT_MAX_LEN,
        is_train: bool = True,
        augment_config: Optional[AugmentConfig] = None,
        nothing_indices: Sequence[int] = (),
        nothing_seed: int = 1234,
        fill_missing: str = FILL_MISSING_DEFAULT,
    ):
        self.items = list(items)
        self.class_to_idx = dict(class_to_idx)
        self.max_len = max_len
        self.is_train = is_train
        self.augment_config = augment_config or DEFAULT_AUGMENT
        self.nothing_indices = list(nothing_indices)
        self.nothing_seed = nothing_seed
        self.fill_missing = fill_missing
        self._nothing_cache: Dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.items) + len(self.nothing_indices)

    def _nothing_clip(self, idx: int) -> torch.Tensor:
        if idx not in self._nothing_cache:
            rng = random.Random(self.nothing_seed + idx)
            self._nothing_cache[idx] = generate_nothing_sample(rng=rng)
        return self._nothing_cache[idx]

    def __getitem__(self, idx: int):
        if idx >= len(self.items):
            label = self.nothing_indices[idx - len(self.items)]
            clip = self._nothing_clip(idx - len(self.items))
        else:
            path, label = self.items[idx]
            clip = as_sequence(torch.load(path, weights_only=True))

        # Fill the drop-out holes with the last valid value BEFORE the
        # augmentation, so the geometric transforms work on a continuous
        # trajectory instead of on zero blocks.
        if self.fill_missing and self.fill_missing != "none":
            clip = fill_missing_temporal(clip, mode=self.fill_missing)

        if self.is_train and self.augment_config is not None:
            clip = augment_pose_sequence(clip, self.augment_config)

        if clip.shape[0] > self.max_len:
            clip = clip[: self.max_len]
        return clip, clip.shape[0], label


def collate_word_batch(batch):
    """Pad a batch of variable-length word clips to ``(B, T_max, 163)``."""
    clips, lengths, labels = zip(*batch)
    max_len = max(int(l) for l in lengths)
    x = torch.zeros(len(clips), max_len, clips[0].shape[-1], dtype=torch.float32)
    for i, clip in enumerate(clips):
        x[i, : clip.shape[0], :] = clip
    return x, torch.tensor([int(l) for l in lengths], dtype=torch.long), \
        torch.tensor([int(l) for l in labels], dtype=torch.long)


# --------------------------------------------------------------------------- #
# The loader factory
# --------------------------------------------------------------------------- #
def get_word_dataloaders(
    data_dir: str = WORD_LEVEL_DIR_DEFAULT,
    batch_size: int = 32,
    min_samples_per_class: int = MIN_SAMPLES_PER_CLASS,
    max_samples_per_class: int = MAX_SAMPLES_PER_CLASS,
    val_fraction: float = 0.2,
    max_len: int = DEFAULT_MAX_LEN,
    include_nothing_class: bool = True,
    nothing_samples: int = 0,
    augment_config: Optional[AugmentConfig] = None,
    fill_missing: str = FILL_MISSING_DEFAULT,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
    split_manifest: Optional[dict] = None,
    use_canonical_split: bool = True,
    save_meta_path: Optional[str] = None,
):
    """Build the word-level train / val dataloaders.

    Returns
    -------
    ``(train_loader, val_loader, class_to_idx, idx_to_class, classes)``

    ``classes`` are the GESTURE classes (without ``[NOTHING]``); the label
    index of ``[NOTHING]`` is ``len(classes)``, so a classification head needs
    ``len(classes) + 1`` logits.
    """
    infos = discover_word_classes(
        data_dir, min_samples_per_class, max_samples_per_class
    )
    if not infos:
        raise RuntimeError(
            f"No class in {data_dir} owns at least {min_samples_per_class} clips - "
            "lower --min_samples_per_class."
        )

    classes = [info.name for info in infos]
    class_to_idx = {name: i for i, name in enumerate(classes)}
    idx_to_class = {i: name for i, name in enumerate(classes)}

    train_items, val_items = split_word_classes(infos, val_fraction=val_fraction, seed=seed)

    # ---- the canonical manifest overrides the computed split ---------------
    # Every phase of the pipeline reads the SAME manifest, so a clip that is
    # training data in one phase is never validation data in another one.
    if use_canonical_split:
        if split_manifest is None:
            from .splits import load_canonical_split
            try:
                split_manifest = load_canonical_split()
            except FileNotFoundError as exc:
                print(f"[SPLIT] {exc}")
                print("[SPLIT] Falling back to the computed split for this run.")
                use_canonical_split = False

    if use_canonical_split or split_manifest is not None:
        from .splits import split_items_from_manifest
        train_items, val_items, manifest_classes = split_items_from_manifest(
            split_manifest, class_to_idx)
        if manifest_classes and set(manifest_classes) != set(classes):
            raise RuntimeError(
                f"[SPLIT] The canonical manifest holds {len(manifest_classes)} classes "
                f"but the corpus selection produced {len(classes)}.\n"
                f"        Regenerate the manifest:  python scripts/save_phase4_split.py --canonical"
            )
        print(f"[SPLIT] Using the canonical manifest "
              f"({len(train_items)} train / {len(val_items)} val clips).")

    # ---- the synthetic idle class ------------------------------------- #
    nothing_label = None
    train_nothing: List[int] = []
    val_nothing: List[int] = []
    if include_nothing_class:
        nothing_label = len(classes)
        class_to_idx[NOTHING_CLASS_NAME] = nothing_label
        idx_to_class[nothing_label] = NOTHING_CLASS_NAME
        n_nothing = nothing_samples or max_samples_per_class
        train_nothing = [nothing_label] * n_nothing
        val_nothing = [nothing_label] * max(1, n_nothing // 3)

    train_dataset = ISLWordTensorDataset(
        train_items, class_to_idx, max_len=max_len, is_train=True,
        augment_config=augment_config if augment_config is not None else DEFAULT_AUGMENT,
        nothing_indices=train_nothing, nothing_seed=seed,
        fill_missing=fill_missing,
    )
    val_dataset = ISLWordTensorDataset(
        val_items, class_to_idx, max_len=max_len, is_train=False,
        augment_config=None, nothing_indices=val_nothing, nothing_seed=seed + 9999,
        fill_missing=fill_missing,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_word_batch, num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_word_batch, num_workers=num_workers, pin_memory=pin_memory,
    )

    # ------------------------------------------------------------------ #
    print("=" * 78)
    print("ISL WORD-LEVEL DATALOADER (top-most frequent classes)")
    print("=" * 78)
    print(f"Corpus directory:            {data_dir}")
    print(f"Classes on disk:             {count_class_dirs(data_dir)}")
    print(f"Classes kept (>= {min_samples_per_class} clips): {len(infos)}")
    print(f"Per-class budget:            max {max_samples_per_class} clips")
    print(f"Total clips on disk (kept):  {sum(i.total_clips for i in infos)}")
    print(f"Total clips loaded:          {sum(i.loaded_clips for i in infos)}")
    print(f"Gesture classes:             {len(classes)}"
          + (f"  + [NOTHING] -> {len(classes) + 1} outputs" if include_nothing_class else ""))
    print(f"Train samples:               {len(train_dataset)}")
    print(f"Val samples:                 {len(val_dataset)}")
    print(f"Batch size:                  {batch_size}")
    print(f"Hole filling:                {fill_missing}")
    print("=" * 78)

    if save_meta_path:
        save_word_class_metadata(
            save_meta_path, classes, class_to_idx,
            data_dir=data_dir,
            min_samples_per_class=min_samples_per_class,
            max_samples_per_class=max_samples_per_class,
            val_fraction=val_fraction,
        )

    return train_loader, val_loader, class_to_idx, idx_to_class, classes


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def save_word_class_metadata(
    path: str,
    classes: Sequence[str],
    class_to_idx: Optional[Dict[str, int]] = None,
    **extra,
) -> str:
    """Persist the class list so every trainer and the inference code agree."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "classes": list(classes),
        "num_classes": len(classes),
        "class_to_idx": class_to_idx or {c: i for i, c in enumerate(classes)},
    }
    payload.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved word class metadata to: {path}")
    return path


def load_word_class_metadata(path: str) -> List[str]:
    """Read back a class list written by :func:`save_word_class_metadata`."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return list(payload["classes"])
    return list(payload)


# --------------------------------------------------------------------------- #
# The published vocabulary
# --------------------------------------------------------------------------- #
def refresh_canonical_classes(
    data_dir: str = WORD_LEVEL_DIR_DEFAULT,
    min_samples_per_class: int = MIN_SAMPLES_PER_CLASS,
    max_samples_per_class: int = MAX_SAMPLES_PER_CLASS,
    path: str = CANONICAL_CLASSES_PATH,
) -> List[str]:
    """Regenerate the published vocabulary from the corpus and write it out.

    The file is a PLAIN JSON list of class names, so that it is a drop-in
    replacement for every ``json.load(classes.json)`` in the repository.
    """
    infos = discover_word_classes(
        data_dir, min_samples_per_class, max_samples_per_class
    )
    classes = [i.name for i in infos]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)
    print(f"Published vocabulary ({len(classes)} classes) -> {path}")
    return classes


def canonical_classes(
    data_dir: str = WORD_LEVEL_DIR_DEFAULT,
    min_samples_per_class: int = MIN_SAMPLES_PER_CLASS,
    max_samples_per_class: int = MAX_SAMPLES_PER_CLASS,
    path: str = CANONICAL_CLASSES_PATH,
    refresh: bool = False,
) -> List[str]:
    """The vocabulary of the project - the ONE class list everybody uses.

    It is read from the published file when it exists (so every process sees
    exactly the same list) and derived from the corpus otherwise.
    """
    if refresh or not os.path.exists(path):
        return refresh_canonical_classes(
            data_dir, min_samples_per_class, max_samples_per_class, path
        )
    return load_word_class_metadata(path)
