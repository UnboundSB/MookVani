"""
THE canonical train/validation split of the word corpus.
=======================================================

Every phase of the word pipeline (pretrainer, phase 2, 3, 4, 5, 6 and the
word trainer) used to compute its OWN split with its own call to
``split_word_classes``.  They agreed only because they happened to use the
same seed and the same class list - and the moment one of them changed
(a different seed, a different clip budget, a re-extracted corpus) the phases
would silently disagree: a clip that was TRAINING data in phase 2 would be
VALIDATION data in phase 4, and the "validation accuracy" of phase 4 would
measure the model on data it had already been trained on.

This module owns ONE manifest:

    Backend/models/word_vocabulary/split.json

which pins, for every class, the exact list of training clips and the exact
list of validation clips.  Every phase reads it, and every phase records its
fingerprint in its metadata, so a mismatch is detected instead of ignored.

The manifest is also the place where the "same class" rule of the split is
enforced: the split is STRATIFIED PER CLASS, so a class that appears in the
training partition always appears in the validation partition too (with the
same class index), and no clip is ever in both.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

from .dataloader_word import (
    MAX_SAMPLES_PER_CLASS,
    MIN_SAMPLES_PER_CLASS,
    WORD_LEVEL_DIR_DEFAULT,
    discover_word_classes,
    split_word_classes,
)

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CANONICAL_SPLIT_PATH = os.path.join(BACKEND_DIR, "models", "word_vocabulary", "split.json")


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #
def build_canonical_split(
    data_dir: str = WORD_LEVEL_DIR_DEFAULT,
    min_samples_per_class: int = MIN_SAMPLES_PER_CLASS,
    max_samples_per_class: int = MAX_SAMPLES_PER_CLASS,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> dict:
    """Compute the split from the corpus with the shared selection rule."""
    infos = discover_word_classes(
        data_dir, min_samples_per_class, max_samples_per_class
    )
    train_items, val_items = split_word_classes(infos, val_fraction=val_fraction, seed=seed)

    classes = [i.name for i in infos]
    train: Dict[str, List[str]] = {c: [] for c in classes}
    val: Dict[str, List[str]] = {c: [] for c in classes}
    for path, label in train_items:
        train[classes[label]].append(path)
    for path, label in val_items:
        val[classes[label]].append(path)

    return {
        "meta": {
            "data_dir": os.path.abspath(data_dir),
            "min_samples_per_class": min_samples_per_class,
            "max_samples_per_class": max_samples_per_class,
            "val_fraction": val_fraction,
            "seed": seed,
            "num_classes": len(classes),
            "num_train_clips": sum(len(v) for v in train.values()),
            "num_val_clips": sum(len(v) for v in val.values()),
        },
        "classes": classes,
        "train": train,
        "val": val,
    }


def save_canonical_split(manifest: dict, path: str = CANONICAL_SPLIT_PATH) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    manifest = dict(manifest)
    manifest["meta"] = dict(manifest.get("meta", {}))
    manifest["meta"]["fingerprint"] = split_fingerprint(manifest)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[SPLIT] Canonical split written -> {path}")
    print(f"        classes={manifest['meta']['num_classes']} "
          f"train={manifest['meta']['num_train_clips']} "
          f"val={manifest['meta']['num_val_clips']} "
          f"fingerprint={manifest['meta']['fingerprint'][:12]}")
    return path


def refresh_canonical_split(
    data_dir: str = WORD_LEVEL_DIR_DEFAULT,
    min_samples_per_class: int = MIN_SAMPLES_PER_CLASS,
    max_samples_per_class: int = MAX_SAMPLES_PER_CLASS,
    val_fraction: float = 0.2,
    seed: int = 42,
    path: str = CANONICAL_SPLIT_PATH,
) -> dict:
    manifest = build_canonical_split(data_dir, min_samples_per_class,
                                     max_samples_per_class, val_fraction, seed)
    save_canonical_split(manifest, path)
    return manifest


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def load_canonical_split(path: str = CANONICAL_SPLIT_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[SPLIT] The canonical split manifest is missing: {path}\n"
            f"        Create it with:  python scripts/save_phase4_split.py --canonical"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_fingerprint(manifest: dict) -> str:
    """A stable hash of the ASSIGNMENT (not of the metadata).

    Two manifests with the same fingerprint assign exactly the same clips to
    the same partitions, whatever their metadata says.
    """
    h = hashlib.sha256()
    for key in ("classes", "train", "val"):
        payload = json.dumps(manifest.get(key, {}), sort_keys=True)
        h.update(payload.encode("utf-8"))
    return h.hexdigest()[:16]


def split_items_from_manifest(
    manifest: dict,
    class_to_idx: Optional[Dict[str, int]] = None,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[str]]:
    """Turn the manifest into the ``(path, label)`` item lists the loaders want.

    Returns ``(train_items, val_items, classes)`` where ``classes`` is the
    ordered class list of the manifest (so every phase uses the SAME class
    indices, and therefore the same meaning for the head of the model).
    """
    classes = list(manifest.get("classes") or [])
    if class_to_idx is None:
        class_to_idx = {c: i for i, c in enumerate(classes)}

    train_items: List[Tuple[str, int]] = []
    val_items: List[Tuple[str, int]] = []
    for name in classes:
        idx = class_to_idx.get(name)
        if idx is None:
            continue
        for path in manifest.get("train", {}).get(name, []):
            train_items.append((path, idx))
        for path in manifest.get("val", {}).get(name, []):
            val_items.append((path, idx))
    return train_items, val_items, classes


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def audit_split(manifest: dict) -> dict:
    """Check the manifest for the three ways a split can leak."""
    classes = list(manifest.get("classes") or [])
    train = manifest.get("train", {})
    val = manifest.get("val", {})

    train_files = [p for v in train.values() for p in v]
    val_files = [p for v in val.values() for p in v]
    train_set, val_set = set(train_files), set(val_files)

    # 1. no clip in both partitions
    overlap = sorted(train_set & val_set)
    # 2. every class is stratified: present in BOTH partitions
    missing_in_val = [c for c in classes if not val.get(c)]
    missing_in_train = [c for c in classes if not train.get(c)]
    # 3. every file belongs to the class directory it is filed under
    misfiled = []
    for name, paths in list(train.items()) + list(val.items()):
        for p in paths:
            if f"\\{name}\\" not in p and f"/{name}/" not in p:
                misfiled.append((name, p))

    return {
        "num_classes": len(classes),
        "num_train": len(train_files),
        "num_val": len(val_files),
        "overlap": overlap,
        "classes_without_val": missing_in_val,
        "classes_without_train": missing_in_train,
        "misfiled": misfiled,
        "fingerprint": split_fingerprint(manifest),
    }
