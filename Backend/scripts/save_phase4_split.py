"""Save the canonical word-corpus train/val split.

The manifest produced here is read by EVERY phase of the word pipeline
(pretrainer, phase 2, 3, 4, 5, 6 and the word trainer), so a clip that is
training data in one phase is never validation data in another one.

Usage
-----
    python scripts/save_phase4_split.py --canonical     # the shared manifest
    python scripts/save_phase4_split.py                 # the legacy Phase-4 file
"""
import os
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from utils.dataloader_word import (  # noqa: E402
    MAX_SAMPLES_PER_CLASS,
    MIN_SAMPLES_PER_CLASS,
    WORD_LEVEL_DIR_DEFAULT,
    discover_word_classes,
    split_word_classes,
)
from utils.splits import (  # noqa: E402
    CANONICAL_SPLIT_PATH,
    audit_split,
    refresh_canonical_split,
    save_canonical_split,
    split_fingerprint,
)

DATA_DIR = os.environ.get("ISL_WORD_DIR", WORD_LEVEL_DIR_DEFAULT)
P3_CLASSES = os.path.join(BACKEND_DIR, "models", "phase3_word_model", "classes.json")
OUT_FILE = os.path.join(BACKEND_DIR, "models", "phase4_word_model", "split_paths.json")

MAX_SAMPLES = MAX_SAMPLES_PER_CLASS
MIN_SAMPLES = MIN_SAMPLES_PER_CLASS
VAL_FRAC = 0.20
SEED = 42


def write_canonical():
    manifest = refresh_canonical_split(DATA_DIR, MIN_SAMPLES, MAX_SAMPLES, VAL_FRAC, SEED)
    report = audit_split(manifest)
    print(f"[SPLIT] audit: overlap={len(report['overlap'])} "
          f"classes_without_val={len(report['classes_without_val'])} "
          f"classes_without_train={len(report['classes_without_train'])} "
          f"misfiled={len(report['misfiled'])}")
    return manifest


def write_phase4_legacy():
    """The Phase-4 file consumed by Phase 5 and Phase 6."""
    manifest = refresh_canonical_split(DATA_DIR, MIN_SAMPLES, MAX_SAMPLES, VAL_FRAC, SEED)
    classes = manifest["classes"]
    split = {"classes": classes, "train": manifest["train"], "val": manifest["val"],
             "fingerprint": manifest["meta"]["fingerprint"]}

    tot_tr = sum(len(v) for v in split["train"].values())
    tot_va = sum(len(v) for v in split["val"].values())
    print(f"Classes: {len(classes)}  Train: {tot_tr}  Val: {tot_va}")

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    print(f"Split saved -> {OUT_FILE}")


import json  # noqa: E402  (kept after the helpers so the module docstring reads first)

if __name__ == "__main__":
    canonical = "--canonical" in sys.argv or "--all" in sys.argv
    write_canonical()
    if not canonical:
        write_phase4_legacy()
