"""Save the word-level train/val split (file paths) as JSON.

Phase 5 and Phase 6 load the EXACT same split from this file, so there is
zero data leakage between the phases.

The split is produced by the SHARED selection rule of
``Backend/utils/dataloader_word.py``:

  * a class that owns fewer than ``MIN_SAMPLES`` clips is eliminated from the
    loader (its files stay on disk - the pretrainer and the other models still
    use them),
  * every surviving class contributes at most ``MAX_SAMPLES`` clips,
  * the classes are ordered by descending clip count (the "top most frequent"
    classes),
  * the train/val split is done per class on the FILE LIST, deterministically.
"""
import os
import sys
import json

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

DATA_DIR = os.environ.get("ISL_WORD_DIR", WORD_LEVEL_DIR_DEFAULT)
P3_CLASSES = os.path.join(BACKEND_DIR, "models", "phase3_word_model", "classes.json")
OUT_FILE = os.path.join(BACKEND_DIR, "models", "phase4_word_model", "split_paths.json")

MAX_SAMPLES = MAX_SAMPLES_PER_CLASS
MIN_SAMPLES = MIN_SAMPLES_PER_CLASS
VAL_FRAC = 0.20
SEED = 42

infos = discover_word_classes(
    DATA_DIR, min_samples_per_class=MIN_SAMPLES, max_samples_per_class=MAX_SAMPLES
)

# Keep the class order of the previous phase when it is available, so the head
# of the warm-started checkpoint keeps its meaning.
phase3_classes = None
if os.path.exists(P3_CLASSES):
    with open(P3_CLASSES, encoding="utf-8") as f:
        payload = json.load(f)
    phase3_classes = payload.get("classes", []) if isinstance(payload, dict) else payload
    order = {c: i for i, c in enumerate(phase3_classes)}
    infos = [i for i in infos if i.name in order]
    infos.sort(key=lambda i: order[i.name])

classes = [i.name for i in infos]
train_items, val_items = split_word_classes(infos, val_fraction=VAL_FRAC, seed=SEED)

split = {"classes": classes, "train": {}, "val": {}}
for path, label in train_items:
    split["train"].setdefault(classes[label], []).append(path)
for path, label in val_items:
    split["val"].setdefault(classes[label], []).append(path)

tot_tr = sum(len(v) for v in split["train"].values())
tot_va = sum(len(v) for v in split["val"].values())
print(f"Classes: {len(classes)}  Train: {tot_tr}  Val: {tot_va}")

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(split, f, indent=2)
print(f"Split saved -> {OUT_FILE}")
