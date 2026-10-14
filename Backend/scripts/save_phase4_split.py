"""Save Phase 4 train/val split (file paths) to JSON so Phase 5 can
load the EXACT same split with zero data leakage."""
import os, sys, glob, json, random

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BACKEND_DIR, "data", "tensors_include50_word_level")
P3_CLASSES  = os.path.join(BACKEND_DIR, "models", "phase3_word_model", "classes.json")
OUT_FILE    = os.path.join(BACKEND_DIR, "models", "phase4_word_model", "split_paths.json")

MAX_SAMPLES = 14
VAL_FRAC    = 0.20

random.seed(42)   # same seed as Phase 4

with open(P3_CLASSES, encoding="utf-8") as f:
    phase3_classes = json.load(f)
phase3_set = set(phase3_classes)

c2paths = {}
for cls in sorted(os.listdir(DATA_DIR)):
    p = os.path.join(DATA_DIR, cls)
    if not os.path.isdir(p): continue
    if cls not in phase3_set: continue
    pts = sorted(glob.glob(os.path.join(p, "*.pt")))  # sorted for reproducibility
    if not pts: continue
    random.shuffle(pts)          # Phase 4 shuffled before capping
    pts = pts[:MAX_SAMPLES]
    c2paths[cls] = pts

# Build ordered class list (same ordering as Phase 4)
classes = [c for c in phase3_classes if c in c2paths]
c2i = {c: i for i, c in enumerate(classes)}

split = {"classes": classes, "train": {}, "val": {}}

for cls, paths in c2paths.items():
    random.shuffle(paths)        # Phase 4 shuffled again before split
    nv = max(1, round(len(paths) * VAL_FRAC))
    split["val"][cls]   = paths[:nv]
    split["train"][cls] = paths[nv:]

tot_tr = sum(len(v) for v in split["train"].values())
tot_va = sum(len(v) for v in split["val"].values())
print(f"Classes: {len(classes)}  Train: {tot_tr}  Val: {tot_va}")

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(split, f, indent=2)
print(f"Split saved -> {OUT_FILE}")
