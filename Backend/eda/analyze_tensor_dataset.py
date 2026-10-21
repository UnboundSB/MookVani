"""
Tensor Dataset Integrity & MediaPipe Collapse Audit
===================================================
Audits the *finished* 163-dim MediaPipe landmark tensor datasets (word level and
sentence level) to determine whether upstream MediaPipe extraction collapsed onto
the data itself.  A collapsed feature space is the classic reason a model trains to
"garbage" (or, for CTC, collapses onto the blank token) no matter how good the
training recipe is.

Feature layout (canonical, see Backend/scripts/extraction.py)
--------------------------------------------------------------
    [  0: 63] left hand   (21 landmarks x 3)
    [ 63:126] right hand  (21 landmarks x 3)
    [126:138] arms        ( 4 landmarks x 3)   shoulders + elbows
    [138:159] face        ( 7 landmarks x 3)   eyes, nose, cheeks, mouth
    [159:163] presence flags  [lh, rh, arms, face]

Detection families
------------------
A. Frame-level degeneracy
   - dead frames          : every landmark coordinate is exactly 0
   - frozen frames        : bitwise identical to the previous frame
   - placeholder mismatch : missing-modality blocks stored as raw zeros (legacy
                            extraction) instead of the canonical resting pose
   - non-finite values    : NaN / +-Inf
   - out-of-range values  : coordinates beyond plausible normalised bounds

B. Modality-level collapse
   - presence rate of each of the four MediaPipe modalities
   - flag/geometry inconsistency (flag says present but geometry is all-zero)
   - stale carry-forward (flag says absent but geometry is non-zero)

C. Sample-level collapse
   - constant samples (every frame identical)
   - motion energy (mean inter-frame displacement)
   - longest static run

D. Discriminability probe  (the decisive test)
   - nearest-centroid and logistic-probe accuracy vs. chance level
   - between-class / within-class scatter ratio
   - PCA effective rank (participation ratio)
   - exact duplicate tensors across different classes (leakage)

If the probe sits at chance level the feature space is collapsed and no training
recipe can recover the lost signal; the fix must happen at extraction time.

Usage
-----
    python Backend/eda/analyze_tensor_dataset.py --dataset word
    python Backend/eda/analyze_tensor_dataset.py --dataset sentence
    python Backend/eda/analyze_tensor_dataset.py --dataset all
"""

import os
import sys
import json
import glob
import hashlib
import argparse
from collections import defaultdict

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

EDA_DIR = os.path.join(BACKEND_DIR, "eda")
PLOTS_DIR = os.path.join(EDA_DIR, "plots")
DATA_DIR = os.path.join(BACKEND_DIR, "data")

# Canonical landmark block boundaries inside the 163-dim vector
BLOCKS = {
    "lh":   (0, 63),
    "rh":   (63, 126),
    "arms": (126, 138),
    "face": (138, 159),
}
FLAG_IDX = {"lh": 159, "rh": 160, "arms": 161, "face": 162}

# Canonical resting pose written by extraction.py for an undetected modality
# (already divided by the per-frame scale, so it is scale invariant).
RESTING_POSE = {
    "lh":   np.array([-1.5, 3.0, 0.0]),
    "rh":   np.array([+1.5, 3.0, 0.0]),
}

# A normalised landmark should live roughly inside this box.
COORD_LIMIT = 12.0

DATASETS = {
    "word": {
        "name": "tensors_include50_word_level",
        "path": os.path.join(DATA_DIR, "tensors_include50_word_level"),
        "layout": "class_dirs",
    },
    "word_cslrt": {
        "name": "tensors_isl_csltr_word_level",
        "path": os.path.join(DATA_DIR, "tensors_isl_csltr_word_level"),
        "layout": "class_dirs",
    },
    "sentence": {
        "name": "tensors_sentence_level",
        "path": os.path.join(DATA_DIR, "tensors_sentence_level"),
        "layout": "index",
    },
}


# =============================================================================
# 1. PER-TENSOR ANALYSIS
# =============================================================================
def _static_stats(arr):
    """Return (longest_frozen_run_frames, frozen_transition_ratio, frozen_transitions)."""
    if arr.shape[0] < 2:
        return 0, 0.0, 0
    same = np.all(arr[1:] == arr[:-1], axis=1)
    d = np.diff(np.concatenate(([0], same.astype(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    longest = int((ends - starts).max()) if len(starts) else 0
    return longest + 1, float(same.mean()), int(same.sum())


def _is_resting_placeholder(block):
    """True when every landmark of the block equals the canonical resting pose."""
    pts = block.reshape(-1, 3)
    if not np.allclose(pts[:, 2], 0.0, atol=1e-4):
        return False
    return bool(
        np.allclose(pts[:, 0], RESTING_POSE["lh"][0], atol=1e-3)
        or np.allclose(pts[:, 0], RESTING_POSE["rh"][0], atol=1e-3)
    )


def analyze_tensor(path, label, sample_meta=None):
    """Full integrity audit of a single (T, 163) landmark tensor."""
    rec = {
        "path": path,
        "label": label,
        "frames": 0,
        "dim": 0,
        "load_error": None,
        "nan": 0,
        "inf": 0,
        "dead_frames": 0,
        "out_of_range_frames": 0,
        "max_abs_coord": 0.0,
        "longest_static_run": 0,
        "frozen_ratio": 0.0,
        "frozen_transitions": 0,
        "motion_energy": 0.0,
        "is_constant_sample": False,
        "flag_presence": {},
        "flag_geom_mismatch_present": 0,   # flag=1 but geometry all-zero
        "flag_geom_mismatch_absent": 0,    # flag=0 but geometry non-zero
        "zero_block_missing_modality": 0,  # absent modality stored as raw zeros
        "placeholder_missing_modality": 0, # absent modality stored as resting pose
        "raw_zero_modality_frames": 0,     # any modality block entirely zero
    }
    if sample_meta:
        rec.update(sample_meta)

    try:
        t = torch.load(path, weights_only=True)
    except Exception as exc:  # pragma: no cover - corrupt file path
        rec["load_error"] = repr(exc)
        return rec, None

    if not isinstance(t, torch.Tensor):
        arr = np.asarray(t, dtype=np.float32)
    else:
        arr = t.detach().cpu().numpy().astype(np.float32)

    if arr.ndim != 2:
        rec["load_error"] = f"unexpected ndim={arr.ndim}"
        return rec, None

    rec["frames"] = int(arr.shape[0])
    rec["dim"] = int(arr.shape[1])
    if arr.shape[1] != 163:
        rec["load_error"] = f"unexpected dim={arr.shape[1]} (expected 163)"
        return rec, None
    if arr.shape[0] == 0:
        rec["load_error"] = "empty tensor"
        return rec, None

    rec["nan"] = int(np.isnan(arr).sum())
    rec["inf"] = int(np.isinf(arr).sum())

    landmarks = arr[:, :159]
    flags = arr[:, 159:163]

    # --- A1. dead frames (total detection failure) ---
    dead = np.all(landmarks == 0.0, axis=1)
    rec["dead_frames"] = int(dead.sum())

    # --- A4. coordinate sanity ---
    finite = landmarks[np.isfinite(landmarks)]
    if finite.size:
        rec["max_abs_coord"] = float(np.abs(finite).max())
        oor = np.any(np.abs(landmarks) > COORD_LIMIT, axis=1)
        rec["out_of_range_frames"] = int(np.sum(oor & np.isfinite(landmarks).all(axis=1)))

    # --- A2. frozen frames ---
    longest, ratio, n_frozen = _static_stats(arr)
    rec["longest_static_run"] = longest
    rec["frozen_ratio"] = ratio
    rec["frozen_transitions"] = n_frozen

    # --- C. sample level motion ---
    if arr.shape[0] > 1:
        deltas = np.linalg.norm(landmarks[1:] - landmarks[:-1], axis=1)
        rec["motion_energy"] = float(np.mean(deltas))
        rec["is_constant_sample"] = bool(np.all(arr[1:] == arr[:-1]))
    else:
        rec["is_constant_sample"] = True

    # --- B. modality presence + flag/geometry consistency ---
    for name, (lo, hi) in BLOCKS.items():
        block = landmarks[:, lo:hi]
        flag = flags[:, FLAG_IDX[name] - 159]
        all_zero = np.all(block == 0.0, axis=1)
        rec["flag_presence"][name] = float(np.mean(flag > 0.5))
        rec["flag_geom_mismatch_present"] += int(np.sum((flag > 0.5) & all_zero))
        rec["flag_geom_mismatch_absent"] += int(np.sum((flag <= 0.5) & (~all_zero)))
        rec["raw_zero_modality_frames"] += int(np.sum(all_zero))
        absent = flag <= 0.5
        if absent.any():
            blk_absent = block[absent]
            zero_absent = np.all(blk_absent == 0.0, axis=1)
            rec["zero_block_missing_modality"] += int(zero_absent.sum())
            rec["placeholder_missing_modality"] += int(
                sum(1 for row in blk_absent[~zero_absent] if _is_resting_placeholder(row))
            )
    return rec, landmarks


# =============================================================================
# 2. DATASET LOADERS
# =============================================================================
def collect_class_dirs(root, limit=None):
    """Word-level layout: <root>/<class>/<sample>.pt"""
    items = []
    if not os.path.isdir(root):
        return items
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    for cls in classes:
        files = sorted(glob.glob(os.path.join(root, cls, "*.pt")))
        for f in files:
            items.append((f, cls))
    if limit:
        items = items[:limit]
    return items


def collect_index(root, limit=None):
    """Sentence-level layout: <root>/index.json pointing at tensor files."""
    items = []
    index_path = os.path.join(root, "index.json")
    if not os.path.exists(index_path):
        return items
    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for entry in data:
        tp = entry.get("tensor_path", "")
        if not tp:
            continue
        if not os.path.exists(tp):
            # tolerate stale absolute paths by rebuilding from the file name
            cand = os.path.join(root, os.path.basename(tp))
            if not os.path.exists(cand):
                continue
            tp = cand
        label = " ".join(entry.get("in_scope_words", [])) or entry.get("sentence", "")
        items.append((tp, label, entry))
    if limit:
        items = items[:limit]
    return items


# =============================================================================
# 3. DISCRIMINABILITY PROBE
# =============================================================================
def build_sample_features(landmarks_list):
    """Temporal summary: mean, std and net displacement of the 159 landmark dims."""
    feats = []
    for arr in landmarks_list:
        feats.append(np.concatenate([arr.mean(0), arr.std(0), arr[-1] - arr[0]]))
    return np.stack(feats).astype(np.float64)


def nearest_centroid_probe(X, y, test_frac=0.3, seed=7):
    """Pure-numpy stratified nearest-centroid probe (no sklearn dependency)."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    if len(classes) < 2:
        return None

    train_idx, test_idx = [], []
    for c in classes:
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        if len(idx) == 1:
            train_idx.append(idx[0])
            continue
        n_test = max(1, int(round(len(idx) * test_frac)))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    if len(test_idx) == 0 or len(train_idx) == 0:
        return None

    mu = X[train_idx].mean(0)
    sd = X[train_idx].std(0)
    sd[sd < 1e-8] = 1.0
    keep = X[train_idx].std(0) > 1e-6
    if keep.sum() < 2:
        return None
    Xtr = (X[train_idx][:, keep] - mu[keep]) / sd[keep]
    Xte = (X[test_idx][:, keep] - mu[keep]) / sd[keep]

    centroids, order = {}, []
    for c in classes:
        mask = y[train_idx] == c
        if mask.sum() == 0:
            continue
        centroids[c] = Xtr[mask].mean(0)
        order.append(c)
    if not order:
        return None

    C = np.stack([centroids[c] for c in order])
    dist = ((Xte[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    pred = np.array(order)[dist.argmin(1)]
    acc = float(np.mean(pred == y[test_idx]))

    per_class = {}
    for c in order:
        m = y[test_idx] == c
        if m.sum():
            per_class[str(c)] = float(np.mean(pred[m] == c))

    # scatter decomposition (separable signal fraction)
    gm = X[train_idx].mean(0)
    between = 0.0
    within = 0.0
    for c in order:
        mask = y[train_idx] == c
        if mask.sum() == 0:
            continue
        Xc = X[train_idx][mask]
        cm = Xc.mean(0)
        between += mask.sum() * float(((cm - gm) ** 2).sum())
        within += float(((Xc - cm) ** 2).sum())
    scatter_ratio = between / max(between + within, 1e-12)

    # PCA participation ratio (effective rank) on the standardised train matrix
    cov = np.cov(Xtr, rowvar=False)
    ev = np.linalg.eigvalsh(cov)
    ev = np.clip(ev, 0.0, None)
    total = ev.sum()
    if total > 0:
        p = ev / total
        eff_rank = float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0]))))
        top10 = float(ev[np.argsort(ev)[::-1][:10]].sum() / total)
    else:
        eff_rank, top10 = 0.0, 0.0

    return {
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_classes": int(len(classes)),
        "chance_accuracy": 1.0 / len(classes),
        "nearest_centroid_acc": acc,
        "lift_over_chance": acc - 1.0 / len(classes),
        "scatter_ratio": scatter_ratio,
        "pca_effective_rank": eff_rank,
        "pca_top10_var_ratio": top10,
        "per_class_recall": per_class,
    }


def logistic_probe(X, y, test_frac=0.3, seed=7):
    """Optional sklearn logistic-regression probe (stronger than nearest centroid)."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
    except Exception:
        return None
    classes = np.unique(y)
    if len(classes) < 2:
        return None
    counts = {c: int(np.sum(y == c)) for c in classes}
    rare = [c for c, n in counts.items() if n < 2]
    if rare:
        mask = ~np.isin(y, rare)
        X, y = X[mask], y[mask]
        classes = np.unique(y)
        if len(classes) < 2:
            return None
    keep = X.std(0) > 1e-6
    if keep.sum() < 2:
        return None
    X = X[:, keep]
    try:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_frac, random_state=seed, stratify=y
        )
        clf = LogisticRegression(max_iter=600, n_jobs=-1)
        clf.fit(Xtr, ytr)
        acc = float(clf.score(Xte, yte))
        return {
            "logistic_acc": acc,
            "chance_accuracy": 1.0 / len(classes),
            "lift_over_chance": acc - 1.0 / len(classes),
            "n_test": int(len(yte)),
        }
    except Exception as exc:  # pragma: no cover
        return {"error": repr(exc)}


# =============================================================================
# 4. DUPLICATE / LEAKAGE SCAN
# =============================================================================
def duplicate_scan(records):
    """Detect byte-identical tensors appearing under different labels."""
    by_hash = defaultdict(list)
    for rec in records:
        if rec.get("load_error"):
            continue
        try:
            with open(rec["path"], "rb") as f:
                h = hashlib.md5(f.read()).hexdigest()
        except OSError:
            continue
        by_hash[h].append(rec)
    dup_groups = [v for v in by_hash.values() if len(v) > 1]
    cross_label = [g for g in dup_groups if len({r["label"] for r in g}) > 1]
    return {
        "duplicate_groups": len(dup_groups),
        "cross_label_duplicate_groups": len(cross_label),
        "cross_label_duplicate_samples": int(sum(len(g) for g in cross_label)),
        "examples": [
            {"labels": sorted({r["label"] for r in g}), "paths": [r["path"] for r in g[:4]]}
            for g in cross_label[:10]
        ],
    }


def geometric_frame_report(records, landmarks_list):
    """Detect which geometric frame a corpus is expressed in.

    The corpora in Backend/data were extracted at different times with different
    normalisation conventions, so the same physical point (the neck = the shoulder
    midpoint) sits in a different place in each of them.  A backbone pretrained on one
    frame and fine-tuned on another sees a completely different skeleton, which is a
    severe domain shift.

    This reports the median neck position of the corpus and compares it with the other
    audited corpora.
    """
    necks = []
    for arr in landmarks_list:
        arms = arr[:, 126:138].reshape(-1, 4, 3)[:, :2, :]
        neck = (arms[:, 0] + arms[:, 1]) / 2.0
        necks.append(np.median(neck, axis=0))
    if not necks:
        return None
    necks = np.stack(necks)
    med = np.median(necks, axis=0)
    spread = np.percentile(np.linalg.norm(necks, axis=1), [10, 90])
    return {
        "neck_median": [round(float(v), 3) for v in med],
        "neck_offset_from_origin": round(float(np.linalg.norm(med)), 3),
        "neck_per_clip_p10_p90": [round(float(v), 3) for v in spread],
    }


def subcorpus_breakdown(records):
    """Break the audit down by the source corpus recorded in index.json.

    The continuous corpus is a merge of several sources, and they do not necessarily use
    the same extraction convention.  The ISL-CSLRT clips store an undetected hand as raw
    zeros while the iSign clips store the canonical resting pose, so the same physical
    state has two different numeric encodings inside a single dataset.
    """
    groups = defaultdict(list)
    for r in records:
        groups[r.get("dataset") or "unknown"].append(r)

    out = {}
    for key, recs in groups.items():
        ok = [r for r in recs if not r.get("load_error")]
        if not ok:
            continue
        total_frames = sum(r["frames"] for r in ok)
        presence = {}
        for m in ["lh", "rh", "arms", "face"]:
            presence[m] = round(100.0 * float(np.mean([r["flag_presence"][m] for r in ok])), 2)
        out[key] = {
            "samples": len(ok),
            "frames": total_frames,
            "modality_presence_pct": presence,
            "raw_zero_blocks": int(sum(r["zero_block_missing_modality"] for r in ok)),
            "resting_placeholder_blocks": int(sum(r["placeholder_missing_modality"] for r in ok)),
            "flag_absent_but_nonzero_geometry": int(sum(r["flag_geom_mismatch_absent"] for r in ok)),
            "dead_frames": int(sum(r["dead_frames"] for r in ok)),
            "motion_energy_median": float(np.median([r["motion_energy"] for r in ok])),
        }
    return out


# =============================================================================
# 5. SUMMARY
# =============================================================================
def _pct(values, q):
    if len(values) == 0:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize(records):
    ok = [r for r in records if not r.get("load_error")]
    errs = [r for r in records if r.get("load_error")]
    if not ok:
        return {"total": len(records), "load_errors": len(errs), "error_messages":
                sorted({r["load_error"] for r in errs})[:10]}

    frames = np.array([r["frames"] for r in ok], dtype=np.float64)
    motion = np.array([r["motion_energy"] for r in ok], dtype=np.float64)
    static_ratio = np.array([r["frozen_ratio"] for r in ok], dtype=np.float64)
    dead = np.array([r["dead_frames"] for r in ok], dtype=np.float64)
    const_samples = sum(1 for r in ok if r["is_constant_sample"])
    total_frames = int(frames.sum())

    per_label = defaultdict(list)
    for r in ok:
        per_label[r["label"]].append(r)

    label_counts = {k: len(v) for k, v in per_label.items()}
    counts = np.array(list(label_counts.values()), dtype=np.float64)

    presence = defaultdict(list)
    for r in ok:
        for k, v in r["flag_presence"].items():
            presence[k].append(v)

    return {
        "total": len(records),
        "load_errors": len(errs),
        "error_messages": sorted({r["load_error"] for r in errs})[:10],
        "n_classes": len(per_label),
        "total_frames": total_frames,
        "frames": {
            "min": int(frames.min()), "max": int(frames.max()),
            "mean": float(frames.mean()), "median": float(np.median(frames)),
            "p05": _pct(frames, 5), "p95": _pct(frames, 95),
        },
        "nan_total": int(sum(r["nan"] for r in ok)),
        "inf_total": int(sum(r["inf"] for r in ok)),
        "dead_frames_total": int(dead.sum()),
        "dead_frames_pct": 100.0 * float(dead.sum()) / max(1, total_frames),
        "constant_samples": int(const_samples),
        "constant_samples_pct": 100.0 * const_samples / max(1, len(ok)),
        "motion_energy": {
            "mean": float(motion.mean()), "median": float(np.median(motion)),
            "p05": _pct(motion, 5), "p95": _pct(motion, 95),
            "near_zero_samples": int(np.sum(motion < 1e-4)),
        },
        "frozen_ratio": {
            "mean": float(static_ratio.mean()), "median": float(np.median(static_ratio)),
            "p95": _pct(static_ratio, 95),
        },
        "max_abs_coord": float(max(r["max_abs_coord"] for r in ok)),
        "modality_presence": {
            k: {
                "mean_presence": float(np.mean(v)),
                "min_sample_presence": float(np.min(v)),
                "p05_sample_presence": _pct(v, 5),
                "samples_never_present": int(np.sum(np.asarray(v) <= 1e-9)),
            }
            for k, v in presence.items()
        },
        "flag_geometry_mismatch": {
            "flag_present_but_zero_geometry": int(sum(r["flag_geom_mismatch_present"] for r in ok)),
            "flag_absent_but_nonzero_geometry": int(sum(r["flag_geom_mismatch_absent"] for r in ok)),
        },
        "missing_modality_encoding": {
            "raw_zero_blocks": int(sum(r["zero_block_missing_modality"] for r in ok)),
            "resting_placeholder_blocks": int(sum(r["placeholder_missing_modality"] for r in ok)),
        },
        "class_balance": {
            "min": int(counts.min()), "max": int(counts.max()),
            "median": float(np.median(counts)), "mean": float(counts.mean()),
            "classes_with_1_sample": int(np.sum(counts == 1)),
        },
    }


# =============================================================================
# 6. PLOTS
# =============================================================================
def plot_report(name, records, summary, probe, out_dir=PLOTS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    ok = [r for r in records if not r.get("load_error")]
    if not ok:
        return

    frames = np.array([r["frames"] for r in ok], dtype=np.float64)
    motion = np.array([r["motion_energy"] for r in ok], dtype=np.float64)
    frozen = np.array([r["frozen_ratio"] for r in ok], dtype=np.float64)
    dead_pct = np.array([100.0 * r["dead_frames"] / max(1, r["frames"]) for r in ok])
    longest = np.array([r["longest_static_run"] for r in ok], dtype=np.float64)

    fig, axs = plt.subplots(2, 3, figsize=(19, 10))
    fig.suptitle(f"MediaPipe Tensor Integrity Audit - {name}", fontsize=17, fontweight="bold")

    axs[0, 0].hist(frames, bins=45, color="#2b6cb0", edgecolor="#1a365d")
    axs[0, 0].set_title(f"Frames per Sample (median {np.median(frames):.0f})")
    axs[0, 0].set_xlabel("frames"); axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].hist(motion, bins=45, color="#38a169", edgecolor="#22543d")
    axs[0, 1].set_title("Motion Energy (mean inter-frame L2)")
    axs[0, 1].set_xlabel("||f[t]-f[t-1]||"); axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].axvline(1e-4, color="#e53e3e", ls="--", label="collapse threshold")
    axs[0, 1].legend()

    axs[0, 2].hist(frozen, bins=45, color="#dd6b20", edgecolor="#7b341e")
    axs[0, 2].set_title("Frozen-Frame Ratio per Sample")
    axs[0, 2].set_xlabel("fraction of duplicate consecutive frames")
    axs[0, 2].grid(True, alpha=0.3)

    mod = summary["modality_presence"]
    keys = ["lh", "rh", "arms", "face"]
    vals = [mod[k]["mean_presence"] * 100 for k in keys]
    dead_bar = [mod[k]["samples_never_present"] for k in keys]
    bars = axs[1, 0].bar(keys, vals, color=["#4299e1", "#38a169", "#d69e2e", "#9f7aea"])
    for b, d in zip(bars, dead_bar):
        axs[1, 0].text(b.get_x() + b.get_width() / 2, 2, f"{d} never",
                       ha="center", fontsize=8, color="#e53e3e")
    axs[1, 0].set_ylim(0, 105)
    axs[1, 0].set_title("Modality Presence Rate (%)")
    axs[1, 0].set_ylabel("% frames with modality")
    axs[1, 0].grid(True, axis="y", alpha=0.3)

    axs[1, 1].hist(dead_pct, bins=45, color="#e53e3e", edgecolor="#742a2a")
    axs[1, 1].set_title("Dead-Frame Rate per Sample (all-zero landmarks)")
    axs[1, 1].set_xlabel("% dead frames"); axs[1, 1].grid(True, alpha=0.3)

    axs[1, 2].hist(longest, bins=45, color="#805ad5", edgecolor="#44337a")
    axs[1, 2].set_title("Longest Frozen Run (frames)")
    axs[1, 2].set_xlabel("frames"); axs[1, 2].grid(True, alpha=0.3)

    for ax in axs.flat:
        ax.tick_params(labelsize=8)

    p1 = os.path.join(out_dir, f"tensor_collapse_audit_{name}.png")
    plt.tight_layout()
    plt.savefig(p1, dpi=150)
    plt.close()

    if not probe:
        return
    per_class = probe.get("per_class_recall", {})
    if per_class:
        items = sorted(per_class.items(), key=lambda kv: kv[1])
        fig2, ax2 = plt.subplots(1, 2, figsize=(16, 7))
        fig2.suptitle(f"Discriminability Probe - {name}", fontsize=16, fontweight="bold")
        lbls = [k for k, _ in items]
        vals = [v for _, v in items]
        colors = ["#e53e3e" if v <= probe["chance_accuracy"] * 2 else "#38a169" for v in vals]
        ax2[0].bar(range(len(vals)), vals, color=colors)
        ax2[0].axhline(probe["chance_accuracy"], color="#e53e3e", ls="--",
                       label=f"chance = {probe['chance_accuracy']*100:.2f}%")
        ax2[0].set_xticks(range(len(lbls)))
        ax2[0].set_xticklabels(lbls, rotation=90, fontsize=6)
        ax2[0].set_title("Nearest-Centroid Per-Class Recall (held-out)")
        ax2[0].set_ylabel("recall")
        ax2[0].legend(); ax2[0].grid(True, axis="y", alpha=0.3)

        ax2[1].axis("off")
        txt = (
            f"chance accuracy      : {probe['chance_accuracy']*100:.2f}%\n"
            f"nearest-centroid acc : {probe['nearest_centroid_acc']*100:.2f}%\n"
            f"lift over chance     : {probe['lift_over_chance']*100:+.2f} pts\n"
            f"scatter ratio (btw/total): {probe['scatter_ratio']:.4f}\n"
            f"PCA effective rank   : {probe['pca_effective_rank']:.2f}\n"
            f"PCA top-10 var share : {probe['pca_top10_var_ratio']*100:.1f}%\n"
            f"train/test samples   : {probe['n_train']} / {probe['n_test']}"
        )
        if "logistic_acc" in probe:
            txt += (
                f"\nlogistic probe acc   : {probe['logistic_acc']*100:.2f}%"
                f" (lift {probe['lift_over_chance']*100:+.2f} pts)"
            )
        ax2[1].text(0.02, 0.95, txt, va="top", family="monospace", fontsize=11)
        plt.tight_layout()
        p2 = os.path.join(out_dir, f"tensor_separability_{name}.png")
        plt.savefig(p2, dpi=150)
        plt.close()


# =============================================================================
# 7. MAIN
# =============================================================================
def audit_dataset(key, limit=None, do_probe=True, do_dup=True):
    cfg = DATASETS[key]
    root = cfg["path"]
    print("=" * 78)
    print(f"TENSOR INTEGRITY AUDIT - {cfg['name']}")
    print(f"Path: {root}")
    print("=" * 78)

    if not os.path.isdir(root):
        print(f"[SKIP] Dataset directory not found: {root}\n")
        return None

    if cfg["layout"] == "class_dirs":
        items = collect_class_dirs(root, limit)
    else:
        items = collect_index(root, limit)

    if not items:
        print(f"[SKIP] No tensors found under {root}\n")
        return None

    print(f"[INFO] Found {len(items):,} tensors. Auditing...")

    records, landmarks_list, labels = [], [], []
    for i, item in enumerate(items):
        if cfg["layout"] == "class_dirs":
            path, label = item
            meta = None
        else:
            path, label, meta = item
            meta = {
                "sentence": meta.get("sentence", ""),
                "n_in_scope": len(meta.get("in_scope_indices", [])),
                "dataset": meta.get("dataset", "unknown"),
            }
        rec, lm = analyze_tensor(path, label, meta)
        records.append(rec)
        if lm is not None:
            landmarks_list.append(lm)
            labels.append(label)
        if (i + 1) % 500 == 0:
            print(f"   ... {i+1}/{len(items)}")

    summary = summarize(records)

    print("\n--- CORPUS SUMMARY ---")
    print(f"Samples scanned          : {summary['total']:,} (load errors: {summary['load_errors']})")
    if summary["load_errors"]:
        for m in summary["error_messages"]:
            print(f"    ! {m}")
    print(f"Classes                 : {summary['n_classes']}")
    print(f"Total frames            : {summary['total_frames']:,}")
    f = summary["frames"]
    print(f"Frames/sample           : min {f['min']} | med {f['median']:.0f} | mean {f['mean']:.1f} | max {f['max']}")
    print(f"NaN / Inf values        : {summary['nan_total']} / {summary['inf_total']}")
    print(f"Dead frames (all-zero)  : {summary['dead_frames_total']:,} ({summary['dead_frames_pct']:.2f}% of frames)")
    print(f"Constant samples        : {summary['constant_samples']:,} ({summary['constant_samples_pct']:.2f}%)")
    m = summary["motion_energy"]
    print(f"Motion energy           : med {m['median']:.4f} | p05 {m['p05']:.4f} | near-zero samples {m['near_zero_samples']}")
    print(f"Max |coordinate|         : {summary['max_abs_coord']:.2f}")

    print("\n--- MODALITY PRESENCE (MediaPipe detection health) ---")
    for k, v in summary["modality_presence"].items():
        flag = "  <-- COLLAPSED (never detected)" if v["samples_never_present"] == summary["total"] else ""
        print(f"  {k:5s}: present {v['mean_presence']*100:6.2f}% of frames | "
              f"samples never present: {v['samples_never_present']}{flag}")

    g = summary["flag_geometry_mismatch"]
    print(f"\n  flag=present but geometry all-zero : {g['flag_present_but_zero_geometry']:,} frames")
    print(f"  flag=absent  but geometry non-zero : {g['flag_absent_but_nonzero_geometry']:,} frames")
    enc = summary["missing_modality_encoding"]
    print(f"  missing modality stored as raw zeros   : {enc['raw_zero_blocks']:,} blocks")
    print(f"  missing modality stored as resting pose : {enc['resting_placeholder_blocks']:,} blocks")
    if enc["raw_zero_blocks"] > 0:
        print("  NOTE: an undetected hand is encoded two different ways across the corpus:")
        print("        raw zeros  -> the hand sits on the skeleton origin (0,0,0)")
        print("        resting pose -> the hand sits at (-1.5,3,0) / (+1.5,3,0)")
        print("        The training-time _reclamp() normalises both to the resting pose and")
        print("        fixes the right-hand sign bug (see Backend/training/train_conformer_ctc.py).")

    cb = summary["class_balance"]
    print(f"\n--- CLASS BALANCE ---")
    print(f"  samples/class: min {cb['min']} | med {cb['median']:.0f} | mean {cb['mean']:.1f} | max {cb['max']}")
    print(f"  classes with a single sample: {cb['classes_with_1_sample']}")

    probe = None
    if do_probe and len(landmarks_list) >= 20 and len(set(labels)) >= 2:
        print("\n--- DISCRIMINABILITY PROBE (decisive test) ---")
        X = build_sample_features(landmarks_list)
        y = np.array(labels)
        probe = nearest_centroid_probe(X, y)
        if probe:
            lg = logistic_probe(X, y)
            if lg:
                probe.update(lg)
                print(f"  chance accuracy        : {probe['chance_accuracy']*100:.2f}%")
                print(f"  nearest-centroid acc   : {probe['nearest_centroid_acc']*100:.2f}% "
                      f"(lift {probe['lift_over_chance']*100:+.2f} pts)")
                if "logistic_acc" in probe:
                    print(f"  logistic probe acc     : {probe['logistic_acc']*100:.2f}%")
                print(f"  scatter ratio (btw/tot): {probe['scatter_ratio']:.4f}  "
                      f"(1.0 = perfectly separable, 0.0 = all classes identical)")
                print(f"  PCA effective rank     : {probe['pca_effective_rank']:.2f} "
                      f"(top-10 var share {probe['pca_top10_var_ratio']*100:.1f}%)")
                if probe["lift_over_chance"] < 0.02:
                    print("  [!!] PROBE AT CHANCE -> the feature space carries no class signal.")
                    print("       Retraining cannot fix this; re-extraction is required.")
                elif probe["lift_over_chance"] < 0.10:
                    print("  [!] WEAK SEPARABILITY -> expect a low ceiling and unstable training.")
                else:
                    print("  [OK] Feature space is discriminative; training-side fixes should help.")

    dups = None
    if do_dup:
        dups = duplicate_scan(records)
        print("\n--- DUPLICATE / LEAKAGE SCAN ---")
        print(f"  duplicate tensor groups           : {dups['duplicate_groups']}")
        print(f"  groups spanning multiple labels   : {dups['cross_label_duplicate_groups']} "
              f"({dups['cross_label_duplicate_samples']} samples)")
        for ex in dups["examples"][:5]:
            print(f"      labels={ex['labels']}")

    subs = subcorpus_breakdown(records)
    if len(subs) > 1:
        print("\n--- PER SOURCE-CORPUS BREAKDOWN ---")
        print(f"  {'source':<14}{'samples':>8}{'frames':>10}{'raw-zero':>10}{'resting':>10}"
              f"{'face%':>8}{'lh%':>8}{'rh%':>8}")
        for key, v in sorted(subs.items()):
            p = v["modality_presence_pct"]
            print(f"  {key:<14}{v['samples']:>8}{v['frames']:>10,}"
                  f"{v['raw_zero_blocks']:>10,}{v['resting_placeholder_blocks']:>10,}"
                  f"{p['face']:>8.1f}{p['lh']:>8.1f}{p['rh']:>8.1f}")
        mixed = [k for k, v in subs.items()
                 if v["raw_zero_blocks"] > 0 and v["resting_placeholder_blocks"] > 0]
        if mixed:
            print(f"  [!!] {mixed} store an undetected modality BOTH as raw zeros and as the "
                  f"canonical resting pose inside the same corpus.")

    geo = geometric_frame_report(records, landmarks_list)
    if geo:
        print("\n--- GEOMETRIC FRAME ---")
        print(f"  median neck (shoulder midpoint) : {geo['neck_median']}")
        print(f"  distance from the origin        : {geo['neck_offset_from_origin']} shoulder widths")
        print(f"  per-clip spread (p10 / p90)     : {geo['neck_per_clip_p10_p90']}")
        if geo["neck_offset_from_origin"] > 1.0:
            print("  [!!] The neck is far from the origin: this corpus was extracted with a")
            print("       different anchor than the current scripts/extraction.py (which puts")
            print("       the neck at the origin).  It therefore lives in a DIFFERENT geometric")
            print("       frame from the other corpora - compare with the cross-corpus report.")
        else:
            print("  [OK] The neck sits close to the origin: this corpus follows the current")
            print("       scripts/extraction.py convention (neck-centred).")
            print("       NOTE: the other corpora may still use a different anchor - compare")
            print("       with the cross-corpus report.")

    plot_report(cfg["name"], records, summary, probe)

    report = {
        "dataset": cfg["name"],
        "root": root,
        "summary": summary,
        "probe": probe,
        "duplicates": dups,
        "subcorpora": subs,
        "geometric_frame": geo,
    }
    out_json = os.path.join(EDA_DIR, f"tensor_collapse_audit_{cfg['name']}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[SAVED] report  -> {out_json}")
    print(f"[SAVED] plots   -> {PLOTS_DIR}")
    print("=" * 78 + "\n")
    return report


def cross_corpus_report(reports):
    """Compare the audited corpora and flag a domain shift between them.

    The word corpus pretrains the backbone and the sentence corpus fine-tunes it, so
    the two must live in the same feature space.  Two kinds of mismatch are detected:
      * modality presence  - e.g. the face block is dead in one corpus and alive in the
                             other, which means 21 of the 163 input dimensions change
                             their meaning between pretraining and fine-tuning.
      * missing-modality encoding - one corpus stores an undetected hand as raw zeros
                             and the other as the canonical resting pose.
    """
    if len(reports) < 2:
        return None

    mods = ["lh", "rh", "arms", "face"]
    out = {"modalities": {}, "missing_modality_encoding": {}, "flags": []}

    for key, rep in reports.items():
        s = rep["summary"]
        out["modalities"][key] = {m: round(s["modality_presence"][m]["mean_presence"] * 100, 2) for m in mods}
        enc = s["missing_modality_encoding"]
        out["missing_modality_encoding"][key] = {
            "raw_zero_blocks": enc["raw_zero_blocks"],
            "resting_placeholder_blocks": enc["resting_placeholder_blocks"],
        }

    keys = list(reports.keys())
    # Modality presence shift
    for m in mods:
        vals = [out["modalities"][k][m] for k in keys]
        if max(vals) - min(vals) > 20.0:
            dead = [k for k in keys if out["modalities"][k][m] < 20.0]
            alive = [k for k in keys if out["modalities"][k][m] >= 20.0]
            extra = ""
            if dead and alive:
                extra = (f"  {m.upper()} is effectively DEAD in {dead} but alive in {alive}, "
                         f"so those dimensions change their meaning between the two corpora.")
            elif dead:
                extra = (f"  {m.upper()} is effectively DEAD in {dead} - the corresponding input "
                         f"dimensions are constant zero there.")
            out["flags"].append(
                "MODALITY SHIFT on '" + m + "': " +
                ", ".join(f"{k}={out['modalities'][k][m]:.1f}%" for k in keys) + "." + extra
            )

    # Missing-modality encoding shift
    for key in keys:
        enc = out["missing_modality_encoding"][key]
        tot = enc["raw_zero_blocks"] + enc["resting_placeholder_blocks"]
        if tot and 0.05 < enc["raw_zero_blocks"] / tot < 0.95:
            out["flags"].append(
                f"MIXED ENCODING in {key}: {enc['raw_zero_blocks']:,} missing-modality blocks "
                f"stored as raw zeros and {enc['resting_placeholder_blocks']:,} as the canonical "
                f"resting pose. The same physical state has two different numeric encodings."
            )

    # Geometric frame shift (the neck = the shoulder midpoint must sit in the same place)
    frames = {k: rep.get("geometric_frame") for k, rep in reports.items()}
    frames = {k: v for k, v in frames.items() if v}
    out["geometric_frame"] = frames
    if len(frames) >= 2:
        necks = {k: np.array(v["neck_median"]) for k, v in frames.items()}
        ref_key = max(necks, key=lambda k: float(np.linalg.norm(necks[k])))
        ref = necks[ref_key]
        for k, n in necks.items():
            if k == ref_key:
                continue
            delta = ref - n
            if float(np.linalg.norm(delta)) > 1.0:
                out["flags"].append(
                    f"GEOMETRIC FRAME SHIFT: the neck sits at {np.round(n, 3).tolist()} in {k} "
                    f"but at {np.round(ref, 3).tolist()} in {ref_key} - a displacement of "
                    f"{round(float(np.linalg.norm(delta)), 2)} shoulder widths. A backbone "
                    f"pretrained on one frame and fine-tuned on the other sees a completely "
                    f"different skeleton. Training-side fix: harmonize_features() in "
                    f"Backend/training/train_conformer_ctc.py."
                )
    return out


def main():
    p = argparse.ArgumentParser(
        description="Audit finished MediaPipe tensor datasets for collapse / degeneracy")
    p.add_argument("--dataset", default="all",
                   choices=["all", "word", "word_cslrt", "sentence"])
    p.add_argument("--limit", type=int, default=None,
                   help="cap the number of tensors scanned per dataset")
    p.add_argument("--no_probe", action="store_true", help="skip the separability probe")
    p.add_argument("--no_dup", action="store_true", help="skip the duplicate scan")
    args = p.parse_args()

    keys = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    reports = {}
    for k in keys:
        rep = audit_dataset(k, limit=args.limit, do_probe=not args.no_probe, do_dup=not args.no_dup)
        if rep:
            reports[k] = rep

    if len(reports) >= 2:
        cross = cross_corpus_report(reports)
        if cross and cross["flags"]:
            print("=" * 78)
            print("CROSS-CORPUS DOMAIN-SHIFT CHECK")
            print("=" * 78)
            print("Modality presence (% of frames):")
            for m in ["lh", "rh", "arms", "face"]:
                row = ", ".join(f"{k}={cross['modalities'][k][m]:6.2f}%" for k in reports)
                print(f"  {m:5s}: {row}")
            print("Missing-modality encoding:")
            for k in reports:
                e = cross["missing_modality_encoding"][k]
                print(f"  {k}: raw-zero={e['raw_zero_blocks']:,}  resting-pose={e['resting_placeholder_blocks']:,}")
            print()
            for f in cross["flags"]:
                print(f"  [!!] {f}")
            out_json = os.path.join(EDA_DIR, "tensor_collapse_audit_cross_corpus.json")
            with open(out_json, "w", encoding="utf-8") as fh:
                json.dump(cross, fh, indent=2, default=str)
            print(f"\n[SAVED] report  -> {out_json}")
            print("=" * 78)


if __name__ == "__main__":
    main()
