"""
audit_accuracy.py — YAP segmentor boundary/presence accuracy audit.

Loads trained segmentor checkpoints and measures how well the YAP front-end
detects gesture boundaries on REAL labeled sentence data.

Ground truth comes from ``Backend/segmentor/data_builder/segmentation_clean.csv``
(frame_index / timestamp boundary annotations per sentence). Sentences are
matched to their pre-extracted tensors in ``Backend/data/tensors_sentence_level/``
(via ``index.json``). Boundary frame indices are normalized from video-frame
space into tensor-frame space by scaling to [0, T-1].

Metrics (per checkpoint, aggregated across sentences):
  * Presence  : frame-level accuracy / precision / recall / F1
                (sign vs transition/idle, threshold 0.5)
  * End-boundary: precision / recall / F1 with a tolerance window
                (a prediction counts if within ``--tol`` frames of a GT boundary)
  * Boundary offset MAE : mean |pred_frame - gt_frame| over matched boundaries
  * Segment IoU   : temporal intersection-over-union of predicted segments
                    (runs between detected ends) vs ground-truth segments

Usage (from D:\MookVani\Backend, venv active):
    python ..\audit_accuracy.py
    python ..\audit_accuracy.py --checkpoint Backend/models/yap_segmentor/yap_segmentor_final.pth --tol 3 --max-sentences 60
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

BACKEND_DIR = Path(__file__).resolve().parent.parent   # Backend
SEGMENTOR_DIR = Path(__file__).resolve().parent         # Backend/segmentor
for _p in (str(BACKEND_DIR), str(SEGMENTOR_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from yap_conformer import (  # noqa: E402
    YAPConfig, YAPpingConformer, ConformerConfig, hand_part,
)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
SEG_CSV = SEGMENTOR_DIR / "data_builder" / "segmentation_clean.csv"
SENT_INDEX = BACKEND_DIR / "data" / "tensors_sentence_level" / "index.json"
TOL_DEFAULT = 2
TRANSITION_WINDOW = 2  # frames around a boundary treated as idle (presence=0)

# --------------------------------------------------------------------------- #
# Checkpoint discovery
# --------------------------------------------------------------------------- #
def discover_checkpoints():
    roots = [SEGMENTOR_DIR / "models", BACKEND_DIR / "models"]
    found = []
    for root in roots:
        if not root.is_dir():
            continue
        for entry in root.iterdir():
            name = entry.name.lower()
            if not entry.is_dir():
                continue
            if "yap" in name or "segmentor" in name:
                for p in sorted(entry.rglob("*.pth")):
                    if p.is_file():
                        found.append((str(entry), p))
                for p in sorted(entry.rglob("*.pt")):
                    if p.is_file():
                        found.append((str(entry), p))
    # de-duplicate
    seen = set(); uniq = []
    for d, p in found:
        if str(p) not in seen:
            seen.add(str(p)); uniq.append((d, p))
    return uniq


def extract_state_dict(sd):
    if not isinstance(sd, dict):
        return sd
    for k in ("model_state_dict", "state_dict", "model", "net"):
        if k in sd and isinstance(sd[k], dict):
            return sd[k]
    return sd


def infer_architecture(state):
    """Detect d_model and joint count J from a checkpoint state dict."""
    d_model = 128
    J = 42
    for k in ("yap.core.lift.0.weight", "yap.core.layers.0.inp.weight"):
        if k in state and isinstance(state[k], torch.Tensor):
            d_model = int(state[k].shape[0]); break
    for k in ("yap.core.cleaner.parent_idx", "yap.core.cleaner.part_of_joint",
              "yap.core.cleaner.project_mask"):
        if k in state and isinstance(state[k], torch.Tensor):
            J = int(state[k].shape[0]); break
    layout = "corpus" if J >= 50 else "hands"
    return d_model, J, layout


def build_model(device, layout="hands", d_model=128):
    # The 163-dim tensors use 2 hands (J=42) + flags; hands layout matches.
    if layout == "corpus":
        ycfg = YAPConfig.corpus(d_model=d_model, n_mem_layers=3, summary_channels=32,
                                hippo_order=16, entity_dim=256)
    else:
        ycfg = YAPConfig(parts=[hand_part("left_hand"), hand_part("right_hand")],
                         d_model=d_model, n_mem_layers=3, summary_channels=32,
                         hippo_order=16, entity_dim=256)
    J = ycfg.total_joints
    ccfg = ConformerConfig(num_words=228, d_model=d_model, heads=4, layers=4)
    model = YAPpingConformer(ycfg, ccfg).to(device)
    model.eval()
    return model, J


# --------------------------------------------------------------------------- #
# Ground-truth building
# --------------------------------------------------------------------------- #
def load_segmentation_groups():
    """Return dict sentence_id -> list of boundary rows."""
    groups = {}
    if not SEG_CSV.exists():
        return groups
    with open(SEG_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            groups.setdefault(row["sentence_id"], []).append({
                "frame_index": int(row["frame_index"]),
                "timestamp": float(row["timestamp"]),
                "sentence": row["sentence"],
                "sentence_video": row["sentence_video"],
                "accepted": int(row.get("accepted", "1") or 1),
            })
    return groups


def load_sentence_index():
    if not SENT_INDEX.exists():
        return {}
    with open(SENT_INDEX, encoding="utf-8") as f:
        idx = json.load(f)
    # map sentence -> entry (first match)
    by_sentence = {}
    for e in idx:
        by_sentence.setdefault(e["sentence"], e)
    return by_sentence


def build_gt(boundaries, T):
    """Build presence/end ground-truth tensors of length T.

    Boundary frame_indices are normalized from video space to [0, T-1] tensor
    space by scaling against the max frame_index for the sentence.
    """
    presence = np.ones(T, dtype=np.float32)
    end_target = np.zeros(T, dtype=np.float32)

    if not boundaries:
        return presence, end_target, []

    max_fi = max(b["frame_index"] for b in boundaries) or 1
    gt_boundary_frames = []
    for b in boundaries:
        if not b.get("accepted", 1):
            continue
        tf = round(b["frame_index"] / max_fi * (T - 1))
        tf = min(max(tf, 0), T - 1)
        gt_boundary_frames.append(tf)
        end_target[tf] = 1.0

    gt_boundary_frames = sorted(set(gt_boundary_frames))
    # transition windows around boundaries are idle (presence = 0)
    for bf in gt_boundary_frames:
        s = max(0, bf - TRANSITION_WINDOW)
        e = min(T, bf + TRANSITION_WINDOW + 1)
        presence[s:e] = 0.0

    # ground-truth segments: runs between consecutive boundaries
    segments = []
    starts = [0] + [bf + 1 for bf in gt_boundary_frames]
    ends = gt_boundary_frames + [T - 1]
    for s, e in zip(starts, ends):
        if e > s:
            segments.append((s, e))
    return presence, end_target, segments


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def local_maxima(x, min_distance=1):
    """Indices of local maxima in 1D array x."""
    if len(x) == 0:
        return []
    peaks = []
    for i in range(len(x)):
        left = x[i - 1] if i > 0 else -np.inf
        right = x[i + 1] if i < len(x) - 1 else -np.inf
        if x[i] >= left and x[i] >= right:
            peaks.append(i)
    return peaks


def presence_metrics(presence_logit, presence_gt, thr=0.5):
    pred = (torch.sigmoid(presence_logit) > thr).long()
    gt = presence_gt.long()
    tp = int(((pred == 1) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, tp=tp, tn=tn, fp=fp, fn=fn)


def end_boundary_metrics(end_logit, end_gt, tol=TOL_DEFAULT, thr=0.5):
    end_prob = torch.sigmoid(end_logit)
    pred_frames = set(local_maxima(end_prob.cpu().numpy(), min_distance=1))
    # restrict to confident predictions
    pred_frames = {i for i in pred_frames if end_prob[i].item() > thr}
    gt_frames = set(int(i) for i in torch.nonzero(end_gt > 0.5).flatten().tolist())

    # tolerance-based matching
    gt_sorted = sorted(gt_frames)
    pred_sorted = sorted(pred_frames)

    matched_gt = set()
    matched_pred = set()
    offsets = []
    # greedy match: for each gt, nearest pred within tol
    for g in gt_sorted:
        best = None; best_d = tol + 1
        for p in pred_sorted:
            if p in matched_pred:
                continue
            d = abs(p - g)
            if d <= tol and d < best_d:
                best = p; best_d = d
        if best is not None:
            matched_gt.add(g); matched_pred.add(best); offsets.append(best_d)

    tp = len(matched_gt)
    fp = len(pred_sorted) - len(matched_pred)
    fn = len(gt_sorted) - len(matched_gt)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    mae = float(np.mean(offsets)) if offsets else float("nan")
    return dict(precision=prec, recall=rec, f1=f1, tp=tp, fp=fp, fn=fn,
                mae=mae, n_pred=len(pred_sorted), n_gt=len(gt_sorted),
                matched=tp)


def segment_iou(pred_end_frames, gt_segments, T):
    """IoU of predicted segments (between detected ends) vs GT segments."""
    if not gt_segments:
        return float("nan")
    pred_sorted = sorted(pred_end_frames)
    # predicted segments: from last end (or 0) to next end
    bounds = [0] + [f + 1 for f in pred_sorted] + [T]
    pred_segments = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        if e > s:
            pred_segments.append((s, e - 1))

    if not pred_segments:
        return 0.0

    inter = 0
    union = 0
    for (ps, pe) in pred_segments:
        for (gs, ge) in gt_segments:
            i = max(0, min(pe, ge) - max(ps, gs) + 1)
            u = max(pe, ge) - min(ps, gs) + 1
            inter += i
            union += u
    return inter / max(union, 1)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="YAP segmentor accuracy audit")
    parser.add_argument("--checkpoint", default=None, help="single checkpoint to audit")
    parser.add_argument("--tol", type=int, default=TOL_DEFAULT,
                        help="tolerance window (frames) for end-boundary matching")
    parser.add_argument("--max-sentences", type=int, default=80,
                        help="max sentences to evaluate per checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random_seed = args.seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device} | tol={args.tol} max_sentences={args.max_sentences}")

    seg_groups = load_segmentation_groups()
    sent_index = load_sentence_index()
    print(f"[SEG] {len(seg_groups)} sentences with boundaries in segmentation_clean.csv")
    print(f"[IDX] {len(sent_index)} sentences in tensor index")

    # candidate sentences: have both boundaries and a tensor
    candidate_ids = []
    for sid, bounds in seg_groups.items():
        # find a sentence entry whose text matches
        sentence_text = bounds[0]["sentence"]
        if sentence_text in sent_index:
            candidate_ids.append((sid, sentence_text, bounds))
    random.seed(random_seed)
    random.shuffle(candidate_ids)
    candidate_ids = candidate_ids[: args.max_sentences]
    print(f"[EVAL] {len(candidate_ids)} candidate sentences")

    checkpoints = [(args.checkpoint, None)] if args.checkpoint else discover_checkpoints()
    if args.checkpoint:
        checkpoints = [("single", Path(args.checkpoint))]

    results = {}
    grand = {"n_checkpoints": 0, "n_sentences": 0}

    for ckpt_dir, ckpt in checkpoints:
        label = ckpt.name if ckpt else ckpt_dir
        print(f"\n{'='*70}")
        print(f"CHECKPOINT: {label}")
        try:
            sd = torch.load(ckpt, map_location=device, weights_only=False)
            state = extract_state_dict(sd)
            d_model, J, layout = infer_architecture(state)
            model, Jm = build_model(device, layout=layout, d_model=d_model)
            # Only load the YAP front-end weights (yap.*); the Conformer blocks
            # / head may use a different d_model and are not needed here.
            yap_state = {k[len("yap."):]: v for k, v in state.items() if k.startswith("yap.")}
            missing, unexpected = model.yap.load_state_dict(yap_state, strict=False)
            real_unexpected = [u for u in unexpected if "num_batches_tracked" not in u]
            print(f"  loaded (d_model={d_model}, J={Jm}, layout={layout}): "
                  f"missing={len(missing)}, unexpected={len(real_unexpected)}")
        except Exception as exc:
            print(f"  [ERROR] load failed: {exc}")
            results[str(ckpt)] = {"error": str(exc)}
            continue

        pres_scores = []; end_scores = []; ious = []; maes = []
        n_eval = 0

        for sid, sentence_text, bounds in candidate_ids:
            entry = sent_index[sentence_text]
            tensor_path = entry["tensor_path"]
            if not Path(tensor_path).exists():
                continue
            try:
                x = torch.load(tensor_path, weights_only=True).float()
            except Exception as exc:
                print(f"  [WARN] cannot load {tensor_path}: {exc}")
                continue
            T = x.shape[0]

            presence_gt, end_gt, gt_segments = build_gt(bounds, T)
            presence_gt_t = torch.from_numpy(presence_gt).to(device)
            end_gt_t = torch.from_numpy(end_gt).to(device)

            # forward
            pose = x[:, :J * 3].contiguous().view(T, J, 3).unsqueeze(0).to(device)
            conf = torch.ones(1, T, J, device=device)
            valid = torch.ones(1, T, J, dtype=torch.bool, device=device)
            t = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(0) / 25.0

            with torch.no_grad():
                out = model.yap.forward_frames(pose, conf, valid, t)
            plog = out["presence"][0]   # (T,)
            elog = out["end"][0]        # (T,)

            pm = presence_metrics(plog, presence_gt_t)
            em = end_boundary_metrics(elog, end_gt_t, tol=args.tol)
            pred_ends = [i for i in local_maxima(torch.sigmoid(elog).cpu().numpy())
                         if torch.sigmoid(elog)[i].item() > 0.5]
            iou = segment_iou(pred_ends, gt_segments, T)

            pres_scores.append(pm); end_scores.append(em); ious.append(iou)
            if not np.isnan(em["mae"]):
                maes.append(em["mae"])
            n_eval += 1

        def avg(key, scores, default=0.0):
            vals = [s[key] for s in scores if key in s and s[key] is not None and not (isinstance(s[key], float) and np.isnan(s[key]))]
            return float(np.mean(vals)) if vals else default

        summary = {
            "checkpoint": str(ckpt),
            "dir": ckpt_dir,
            "n_sentences": n_eval,
            "presence": {k: avg(k, pres_scores) for k in ("accuracy", "precision", "recall", "f1")},
            "end_boundary": {k: avg(k, end_scores) for k in ("precision", "recall", "f1", "mae")},
            "segment_iou": avg("x", [{"x": v} for v in ious]) if ious else 0.0,
            "avg_n_pred_end": float(np.mean([e["n_pred"] for e in end_scores])) if end_scores else 0.0,
            "avg_n_gt_end": float(np.mean([e["n_gt"] for e in end_scores])) if end_scores else 0.0,
        }
        results[str(ckpt)] = summary
        grand["n_checkpoints"] += 1
        grand["n_sentences"] += n_eval

        print(f"  sentences evaluated : {n_eval}")
        print(f"  presence            : acc={summary['presence']['accuracy']:.3f} "
              f"prec={summary['presence']['precision']:.3f} rec={summary['presence']['recall']:.3f} f1={summary['presence']['f1']:.3f}")
        print(f"  end boundary        : prec={summary['end_boundary']['precision']:.3f} "
              f"rec={summary['end_boundary']['recall']:.3f} f1={summary['end_boundary']['f1']:.3f} "
              f"mae={summary['end_boundary']['mae']:.2f} frames")
        print(f"  segment IoU         : {summary['segment_iou']:.3f}")

    # ---- Aggregate summary ----
    print(f"\n{'='*70}")
    print("ACCURACY AUDIT SUMMARY")
    print(f"{'='*70}")
    print(f"Checkpoints audited : {grand['n_checkpoints']}")
    print(f"Sentences evaluated : {grand['n_sentences']} (per checkpoint)")

    best_f1 = None; best_ckpt = None
    for ckpt, s in results.items():
        if "error" in s:
            continue
        f1 = s["presence"]["f1"]
        if best_f1 is None or f1 > best_f1:
            best_f1 = f1; best_ckpt = ckpt
    if best_ckpt:
        print(f"Best presence F1    : {best_f1:.3f}  ({best_ckpt})")

    report_path = BACKEND_DIR / "audit_accuracy_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"summary": grand, "results": results}, f, indent=2)
    print(f"\n[REPORT] written to {report_path}")


if __name__ == "__main__":
    main()
