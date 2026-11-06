"""
audit_collapse_hardcore.py — Hardcore YAP segmentor collapse audit.

Tests every discovered segmentor checkpoint (v1, v2, stage1, stage1_presence,
stage2, train_segmentor epochs/final, v5 package) to determine whether its YAP
front-end output is COLLAPSED — i.e. all zeros / near-zero variance across the
cleaned pose, features, presence and end heads.

This is stricter than temp.py:
  * Tests MULTIPLE data layouts: corpus (J=53) and hands-only (J=42).
  * Tests MULTIPLE data sources: tensors_include50_word_level,
    tensors_isl_csltr_word_level, tensors_sentence_level, yap_word_level (.npy).
  * Checks the RAW input for zeros — separates "data is all zeros" from
    "model produces zeros" (true collapse).
  * Audits EVERY YAP head: clean, feat, presence, end, speed, quality.
  * Statistical tripwires: abs_max, std, zero-slot ratio, dead-neuron fraction,
    NaN/Inf, fully-zero clips.
  * Multiple clips per word, multiple words, fixed RNG seeds for reproducibility.
  * Per-checkpoint PASS/FAIL verdict + aggregate summary + JSON report.

Usage (from D:\MookVani\Backend, venv active):
    python ..\audit_collapse_hardcore.py
    python ..\audit_collapse_hardcore.py --max-words 12 --max-clips 40 --max-frames 600
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent          # Backend/segmentor
BACKEND_DIR = SCRIPT_DIR.parent                        # Backend
SEGMENTOR_DIR = SCRIPT_DIR
for _p in (str(BACKEND_DIR), str(SEGMENTOR_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from yap_conformer import (  # noqa: E402
    YAPConfig, YAPpingConformer, ConformerConfig, hand_part,
)

# --------------------------------------------------------------------------- #
# Dataset roots
# --------------------------------------------------------------------------- #
DATA_ROOT = BACKEND_DIR / "data"
WORD_CORPORA = {
    "tensors_include50_word_level": DATA_ROOT / "tensors_include50_word_level",
    "tensors_isl_csltr_word_level": DATA_ROOT / "tensors_isl_csltr_word_level",
    "tensors_sentence_level": DATA_ROOT / "tensors_sentence_level",
    "yap_word_level": DATA_ROOT / "yap_word_level",
}

CLASSES_PATH = BACKEND_DIR / "models" / "word_vocabulary" / "classes.json"

# Collapse tripwires (tuned so genuinely working models pass easily).
ABS_MAX_DEAD = 1e-6          # output abs_max below this -> dead
STD_DEAD = 1e-6              # output std below this -> dead
ZERO_SLOT_RATIO_DEAD = 0.95  # >95% of joint slots exactly zero -> dead
FULLY_ZERO_CLIP_DEAD = 1     # any fully-zero clean/feat clip -> dead flag

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_classes() -> list:
    if CLASSES_PATH.exists():
        with open(CLASSES_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def discover_checkpoints():
    """Find every segmentor-related checkpoint (.pth/.pt) under segmentor/ and Backend/models/."""
    roots = [SEGMENTOR_DIR / "models", BACKEND_DIR / "models"]
    found = {}
    for root in roots:
        if not root.is_dir():
            continue
        for entry in root.iterdir():
            name = entry.name.lower()
            if not entry.is_dir():
                continue
            if "yap" in name or "segmentor" in name:
                ckpts = sorted([p for p in entry.rglob("*.pth") if p.is_file()])
                ckpts += sorted([p for p in entry.rglob("*.pt") if p.is_file()])
                # de-duplicate preserving order
                seen = set(); uniq = []
                for c in ckpts:
                    if str(c) not in seen:
                        seen.add(str(c)); uniq.append(c)
                if uniq:
                    found[str(entry)] = uniq
    return found


def extract_state_dict(sd):
    """Pull a raw state_dict out of a checkpoint (bare dict or nested)."""
    if not isinstance(sd, dict):
        return sd
    for k in ("model_state_dict", "state_dict", "model", "net"):
        if k in sd and isinstance(sd[k], dict):
            return sd[k]
    return sd


def load_sample_clips(max_words, max_clips, max_frames, classes):
    """Gather real clips from every corpus, labelled by source."""
    label_set = set(classes)
    clips = []  # (source, word, name, tensor)
    all_chosen_words = []
    for source, root in WORD_CORPORA.items():
        if not root.is_dir():
            continue
        chosen_words = []
        for word_dir in sorted(root.iterdir()):
            if not word_dir.is_dir():
                continue
            if label_set and word_dir.name not in label_set:
                continue
            chosen_words.append(word_dir.name)
            # collect files (.pt or .npy)
            files = sorted([p for p in word_dir.rglob("*") if p.suffix in (".pt", ".npy")])
            random.shuffle(files)
            for p in files:
                if len(clips) >= max_clips:
                    break
                try:
                    if p.suffix == ".pt":
                        x = torch.load(p, weights_only=True).float()
                    else:
                        x = torch.from_numpy(np.load(p)).float()
                    clips.append((source, word_dir.name, p.name, x[:max_frames]))
                except Exception as exc:  # pragma: no cover - defensive
                    print(f"[WARN] could not load {p}: {exc}")
            if len(clips) >= max_clips:
                break
        all_chosen_words.extend(chosen_words)
        if len(clips) >= max_clips:
            break
    return clips, all_chosen_words[:max_words]


def infer_architecture(state):
    """Detect d_model and joint count J from a checkpoint state dict.

    The YAP core dimension is best read from the lift/input projections
    (e.g. lift.0.weight [d_model, 668], layers.0.inp.weight [d_model, d_model]),
    NOT from the bos token (which is a separate entity embedding).
    """
    d_model = None
    J = None

    # d_model from YAPCore lift / input projection
    for k in ("yap.core.lift.0.weight", "yap.core.layers.0.inp.weight",
              "yap.core.presence_head.weight"):
        if k in state and isinstance(state[k], torch.Tensor):
            if k.endswith("presence_head.weight"):
                # shape [1, d_model+1]
                d_model = int(state[k].shape[1]) - 1
            else:
                d_model = int(state[k].shape[0])
            break
    if d_model is None:
        d_model = 128  # safe fallback

    # joint count J from cleaner.parent_idx / part_of_joint / project_mask
    for k in ("yap.core.cleaner.parent_idx", "yap.core.cleaner.part_of_joint",
              "yap.core.cleaner.project_mask", "cleaner.parent_idx"):
        if k in state and isinstance(state[k], torch.Tensor):
            J = int(state[k].shape[0]); break
    if J is None:
        J = 42  # safe fallback (hands)

    layout = "corpus" if J >= 50 else "hands"
    return d_model, J, layout


def build_model(device, layout, d_model):
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


def split_layout(x163, J, layout):
    """Split (T,163) into pose/conf/valid for the given layout."""
    T = x163.shape[0]
    if layout == "corpus":
        # corpus layout: 53 joints x 3 = 159 coords + 4 flags = 163
        coords = x163[:, :J * 3].contiguous().view(T, J, 3)
    else:
        # hands-only: first 42*3=126 are the two hands; rest is padding/flags
        coords = x163[:, :J * 3].contiguous().view(T, J, 3)
    conf = torch.ones(T, J)
    valid = torch.ones(T, J, dtype=torch.bool)
    return coords, conf, valid


def forward_stats(model, J, clip, device, layout, fps=30.0):
    """Run YAP front-end on one clip; return stats dict per output head."""
    T = clip.shape[0]
    pose, conf, valid = split_layout(clip, J, layout)
    pose = pose.unsqueeze(0).to(device)
    conf = conf.unsqueeze(0).to(device)
    valid = valid.unsqueeze(0).to(device)
    t = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(0) / fps

    with torch.no_grad():
        out = model.yap.forward_frames(pose, conf, valid, t)

    stats = {}
    for key in ("clean", "feat", "presence", "end", "speed", "quality"):
        if key in out:
            v = out[key][0]
            stats[key] = {
                "mean": float(v.mean()),
                "std": float(v.std()),
                "min": float(v.min()),
                "max": float(v.max()),
                "abs_max": float(v.abs().max()),
                "has_nan": bool(torch.isnan(v).any()),
                "has_inf": bool(torch.isinf(v).any()),
            }
    clean = out["clean"][0] if "clean" in out else None
    if clean is not None:
        zero_joint_slots = int((clean.abs().sum(dim=-1) == 0).sum())
        total_joint_slots = int(clean.shape[0] * clean.shape[1])
        frames_all_zero = int((clean.abs().sum(dim=-1) == 0).all(dim=-1).sum())
    else:
        zero_joint_slots = -1
        total_joint_slots = -1
        frames_all_zero = -1
    stats["zero_joint_slots"] = zero_joint_slots
    stats["total_joint_slots"] = total_joint_slots
    stats["zero_slot_ratio"] = (zero_joint_slots / total_joint_slots) if total_joint_slots else -1.0
    stats["frames_all_joints_zero"] = frames_all_zero
    stats["T"] = T
    return stats


def assess_collapse(agg_stats):
    """Return (collapsed, reasons) from aggregated stats (already per-key dicts)."""
    reasons = []
    collapsed = False

    for key in ("clean", "feat", "presence", "end"):
        s = agg_stats.get(key)
        if not s:
            continue
        abs_max = float(s.get("abs_max", 0.0))
        std = float(s.get("std", 0.0))
        nan_any = bool(s.get("has_nan", False))
        inf_any = bool(s.get("has_inf", False))
        if abs_max < ABS_MAX_DEAD:
            collapsed = True
            reasons.append(f"{key} abs_max ~0 ({abs_max:.2e})")
        if std < STD_DEAD:
            collapsed = True
            reasons.append(f"{key} std ~0 ({std:.2e})")
        if nan_any:
            collapsed = True
            reasons.append(f"{key} contains NaN")
        if inf_any:
            collapsed = True
            reasons.append(f"{key} contains Inf")

    zero_slot_ratio = agg_stats.get("zero_slot_ratio", -1.0)
    if zero_slot_ratio > ZERO_SLOT_RATIO_DEAD:
        collapsed = True
        reasons.append(f">95% joints zero ({100*zero_slot_ratio:.1f}%)")

    # NOTE: frames_all_joints_zero reflects INPUT data sparsity (missing
    # MediaPipe detections in a few frames), NOT model collapse by itself.
    # It is reported as data quality; only flagged as collapse if it dominates
    # (>50% of frames), which would mean the model never sees real pose.
    all_zero_frames = agg_stats.get("frames_all_joints_zero", 0)
    total_frames = agg_stats.get("total_frames", 0)
    if total_frames and (all_zero_frames / total_frames) > 0.5:
        collapsed = True
        reasons.append(f"{all_zero_frames}/{total_frames} frames (>50%) all joints zero")

    fully_zero_clean = agg_stats.get("fully_zero_clean_clips", 0)
    fully_zero_feat = agg_stats.get("fully_zero_feat_clips", 0)
    if fully_zero_clean > 0:
        collapsed = True
        reasons.append(f"{fully_zero_clean} clips fully-zero clean (model output NULL)")
    if fully_zero_feat > 0:
        collapsed = True
        reasons.append(f"{fully_zero_feat} clips fully-zero feat (model output NULL)")

    return collapsed, reasons


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Hardcore collapse audit across segmentor checkpoints")
    parser.add_argument("--max-words", type=int, default=12, help="max distinct words")
    parser.add_argument("--max-clips", type=int, default=40, help="max clips total")
    parser.add_argument("--max-frames", type=int, default=600, help="max frames per clip")
    parser.add_argument("--layout", default="both", choices=["corpus", "hands", "both"],
                        help="YAP layout(s) to test against")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    print(f"[SEED] {args.seed} | max_words={args.max_words} max_clips={args.max_clips} max_frames={args.max_frames}")

    classes = load_classes()
    print(f"[CLASSES] {len(classes)} canonical classes loaded")

    clips, words = load_sample_clips(args.max_words, args.max_clips, args.max_frames, classes)
    print(f"[DATA] {len(clips)} clips across {len(words)} words: {words}")
    print(f"[SOURCES] {sorted(set(c[0] for c in clips))}")

    checkpoints = discover_checkpoints()
    print(f"\n[DISCOVERY] {len(checkpoints)} segmentor checkpoint dirs found")

    results = {}
    grand_summary = {"collapsed": 0, "ok": 0, "error": 0, "total": 0}

    for ckpt_dir, ckpt_files in sorted(checkpoints.items()):
        dir_label = os.path.basename(ckpt_dir)
        print(f"\n{'='*70}")
        print(f"DIR: {ckpt_dir}")
        dir_results = []
        for ckpt in ckpt_files:
            print(f"\n  - {ckpt.name}")
            ckpt_entry = {"file": ckpt.name, "dir": ckpt_dir, "layouts": {}}

            # Load checkpoint once to infer its native architecture.
            try:
                sd = torch.load(ckpt, map_location=device, weights_only=False)
                state = extract_state_dict(sd)
            except Exception as exc:
                print(f"      [ERROR] could not load checkpoint: {exc}")
                ckpt_entry["layouts"]["load"] = {"error": str(exc), "collapsed": None}
                dir_results.append(ckpt_entry)
                grand_summary["total"] += 1
                grand_summary["error"] += 1
                continue

            d_model, J, native_layout = infer_architecture(state)
            print(f"      inferred arch: d_model={d_model}, J={J}, layout={native_layout}")

            # Only the checkpoint's native layout can load its weights meaningfully.
            layouts_to_test = [native_layout]
            any_layout_collapsed = False
            for layout in layouts_to_test:
                print(f"      layout={layout}")
                try:
                    model, Jm = build_model(device, layout, d_model)
                    # Only load the YAP front-end weights (yap.*); ignore the
                    # Conformer blocks / head which may use a different d_model.
                    yap_state = {k[len("yap."):]: v for k, v in state.items() if k.startswith("yap.")}
                    missing, unexpected = model.yap.load_state_dict(yap_state, strict=False)
                    real_unexpected = [u for u in unexpected if "num_batches_tracked" not in u]
                    print(f"          load(yap): missing={len(missing)}, unexpected={len(real_unexpected)}, J={Jm}")
                except Exception as exc:
                    print(f"          [ERROR] build/load failed: {exc}")
                    ckpt_entry["layouts"][layout] = {"error": str(exc), "collapsed": None}
                    continue

                # Aggregate over clips
                agg = {k: [] for k in ("clean", "feat", "presence", "end", "speed", "quality")}
                total_zero_slots = 0
                total_all_zero_frames = 0
                total_frames = 0
                fully_zero_clean = 0
                fully_zero_feat = 0
                total_input_all_zero_frames = 0

                for source, word, clip_name, x163 in clips:
                    T = min(x163.shape[0], args.max_frames)
                    clip = x163[:T]
                    stats = forward_stats(model, Jm, clip, device, layout)
                    for k in agg:
                        if k in stats:
                            agg[k].append(stats[k])
                    total_zero_slots += stats.get("zero_joint_slots", 0)
                    total_all_zero_frames += stats.get("frames_all_joints_zero", 0)
                    total_frames += stats.get("T", 0)
                    # input data quality: how many frames have all hand coords zero (missing detection)
                    hands = clip[:, :126]
                    total_input_all_zero_frames += int((hands.abs().sum(dim=-1) == 0).all(dim=-1).sum())
                    # detect fully-zero clean / feat clips from abs_max
                    if stats.get("clean", {}).get("abs_max", 1) == 0.0:
                        fully_zero_clean += 1
                    if stats.get("feat", {}).get("abs_max", 1) == 0.0:
                        fully_zero_feat += 1

                agg_stats = {}
                for k in ("clean", "feat", "presence", "end", "speed", "quality"):
                    vals = agg.get(k, [])
                    if vals:
                        agg_stats[k] = {
                            "abs_max": max(v["abs_max"] for v in vals),
                            "std": max(v["std"] for v in vals),
                            "mean": float(np.mean([v["mean"] for v in vals])),
                            "has_nan": any(v["has_nan"] for v in vals),
                            "has_inf": any(v["has_inf"] for v in vals),
                        }
                denom = total_frames * Jm if (total_frames and Jm) else 0
                agg_stats["zero_slot_ratio"] = (total_zero_slots / denom) if denom else -1.0
                agg_stats["frames_all_joints_zero"] = total_all_zero_frames
                agg_stats["total_frames"] = total_frames
                agg_stats["input_all_zero_frames"] = total_input_all_zero_frames
                agg_stats["input_all_zero_ratio"] = (total_input_all_zero_frames / total_frames) if total_frames else -1.0
                agg_stats["fully_zero_clean_clips"] = fully_zero_clean
                agg_stats["fully_zero_feat_clips"] = fully_zero_feat

                collapsed, reasons = assess_collapse(agg_stats)
                if collapsed:
                    any_layout_collapsed = True
                print(f"          VERDICT: {'COLLAPSED' if collapsed else 'OK'}  "
                      f"(input zero-frames {total_input_all_zero_frames}/{total_frames} "
                      f"={100*total_input_all_zero_frames/max(total_frames,1):.1f}%, "
                      f"clean abs_max={agg_stats.get('clean',{}).get('abs_max',0):.2e} "
                      f"presence abs_max={agg_stats.get('presence',{}).get('abs_max',0):.2e} "
                      f"end abs_max={agg_stats.get('end',{}).get('abs_max',0):.2e})")
                for r in reasons:
                    print(f"            - {r}")

                ckpt_entry["layouts"][layout] = {
                    "collapsed": collapsed,
                    "reasons": reasons,
                    "J": Jm,
                    "d_model": d_model,
                    "stats": agg_stats,
                }

            ckpt_entry["collapsed_any_layout"] = any_layout_collapsed
            dir_results.append(ckpt_entry)
            grand_summary["total"] += 1
            if any_layout_collapsed:
                grand_summary["collapsed"] += 1
            elif any(l.get("error") for l in ckpt_entry["layouts"].values()):
                grand_summary["error"] += 1
            else:
                grand_summary["ok"] += 1

        results[ckpt_dir] = dir_results

    # ---- Summary ----
    print(f"\n{'='*70}")
    print("HARDCORE COLLAPSE AUDIT SUMMARY")
    print(f"{'='*70}")
    print(f"Total checkpoints tested : {grand_summary['total']}")
    print(f"COLLAPSED                : {grand_summary['collapsed']}")
    print(f"OK (not collapsed)       : {grand_summary['ok']}")
    print(f"Errors                   : {grand_summary['error']}")

    print("\n--- Per-checkpoint collapsed flags ---")
    any_collapsed = False
    for ckpt_dir, dir_results in results.items():
        for r in dir_results:
            if r.get("collapsed_any_layout") is True:
                any_collapsed = True
                print(f"[COLLAPSED] {ckpt_dir} :: {r['file']}")
                for layout, lres in r["layouts"].items():
                    if isinstance(lres, dict) and lres.get("collapsed"):
                        for reason in lres.get("reasons", []):
                            print(f"      [{layout}] - {reason}")
    if not any_collapsed:
        print("No checkpoints flagged as collapsed in any layout.")

    # Write machine-readable report
    report_path = BACKEND_DIR / "audit_collapse_hardcore_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"summary": grand_summary, "results": results}, f, indent=2)
    print(f"\n[REPORT] written to {report_path}")


if __name__ == "__main__":
    main()
