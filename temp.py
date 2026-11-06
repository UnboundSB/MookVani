"""
temp.py — collapse audit across YAP segmentor v1 / v2 / v3 / ... checkpoints.

Walks every segmentor checkpoint directory under Backend/models/ and loads each
.pt / .pth checkpoint, then runs the YAP front-end forward on a small sample of
real word clips. If the cleaned pose / features / presence / end outputs are all
zeros (or near-zero variance) for a checkpoint, that version is flagged as
"collapsed" — i.e. v1, v2, v3, etc. have collapsed all along.

Usage (from D:\MookVani\Backend):
    python ..\\temp.py
    python ..\\temp.py --max-clips 20 --max-frames 300
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

BACKEND_DIR = Path(__file__).resolve().parent.parent
SEGMENTOR_DIR = BACKEND_DIR / "segmentor"
for _p in (str(BACKEND_DIR), str(SEGMENTOR_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from yap_conformer import YAPConfig, YAPpingConformer, ConformerConfig, hand_part  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_classes() -> list:
    classes_path = BACKEND_DIR / "models" / "word_vocabulary" / "classes.json"
    if classes_path.exists():
        with open(classes_path, encoding="utf-8") as f:
            return json.load(f)
    return []


def find_word_clips(max_clips: int, max_words: int, max_frames: int):
    """Gather a few real word clips from the corpora for the forward pass."""
    candidates = [
        BACKEND_DIR / "data" / "tensors_include50_word_level",
        BACKEND_DIR / "data" / "tensors_isl_csltr_word_level",
    ]
    classes = load_classes()
    label_set = set(classes)
    clips = []
    chosen_words = []
    for data_dir in candidates:
        if not data_dir.is_dir():
            continue
        for word_dir in sorted(data_dir.iterdir()):
            if not word_dir.is_dir() or (label_set and word_dir.name not in label_set):
                continue
            chosen_words.append(word_dir.name)
            pts = sorted(word_dir.rglob("*.pt"))
            random.shuffle(pts)
            for p in pts:
                if len(clips) >= max_clips:
                    break
                try:
                    x = torch.load(p, weights_only=True).float()
                    clips.append((word_dir.name, p.name, x[:max_frames]))
                except Exception as exc:  # pragma: no cover - defensive
                    print(f"[WARN] could not load {p}: {exc}")
            if len(clips) >= max_clips:
                break
        if len(clips) >= max_clips:
            break
    return clips, chosen_words[:max_words]


def build_model(device: torch.device, layout: str = "corpus"):
    if layout == "corpus":
        ycfg = YAPConfig.corpus(d_model=128, n_mem_layers=3, summary_channels=32,
                                hippo_order=16, entity_dim=256)
    else:
        ycfg = YAPConfig(parts=[hand_part("left_hand"), hand_part("right_hand")],
                         d_model=128, n_mem_layers=3, summary_channels=32,
                         hippo_order=16, entity_dim=256)
    J = ycfg.total_joints
    ccfg = ConformerConfig(num_words=228, d_model=128, heads=4, layers=4)
    model = YAPpingConformer(ycfg, ccfg).to(device)
    model.eval()
    return model, J, ycfg


def split_163_to_pose(x163: torch.Tensor, J: int):
    T = x163.shape[0]
    coords = x163[:, :J * 3].contiguous().view(T, J, 3)
    conf = torch.ones(T, J)
    valid = torch.ones(T, J, dtype=torch.bool)
    return coords, conf, valid


def forward_stats(model, J: int, clip: torch.Tensor, device: torch.device, fps: float = 30.0):
    """Run YAP front-end on one clip; return per-output variance stats."""
    T = clip.shape[0]
    pose, conf, valid = split_163_to_pose(clip, J)
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
            }
    # zero-join / fully-zero diagnostics
    clean = out["clean"][0] if "clean" in out else None
    zero_joint_slots = int((clean.abs().sum(dim=-1) == 0).sum()) if clean is not None else -1
    frames_all_zero = int((clean.abs().sum(dim=-1) == 0).all(dim=-1).sum()) if clean is not None else -1
    stats["zero_joint_slots"] = zero_joint_slots
    stats["frames_all_joints_zero"] = frames_all_zero
    return stats


def is_collapsed(stats: dict) -> tuple[bool, list]:
    """Return (collapsed, reasons)."""
    reasons = []
    for key in ("clean", "feat", "presence", "end"):
        if key not in stats:
            continue
        s = stats[key]
        if s["abs_max"] < 1e-6:
            reasons.append(f"{key} abs_max ~ 0 ({s['abs_max']:.2e})")
        if s["std"] < 1e-6:
            reasons.append(f"{key} std ~ 0 ({s['std']:.2e})")
    if stats.get("frames_all_joints_zero", 0) > 0:
        reasons.append(f"{stats['frames_all_joints_zero']} frames all joints zero")
    return (len(reasons) > 0), reasons


# --------------------------------------------------------------------------- #
# Checkpoint discovery
# --------------------------------------------------------------------------- #
def discover_segmentor_checkpoints():
    """Find every segmentor model dir and the checkpoints within."""
    roots = [
        SEGMENTOR_DIR / "models",
        BACKEND_DIR / "models",
    ]
    found = {}
    for root in roots:
        if not root.is_dir():
            continue
        for entry in root.iterdir():
            name = entry.name.lower()
            if not entry.is_dir():
                continue
            # any dir that looks segmentor-related
            if "yap" in name or "segmentor" in name or "yap_segmentor" in name:
                ckpts = sorted(entry.rglob("*.pth")) + sorted(entry.rglob("*.pt"))
                ckpts = [c for c in ckpts if c.is_file()]
                if ckpts:
                    found[str(entry)] = ckpts
    return found


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Collapse audit across segmentor v1..N checkpoints")
    parser.add_argument("--max-clips", type=int, default=30, help="max word clips to test")
    parser.add_argument("--max-words", type=int, default=10, help="max distinct words")
    parser.add_argument("--max-frames", type=int, default=900, help="max frames per clip")
    parser.add_argument("--layout", default="corpus", choices=["corpus", "hands"],
                        help="YAP layout to test against")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    model, J, ycfg = build_model(device, layout=args.layout)
    print(f"[LAYOUT] {args.layout} J={J}")

    clips, words = find_word_clips(args.max_clips, args.max_words, args.max_frames)
    print(f"[DATA] testing {len(clips)} clips across {len(words)} words: {words}")

    checkpoints = discover_segmentor_checkpoints()
    print(f"\n[DISCOVERY] {len(checkpoints)} segmentor checkpoint dirs found")

    results = {}
    for ckpt_dir, ckpt_files in sorted(checkpoints.items()):
        print(f"\n{'='*70}")
        print(f"DIR: {ckpt_dir}")
        dir_results = []
        for ckpt in ckpt_files:
            print(f"\n  - {ckpt.name}")
            try:
                sd = torch.load(ckpt, map_location=device, weights_only=False)
            except Exception as exc:
                print(f"    [ERROR] load failed: {exc}")
                dir_results.append({"file": ckpt.name, "error": str(exc), "collapsed": None})
                continue

            # Load state dict flexibly (checkpoint may be a bare dict or nested)
            state = sd
            if isinstance(sd, dict):
                for k in ("model_state_dict", "state_dict", "model"):
                    if k in sd and isinstance(sd[k], dict):
                        state = sd[k]
                        break

            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"    load: missing={len(missing)}, unexpected={len(unexpected)}")

            # Aggregate forward stats across clips
            agg = {k: [] for k in ("clean", "feat", "presence", "end", "speed", "quality")}
            total_zero_slots = 0
            total_all_zero_frames = 0
            total_frames = 0
            fully_zero_clean = 0
            fully_zero_feat = 0

            for word, clip_name, x163 in clips:
                T = min(x163.shape[0], args.max_frames)
                clip = x163[:T]
                stats = forward_stats(model, J, clip, device)
                for k in agg:
                    if k in stats:
                        agg[k].append(stats[k])
                total_zero_slots += stats.get("zero_joint_slots", 0)
                total_all_zero_frames += stats.get("frames_all_joints_zero", 0)
                total_frames += T

            # Compute collapsed verdict from aggregate
            collapsed = False
            reasons = []
            for k in ("clean", "feat", "presence", "end"):
                vals = agg.get(k, [])
                if not vals:
                    continue
                abs_max = max(v["abs_max"] for v in vals)
                std = max(v["std"] for v in vals)
                if abs_max < 1e-6:
                    collapsed = True
                    reasons.append(f"{k} abs_max ~ 0 ({abs_max:.2e})")
                if std < 1e-6:
                    collapsed = True
                    reasons.append(f"{k} std ~ 0 ({std:.2e})")
            if total_all_zero_frames > 0:
                collapsed = True
                reasons.append(f"{total_all_zero_frames}/{total_frames} frames all joints zero")
            if fully_zero_clean > 0 or fully_zero_feat > 0:
                collapsed = True
                reasons.append("fully-zero clean/feat clip(s)")

            verdict = "COLLAPSED" if collapsed else "NOT COLLAPSED"
            print(f"    VERDICT: {verdict}")
            for r in reasons:
                print(f"      - {r}")

            dir_results.append({
                "file": ckpt.name,
                "collapsed": collapsed,
                "reasons": reasons,
                "total_frames": total_frames,
                "zero_joint_slots": total_zero_slots,
                "all_zero_frames": total_all_zero_frames,
            })
        results[ckpt_dir] = dir_results

    # ---- Summary ----
    print(f"\n{'='*70}")
    print("COLLAPSE AUDIT SUMMARY")
    print(f"{'='*70}")
    any_collapsed = False
    for ckpt_dir, dir_results in results.items():
        for r in dir_results:
            if r.get("collapsed") is True:
                any_collapsed = True
                print(f"[COLLAPSED] {ckpt_dir} :: {r['file']}")
                for reason in r.get("reasons", []):
                    print(f"      - {reason}")
    if not any_collapsed:
        print("No checkpoints flagged as collapsed (all have meaningful variance).")

    # Write a machine-readable report
    report_path = BACKEND_DIR / "temp_collapse_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[REPORT] written to {report_path}")


if __name__ == "__main__":
    main()
