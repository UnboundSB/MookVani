"""
verify_no_collapse.py — FIXED
=============================

Check that YAP Stage 1 front-end output is NOT collapsed to NULL/zeros
across a sample of real ISL-CSLRT word clips.

Also checks the RAW input tensors for zero-columns to distinguish
"raw data has missing landmarks" from "YAP produces zeros".

Usage (from D:\MookVani\Backend):
    .\venv\Scripts\python.exe segmentor\verify_no_collapse.py
"""

from __future__ import annotations

import json, os, random, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from yap_conformer import YAPConfig, YAPpingConformer, ConformerConfig


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_SRC = os.path.join(BACKEND_DIR, "data", "tensors_include50_word_level")
CLASSES  = os.path.join(BACKEND_DIR, "models", "word_vocabulary", "classes.json")
INIT     = os.path.join(BACKEND_DIR, "segmentor", "models", "yap_stage1", "yap_noise_robust.pth")

SAMPLE_CLIPS = 30
SAMPLE_WORDS = 10
MAX_LEN    = 900
LAYOUT     = "corpus"      # J=53, must match Stage 1 weights
FPS        = 30.0


def load_sample_clips(data_src: str, classes: list[str], n_words: int, n_clips: int):
    label_map = {c: i for i, c in enumerate(classes)}
    d = Path(data_src)
    available_words = [f.name for f in d.iterdir() if f.is_dir() and f.name in label_map]
    random.shuffle(available_words)
    chosen_words = available_words[:n_words]

    clips = []
    for word in chosen_words:
        folder = d / word
        pts = sorted(folder.rglob("*.pt"))
        random.shuffle(pts)
        for p in pts:
            if len(clips) >= n_clips:
                break
            x = torch.load(p, weights_only=True).float()
            clips.append((word, p.name, x))
        if len(clips) >= n_clips:
            break
    return clips


def split_163_to_pose(x163: torch.Tensor, J: int):
    T = x163.shape[0]
    coords = x163[:, :J * 3].contiguous().view(T, J, 3)
    conf = torch.ones(T, J)
    valid = torch.ones(T, J, dtype=torch.bool)
    return coords, conf, valid


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    with open(CLASSES, encoding="utf-8") as f:
        classes = json.load(f)
    num_classes = len(classes)

    clips = load_sample_clips(DATA_SRC, classes, SAMPLE_WORDS, SAMPLE_CLIPS)
    print(f"[DATA] testing {len(clips)} clips across {SAMPLE_WORDS} words")

    ycfg = YAPConfig.corpus(d_model=128, n_mem_layers=3, summary_channels=32,
                            hippo_order=16, entity_dim=256)
    J = ycfg.total_joints
    print(f"[LAYOUT] corpus J={J}")

    ccfg = ConformerConfig(num_words=num_classes, d_model=128, heads=4, layers=4)
    model = YAPpingConformer(ycfg, ccfg).to(device)
    model.eval()

    if os.path.exists(INIT):
        sd = torch.load(INIT, map_location=device, weights_only=True)
        from utils.checkpoint_utils import load_flexible
        missing, unexpected = load_flexible(model, sd, label="STAGE1")
        print(f"[INIT] missing {len(missing)}, unexpected {len(unexpected)}")
    else:
        print("[WARN] no init -- random weights")

    # Accumulators
    total_frames = 0
    total_joint_slots = 0
    zero_joint_slots = 0
    frames_all_joints_zero = 0

    zero_input_slots = 0
    total_input_slots = 0

    clean_all = []
    feat_all = []
    pres_all = []
    end_all = []
    speed_all = []
    quality_all = []

    fully_zero_clean_clips = 0
    fully_zero_feat_clips = 0

    with torch.no_grad():
        for word, clip_name, x163 in clips:
            T = min(x163.shape[0], MAX_LEN)
            clip = x163[:T]

            # ---- check RAW input for zeros ----
            raw_coords = clip[:, :J * 3].view(T, J, 3)
            total_input_slots += T * J
            zero_input_slots += (raw_coords.abs().sum(dim=-1) == 0).sum().item()

            pose, conf, valid = split_163_to_pose(clip, J)
            pose = pose.unsqueeze(0).to(device)
            conf = conf.unsqueeze(0).to(device)
            valid = valid.unsqueeze(0).to(device)
            t = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(0) / FPS

            out = model.yap.forward_frames(pose, conf, valid, t)

            clean = out["clean"][0]           # (T, J, 3)
            feat  = out["feat"][0]            # (T, C)
            pres  = out["presence"][0]        # (T,)
            end   = out["end"][0]             # (T,)
            speed = out["speed"][0]           # (T,)
            quality = out["quality"][0]       # (T,)

            total_frames += T
            total_joint_slots += T * J

            # Count zero joints (all 3 coords exactly 0)
            zero_joints = (clean.abs().sum(dim=-1) == 0)  # (T, J)
            zero_joint_slots += zero_joints.sum().item()
            frames_all_joints_zero += (zero_joints.all(dim=-1)).sum().item()  # frames where ALL joints are zero

            if clean.abs().max().item() == 0.0:
                fully_zero_clean_clips += 1
                print(f"  !! FULLY ZERO CLEAN: {word}/{clip_name}")
            if feat.abs().max().item() == 0.0:
                fully_zero_feat_clips += 1
                print(f"  !! FULLY ZERO FEAT: {word}/{clip_name}")

            clean_all.append(clean.cpu().numpy())
            feat_all.append(feat.cpu().numpy())
            pres_all.append(torch.sigmoid(pres).cpu().numpy())
            end_all.append(torch.sigmoid(end).cpu().numpy())
            speed_all.append(speed.cpu().numpy())
            quality_all.append(quality.cpu().numpy())

    # ---- Aggregate ----
    clean_concat = np.concatenate(clean_all, axis=0)
    feat_concat = np.concatenate(feat_all, axis=0)
    pres_concat = np.concatenate(pres_all, axis=0)
    end_concat = np.concatenate(end_all, axis=0)
    speed_concat = np.concatenate(speed_all, axis=0)
    quality_concat = np.concatenate(quality_all, axis=0)

    print("\n" + "=" * 60)
    print("COLLAPSE CHECK RESULTS")
    print("=" * 60)
    print(f"Clips tested       : {len(clips)}")
    print(f"Total frames       : {total_frames}")
    print(f"Total joint-slots  : {total_joint_slots} ({total_frames} frames x {J} joints)")
    print(f"Zero joint-slots   : {zero_joint_slots} ({100*zero_joint_slots/total_joint_slots:.2f}%)")
    print(f"Zero INPUT slots   : {zero_input_slots} ({100*zero_input_slots/total_input_slots:.2f}%)")
    print(f"Frames ALL joints zero: {frames_all_joints_zero} ({100*frames_all_joints_zero/total_frames:.2f}%)")
    print(f"Fully zero clean clips : {fully_zero_clean_clips} / {len(clips)}")
    print(f"Fully zero feat clips  : {fully_zero_feat_clips} / {len(clips)}")

    print("\n--- Cleaned pose (out['clean']) ---")
    print(f"  mean  = {clean_concat.mean():.6f}   std = {clean_concat.std():.6f}")
    print(f"  min   = {clean_concat.min():.6f}   max = {clean_concat.max():.6f}")

    print("\n--- Features (out['feat']) ---")
    print(f"  mean  = {feat_concat.mean():.6f}   std = {feat_concat.std():.6f}")
    print(f"  min   = {feat_concat.min():.6f}   max = {feat_concat.max():.6f}")

    print("\n--- Presence (sigmoid) ---")
    print(f"  mean  = {pres_concat.mean():.6f}   std = {pres_concat.std():.6f}")
    print(f"  min   = {pres_concat.min():.6f}   max = {pres_concat.max():.6f}")

    print("\n--- End prob (sigmoid) ---")
    print(f"  mean  = {end_concat.mean():.6f}   std = {end_concat.std():.6f}")
    print(f"  min   = {end_concat.min():.6f}   max = {end_concat.max():.6f}")

    print("\n--- Speed ---")
    print(f"  mean  = {speed_concat.mean():.6f}   max = {speed_concat.max():.6f}")

    print("\n--- Quality ---")
    print(f"  mean  = {quality_concat.mean():.6f}   min = {quality_concat.min():.6f}   max = {quality_concat.max():.6f}")

    # ---- Verdict ----
    print("\n" + "=" * 60)
    reasons = []
    if zero_joint_slots / total_joint_slots > 0.5:
        reasons.append(f">50% of joints are zero ({100*zero_joint_slots/total_joint_slots:.1f}%)")
    if clean_concat.std() < 1e-6:
        reasons.append("cleaned pose std ~ 0")
    if feat_concat.std() < 1e-6:
        reasons.append("features std ~ 0")
    if fully_zero_clean_clips > 0:
        reasons.append(f"{fully_zero_clean_clips} clips fully zero")

    if reasons:
        print("VERDICT: COLLAPSED")
        for r in reasons:
            print(f"  - {r}")
    else:
        print("VERDICT: NOT COLLAPSED — output has meaningful variance")
    print("=" * 60)


if __name__ == "__main__":
    main()
