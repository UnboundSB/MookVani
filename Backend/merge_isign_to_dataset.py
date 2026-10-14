"""
========================================================================================
MOOKVANI ISL: iSign Dataset Collision Matching & Pose Extraction Pipeline
========================================================================================
This script identifies all INCLUDE50 classes in `Backend/data/tensors_include50_word_level` 
that have fewer than 14 samples, finds matching isolated word videos from the 127k iSign 
dataset, extracts 163-dimensional MediaPipe landmark feature tensors, and merges them 
directly into the local dataset folders to bump all matching classes up to the 14-sample mark.

Features Extracted per Frame (163-dim canonical layout):
  [0:63]    Left Hand (21 landmarks x 3)
  [63:126]  Right Hand (21 landmarks x 3)
  [126:138] Arms (4 landmarks x 3: shoulders + elbows)
  [138:159] Face (7 key landmarks x 3)
  [159:163] Tracking Flags [lh_flag, rh_flag, arms_flag, face_flag]
========================================================================================
"""

import os
import sys
import glob
import re
import cv2
import json
import time
import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

import mediapipe as mp

# Ensure backend root is on sys.path
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Import the EXACT SAME extraction functions used by INCLUDE50 (scripts/extract_word.py and scripts/extract_video.py)
from scripts.extract_video import extract_video_features
from scripts.extraction import clean_word


# --- Main Merge and Extraction Pipeline ---
def run_merge_pipeline(
    data_dir=r"d:\MookVani\Backend\data\tensors_include50_word_level",
    csv_paths=[
        r"E:\channel\mp\vrutti\iSign_v1.1.csv",
        r"F:\dataset\isign_extracted\word-description-dataset_v1.1.csv",
        r"F:\dataset\isign_extracted\word-presence-dataset_v1.1.csv",
    ],
    isign_video_dir=r"F:\dataset\isign_extracted\iSign-videos_v1.1",
    target_count=14,
    dry_run=False
):
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print("=" * 75)
    print("MOOKVANI ISL: MERGE & BUMP DATASET PIPELINE")
    print("=" * 75)
    print(f"Target Dataset Directory: {data_dir}")
    print(f"iSign Video Directory:    {isign_video_dir}")
    print(f"Target Samples per Class: {target_count} samples")
    print(f"Dry Run Mode:             {dry_run}")
    print("=" * 75 + "\n")

    # 1. Scan Current Dataset
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Target dataset directory does not exist: {data_dir}")

    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d != "NONE"])
    current_counts = {}
    for c in classes:
        current_counts[c] = len(glob.glob(os.path.join(data_dir, c, "*.pt")))

    weak_classes = {c: count for c, count in current_counts.items() if count < target_count}
    print(f"Total gesture classes in dataset: {len(classes)}")
    print(f"Classes already at or above {target_count}: {len(classes) - len(weak_classes)}")
    print(f"Classes below target ({target_count}):     {len(weak_classes)}")
    print("-" * 75)

    # 2. Parse iSign Annotations
    active_csv = None
    for p in csv_paths:
        if os.path.exists(p):
            active_csv = p
            break

    if not active_csv:
        raise FileNotFoundError(f"Could not locate iSign CSV. Looked in: {csv_paths}")

    print(f"Loading iSign annotations from: {active_csv}")
    df = pd.read_csv(active_csv)
    
    # Map UID and word
    uid_col = 'uid' if 'uid' in df.columns else 'word_id'
    text_col = 'text' if 'text' in df.columns else 'word'

    df['clean_word'] = df[text_col].apply(clean_word)
    isign_map = df.groupby('clean_word')[uid_col].apply(list).to_dict()
    print(f"Loaded {len(df)} rows ({len(isign_map)} unique clean words) from iSign annotations.\n")

    # 3. Identify Candidates to Bump
    candidates = {}
    for c, count in weak_classes.items():
        clean_c = clean_word(c)
        uids = isign_map.get(clean_c, [])
        needed = target_count - count
        
        # Verify video files exist
        valid_uids = []
        for uid in uids:
            vid_path = os.path.join(isign_video_dir, f"{uid}.mp4")
            if os.path.exists(vid_path):
                valid_uids.append((uid, vid_path))
                if len(valid_uids) == needed:
                    break

        if valid_uids:
            candidates[c] = {
                'current_count': count,
                'needed': needed,
                'videos_to_process': valid_uids
            }

    print("=" * 75)
    print(f"BUMP CANDIDATES IDENTIFIED ({len(candidates)} classes can be bumped):")
    print(f"{'CLASS NAME':<22} | {'CURRENT':<8} | {'NEEDED':<8} | {'ISIGN VIDEOS FOUND':<18}")
    print("-" * 75)
    for c, info in candidates.items():
        print(f"{c:<22} | {info['current_count']:<8} | {info['needed']:<8} | {len(info['videos_to_process']):<18}")
    print("=" * 75 + "\n")

    if dry_run:
        print("[Dry Run Complete] No files were extracted or saved.")
        return

    if not candidates:
        print("No candidates available to bump. Pipeline finished.")
        return

    # 4. Extract Using Canonical INCLUDE50 Extraction Pipeline
    print("Extracting features using canonical INCLUDE50 extract_video_features pipeline...")
    total_extracted = 0
    start_time = time.time()

    for c, info in candidates.items():
        out_class_dir = os.path.join(data_dir, c)
        os.makedirs(out_class_dir, exist_ok=True)
        
        pbar = tqdm(info['videos_to_process'], desc=f"Extracting [{c}]")
        for uid, vid_path in pbar:
            out_tensor_path = os.path.join(out_class_dir, f"isign_{uid}.pt")
            if os.path.exists(out_tensor_path):
                continue

            tensor_data = extract_video_features(vid_path)
            if tensor_data is not None:
                torch.save(tensor_data, out_tensor_path)
                total_extracted += 1

    elapsed = time.time() - start_time
    print("\n" + "=" * 75)
    print("EXTRACTION & MERGE COMPLETE!")
    print(f"Total new pose tensors added: {total_extracted}")
    print(f"Total elapsed time:           {elapsed / 60:.2f} minutes")
    print(f"Updated Dataset Location:     {data_dir}")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge iSign videos into MookVani word dataset to bump classes to 14.")
    parser.add_argument("--target_count", type=int, default=14, help="Target minimum videos per class (default: 14)")
    parser.add_argument("--dry_run", action="store_true", help="Print collision plan without running MediaPipe extraction")
    args = parser.parse_args()

    run_merge_pipeline(target_count=args.target_count, dry_run=args.dry_run)
