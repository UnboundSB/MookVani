"""
Multi-Core Word-Level Frame Feature Extractor
==============================================

Parallel extraction of word-level frames using MediaPipe Holistic.
Saves tensors with labels to: mookvani_dataset/islcsltr_corpus_extracted/

Output per word class:
    <word>/
        <take>.pt  — {'landmarks': (T, 474), 'label': int, 'word': str, 'take': str}
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import pandas as pd
import mediapipe as mp
import sys

# Add backend to path
sys.path.insert(0, r"D:\mookvani\Backend")
from utils.landmark_extractor import (
    get_holistic_landmarker,
    extract_video_clip_landmarks,
    N_DIMS
)

# Constants
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# ──────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────

SOURCE_FRAMES_ROOT = r"D:\mookvani\Backend\data\ISL_CSLRT_Corpus\Frames_Word_Level"
OUTPUT_ROOT = r"D:\mookvani\mookvani_dataset\islcsltr_corpus_extracted"
CSV_PATH = r"D:\mookvani\Backend\data\ISL_CSLRT_Corpus\corpus_csv_files\ISL_CSLRT_Corpus_word_details.xlsx"
MODEL_ROOT = r"D:\mookvani"

# Processing
N_WORKERS = max(1, cpu_count() - 1)  # Leave 1 core free
MAX_FRAMES = 120

# ──────────────────────────────────────────────────────────────────
# WORKER FUNCTION (must be at module level for pickling)
# ──────────────────────────────────────────────────────────────────

# Global landmarker (initialized per worker)
_worker_landmarker = None

def init_worker():
    """Initialize MediaPipe Holistic in each worker process."""
    global _worker_landmarker
    from utils.landmark_extractor import get_holistic_landmarker
    _worker_landmarker = get_holistic_landmarker()

def extract_single_clip(args):
    """Extract landmarks for a single take directory. Runs in worker process."""
    take_dir, word, label, word_dir, max_frames = args
    
    try:
        from utils.landmark_extractor import extract_video_clip_landmarks
        clip = extract_video_clip_landmarks(take_dir, _worker_landmarker, max_frames=max_frames)
        
        # Save as .pt with metadata
        out_dir = Path(r"D:\mookvani\mookvani_dataset\islcsltr_corpus_extracted") / word.upper().replace(' ', '_')
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{take_dir.name}.pt"
        
        torch.save({
            'landmarks': torch.from_numpy(clip).float(),  # (T, 474)
            'label': label,
            'word': word,
            'take': take_dir.name,
            'num_frames': clip.shape[0],
            'dims': 474
        }, out_path)
        
        return (word, take_dir.name, True, None)
    
    except Exception as e:
        return (word, take_dir.name, False, str(e))

# ──────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────

def build_task_list():
    """Build list of (take_dir, word, label, word_dir, max_frames) for all clips."""
    csv_path = Path(r"D:\mookvani\Backend\data\ISL_CSLRT_Corpus\corpus_csv_files\ISL_CSLRT_Corpus_word_details.xlsx")
    df = pd.read_excel(csv_path)
    words = sorted(df['Word'].unique())
    
    source_root = Path(r"D:\mookvani\Backend\data\ISL_CSLRT_Corpus\Frames_Word_Level")
    
    tasks = []
    for label, word in enumerate(words):
        folder_name = word.upper().replace(' ', '_')
        word_dir = Path(r"D:\mookvani\Backend\data\ISL_CSLRT_Corpus\Frames_Word_Level") / folder_name
        
        if not word_dir.exists():
            continue
        
        # Find take directories
        takes = [d for d in word_dir.iterdir() if d.is_dir()]
        if not takes:
            # Flat structure
            tasks.append((word_dir, word, label, word_dir, MAX_FRAMES))
        else:
            for take_dir in takes:
                tasks.append((take_dir, word, label, word_dir, MAX_FRAMES))
    
    return tasks

def extract_parallel():
    """Main parallel extraction function."""
    tasks = build_task_list()
    print(f"Total clips to extract: {len(tasks)}")
    print(f"Using {N_WORKERS} workers...")
    
    # Process in parallel
    with Pool(processes=N_WORKERS, initializer=init_worker) as pool:
        results = list(tqdm(pool.imap(extract_single_clip, tasks), total=len(tasks), desc="Extracting"))
    
    # Report results
    success = sum(1 for r in results if r[2])
    failed = [(w, t, e) for w, t, s, e in results if not s]
    
    print(f"\n{'='*50}")
    print(f"Extraction complete: {success}/{len(results)} succeeded")
    if failed:
        print(f"Failed ({len(failed)}):")
        for w, t, e in failed[:10]:
            print(f"  {w}/{t}: {e}")
    print(f"{'='*50}")

if __name__ == "__main__":
    extract_parallel()