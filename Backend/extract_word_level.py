"""
Frame-wise Word-Level Feature Extraction Pipeline (Holistic)
============================================================

Extracts landmarks using MediaPipe Holistic model (single unified model).

Output: mookvani_dataset/islcsltr_corpus_extracted/<WORD>/<take>.npy
Format: (T, 504) float32 — 168 landmarks × 3 (x,y,z) — NO FLAGS

Landmarks:
- 21 × 2 hands = 42
- 33 pose (upper body: 11,12,13,14,15,16,23,24) = 8
- 108 face boundary (from 478 FaceMesh) = 108
Total: 168 landmarks × 3 = 504 dims — NO FLAGS

Constants at top — modify these to change behavior.
"""

# ──────────────────────────────────────────────────────────────────
# CONSTANTS — MODIFY THESE TO CHANGE BEHAVIOR
# ──────────────────────────────────────────────────────────────────

# Paths
SOURCE_FRAMES_ROOT = r"D:\mookvani\Backend\data\ISL_CSLRT_Corpus\Frames_Word_Level"
OUTPUT_ROOT = r"D:\mookvani\mookvani_dataset\islcsltr_corpus_extracted"
MODEL_ROOT = r"D:\mookvani"  # MediaPipe .task files location

# MediaPipe model file (Holistic includes face, hands, pose)
HOLISTIC_MODEL = "holistic_landmarker.task"

# Processing
MAX_FRAMES = 120           # Max frames per clip (pad/truncate)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# ──────────────────────────────────────────────────────────────────
# LANDMARK INDICES — MediaPipe Holistic Model
# ──────────────────────────────────────────────────────────────────

# Hands: 21 landmarks each (MediaPipe HandLandmarker standard)
# 0-20: left hand, 21-41: right hand

# Pose (33 landmarks): we use upper body only
POSE_UPPER_INDICES = [11, 12, 13, 14, 15, 16, 23, 24]  # shoulders, elbows, wrists, hips

# Face boundary indices from 478 FaceMesh
FACE_BOUNDARY_INDICES = [
    # Face oval (jawline) - 36 points
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    # Eye corners - 4 points
    33, 133, 362, 263,
    # Iris - 10 points
    468, 469, 470, 471, 472, 473, 474, 475, 476, 477,
    # Eyebrows - 14 points
    70, 63, 105, 66, 107, 55, 193, 300, 293, 334, 296, 336, 285, 417,
    # Nose - 7 points
    1, 2, 4, 5, 6, 19, 20,
    # Mouth outer + inner - 36 points
    61, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
    # Ears - 16 points
    234, 127, 162, 21, 54, 103, 67, 109, 454, 356, 389, 251, 284, 332, 297, 338,
]

# Verify counts
N_HAND_LANDMARKS = 21
N_POSE_LANDMARKS = len(POSE_UPPER_INDICES)  # 8
N_FACE_LANDMARKS = len(FACE_BOUNDARY_INDICES)  # 108
N_TOTAL_LANDMARKS = 2 * N_HAND_LANDMARKS + N_POSE_LANDMARKS + N_FACE_LANDMARKS  # 42 + 8 + 108 = 158
N_DIMS = N_TOTAL_LANDMARKS * 3  # 474 (NO FLAGS)

print(f"Landmark counts: hands={2*N_HAND_LANDMARKS}, pose={N_POSE_LANDMARKS}, face={N_FACE_LANDMARKS}, total={N_TOTAL_LANDMARKS}, dims={N_DIMS}")

# ──────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional
import mediapipe as mp
import pandas as pd

# ──────────────────────────────────────────────────────────────────
# LANDMARK EXTRACTION
# ──────────────────────────────────────────────────────────────────

def extract_frame_landmarks(
    holistic_result
) -> np.ndarray:
    """
    Extract feature vector from MediaPipe Holistic result.
    
    Returns:
        feat: (474,) float32 — 158 landmarks × 3 (x,y,z) — NO FLAGS
    """
    feat = np.zeros(N_DIMS, dtype=np.float32)  # 158 × 3 = 474
    idx = 0
    
    # ─── Hands (0-41) ───
    # Left hand (0-20)
    if holistic_result.left_hand_landmarks:
        for j, lm in enumerate(holistic_result.left_hand_landmarks):
            base = j * 3
            feat[base:base+3] = [lm.x, lm.y, lm.z]
    idx += N_HAND_LANDMARKS * 3
    
    # Right hand (21-41)
    if holistic_result.right_hand_landmarks:
        for j, lm in enumerate(holistic_result.right_hand_landmarks):
            base = (N_HAND_LANDMARKS + j) * 3
            feat[base:base+3] = [lm.x, lm.y, lm.z]
    idx += N_HAND_LANDMARKS * 3
    
    # ─── Pose Upper Body (8 landmarks) ───
    if holistic_result.pose_landmarks:
        plm = holistic_result.pose_landmarks
        for j, pose_idx in enumerate(POSE_UPPER_INDICES):
            lm = plm[pose_idx]
            base = (2 * N_HAND_LANDMARKS + j) * 3
            feat[base:base+3] = [lm.x, lm.y, lm.z]
    idx += N_POSE_LANDMARKS * 3
    
    # ─── Face Boundary (108 landmarks) ───
    if holistic_result.face_landmarks:
        flm = holistic_result.face_landmarks
        for j, face_idx in enumerate(FACE_BOUNDARY_INDICES):
            lm = flm[face_idx]
            base = (2 * N_HAND_LANDMARKS + N_POSE_LANDMARKS + j) * 3
            feat[base:base+3] = [lm.x, lm.y, lm.z]
    
    return feat  # (474,) — NO FLAGS


def extract_video_clip(
    frames_dir: Path,
    holistic_lm,
    max_frames: int = MAX_FRAMES
) -> np.ndarray:
    """
    Extract features from a directory of frame images.
    
    Args:
        frames_dir: Directory containing frame images (.jpg/.png)
        holistic_lm: Initialized HolisticLandmarker
        max_frames: Max frames to extract (pad/truncate)
    
    Returns:
        clip: (max_frames, 474) float32
    """
    # Collect frame files
    frame_files = []
    for ext in IMAGE_EXTENSIONS:
        frame_files.extend(sorted(frames_dir.glob(f"*{ext}")))
    
    if not frame_files:
        return np.zeros((max_frames, N_DIMS), dtype=np.float32)
    
    # Uniformly sample if too many frames
    if len(frame_files) > max_frames:
        indices = np.linspace(0, len(frame_files) - 1, max_frames, dtype=int)
        frame_files = [frame_files[i] for i in indices]
    
    clip = []
    
    for frame_path in frame_files[:max_frames]:
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        result = holistic_lm.detect(mp_img)
        feat = extract_frame_landmarks(result)
        clip.append(feat)
    
    if not clip:
        return np.zeros((max_frames, N_DIMS), dtype=np.float32)
    
    clip = np.array(clip, dtype=np.float32)  # (T, 474)
    T = clip.shape[0]
    
    # Pad if needed
    if T < max_frames:
        pad = np.zeros((max_frames - T, N_DIMS), dtype=np.float32)
        clip = np.vstack([clip, pad])
    
    return clip  # (max_frames, 474)


# ──────────────────────────────────────────────────────────────────
# MAIN EXTRACTION PIPELINE
# ──────────────────────────────────────────────────────────────────

def extract_word_level_corpus():
    """
    Main pipeline: Extract all word-level clips using Holistic model.
    
    Output structure:
        OUTPUT_ROOT/
            WORD_NAME/
                take_name.npy    # (MAX_FRAMES, 474) float32
    """
    source_root = Path(SOURCE_FRAMES_ROOT)
    output_root = Path(OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Get word classes from CSV
    csv_path = Path(r"D:\mookvani\Backend\data\ISL_CSLRT_Corpus\corpus_csv_files\ISL_CSLRT_Corpus_word_details.xlsx")
    df = pd.read_excel(csv_path)
    words = sorted(df['Word'].unique())
    print(f"Found {len(words)} word classes")
    
    # Initialize Holistic model once
    BaseOptions = mp.tasks.BaseOptions
    Mode = mp.tasks.vision.RunningMode.IMAGE
    holistic_opt = mp.tasks.vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(MODEL_ROOT, HOLISTIC_MODEL)),
        running_mode=Mode,
        min_face_detection_confidence=0.1,
        min_face_presence_confidence=0.1,
        min_hand_landmarks_confidence=0.1,
        min_pose_detection_confidence=0.1,
        min_pose_presence_confidence=0.1)
    
    print("Initializing MediaPipe Holistic model...")
    
    with mp.tasks.vision.HolisticLandmarker.create_from_options(holistic_opt) as holistic_lm:
        print("Model initialized. Starting extraction...")
        
        # Process each word
        for word in words:
            folder_name = word.upper().replace(' ', '_')
            word_dir = Path(SOURCE_FRAMES_ROOT) / folder_name
            
            if not word_dir.exists():
                print(f"  Skip {word} (folder {folder_name} not found)")
                continue
            
            out_dir = Path(OUTPUT_ROOT) / folder_name
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Find take directories
            takes = [d for d in word_dir.iterdir() if d.is_dir()]
            if not takes:
                # Flat structure: images directly in word folder
                clip = extract_video_clip(word_dir, holistic_lm)
                out_path = out_dir / f"{folder_name}.npy"
                np.save(out_path, clip)
                print(f"  {folder_name}: 1 clip -> {out_path}")
                continue
            
            # Nested takes
            for take_dir in takes:
                clip = extract_video_clip(take_dir, holistic_lm)
                out_path = out_dir / f"{take_dir.name}.npy"
                np.save(out_path, clip)
            
            print(f"  {folder_name}: {len(takes)} clips -> {out_dir}")
    
    print("\nDone!")


# ──────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import cv2
    import os
    
    print("=" * 60)
    print("Frame-wise Word-Level Feature Extraction (Holistic)")
    print(f"Source: {SOURCE_FRAMES_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Max frames: {MAX_FRAMES}")
    print(f"Output dims: {N_DIMS} (no flags)")
    print(f"Landmarks: {2*N_HAND_LANDMARKS} hands + {N_POSE_LANDMARKS} pose + {N_FACE_LANDMARKS} face = {N_TOTAL_LANDMARKS} total")
    print("=" * 60)
    
    extract_word_level_corpus()