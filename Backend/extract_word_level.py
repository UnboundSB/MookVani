"""
Frame-wise Word-Level Feature Extraction Pipeline
==================================================

Extracts above-waist + face boundary landmarks from ISL CSLRT word-level frames.

Output: mookvani_dataset/islcsltr_corpus_extracted/<word>/<take>.npy
Format: (T, 462) float32 — 154 landmarks × 3 (x,y,z) — NO FLAGS

Constants at top — modify these to change behavior.
"""

# ──────────────────────────────────────────────────────────────────
# CONSTANTS — MODIFY THESE TO CHANGE BEHAVIOR
# ──────────────────────────────────────────────────────────────────

# Paths
SOURCE_FRAMES_ROOT = r"D:\mookvani\Backend\data\ISL_CSLRT_Corpus\Frames_Word_Level"
OUTPUT_ROOT = r"D:\mookvani\mookvani_dataset\islcsltr_corpus_extracted"
MODEL_ROOT = r"D:\mookvani"  # MediaPipe .task files location

# MediaPipe model files (must exist in MODEL_ROOT)
HAND_MODEL = "hand_landmarker.task"
POSE_MODEL = "pose_landmarker.task"
FACE_MODEL = "face_landmarker.task"

# Processing
MAX_FRAMES = 120           # Max frames per clip (pad/truncate)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# Landmark counts (constants — DO NOT CHANGE)
N_HAND_LANDMARKS = 21
N_ARM_LANDMARKS = 4
N_FACE_LANDMARKS = 108  # face boundary landmarks
N_TOTAL_LANDMARKS = 2 * N_HAND_LANDMARKS + N_ARM_LANDMARKS + N_FACE_LANDMARKS  # 154
N_DIMS = N_TOTAL_LANDMARKS * 3  # 462 (NO FLAGS)

# MediaPipe indices (constants)
HAND_LANDMARK_NAMES = [
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
]

POSE_UPPER = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
}

# FaceMesh boundary indices
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LEFT_EYE_CORNERS = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]
LEFT_EYE_IRIS = [468, 469, 470, 471, 472]
RIGHT_EYE_IRIS = [473, 474, 475, 476, 477]
LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 193]
RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 417]
NOSE_TIP = [1, 2, 4, 5]
NOSE_BRIDGE = [6, 19, 20]
MOUTH_OUTER = [61, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
MOUTH_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
LEFT_EAR = [234, 127, 162, 21, 54, 103, 67, 109]
RIGHT_EAR = [454, 356, 389, 251, 284, 332, 297, 338]

FACE_BOUNDARY_INDICES = sorted(set(
    FACE_OVAL + LEFT_EYE_CORNERS + RIGHT_EYE_CORNERS +
    LEFT_EYE_IRIS + RIGHT_EYE_IRIS + LEFT_EYEBROW + RIGHT_EYEBROW +
    NOSE_TIP + NOSE_BRIDGE + MOUTH_OUTER + MOUTH_INNER +
    LEFT_EAR + RIGHT_EAR
))

# ──────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import mediapipe as mp

# ──────────────────────────────────────────────────────────────────
# LANDMARK EXTRACTION
# ──────────────────────────────────────────────────────────────────

def extract_frame_landmarks(
    hands_result,
    pose_result,
    face_result
) -> np.ndarray:
    """
    Extract 462-dim feature vector from MediaPipe results (NO FLAGS).
    
    Returns:
        feat: (462,) float32 — 154 landmarks × 3 (x,y,z)
    """
    feat = np.zeros(462, dtype=np.float32)  # 154 × 3
    
    # ─── Hands (0-41) → indices 0-41 ───
    if hands_result and hands_result.hand_landmarks:
        for i, handedness in enumerate(hands_result.handedness):
            label = handedness[0].category_name  # "Left" or "Right"
            lms = hands_result.hand_landmarks[i]
            offset = 0 if label == "Left" else 21
            for j, lm in enumerate(lms):
                base = (offset + j) * 3
                feat[base:base+3] = [lm.x, lm.y, lm.z]
    
    # ─── Arms (42-45) → indices 42-45 ───
    if pose_result and pose_result.pose_landmarks:
        plm = pose_result.pose_landmarks[0]
        for j, (_, idx) in enumerate(POSE_UPPER.items()):
            lm = plm[idx]
            base = (42 + j) * 3
            feat[base:base+3] = [lm.x, lm.y, lm.z]
    
    # ─── Face Boundary (46-92) → indices 46-92 ───
    if face_result and face_result.face_landmarks:
        flm = face_result.face_landmarks[0]
        for j, idx in enumerate(FACE_BOUNDARY_INDICES):
            lm = flm[idx]
            base = (46 + j) * 3
            feat[base:base+3] = [lm.x, lm.y, lm.z]
    
    return feat  # (462,) — NO FLAGS


def extract_video_clip(
    frames_dir: Path,
    max_frames: int = MAX_FRAMES
) -> np.ndarray:
    """
    Extract features from a directory of frame images.
    
    Args:
        frames_dir: Directory containing frame images (.jpg/.png)
        max_frames: Max frames to extract (pad/truncate)
    
    Returns:
        clip: (T, 462) float32 — T = min(num_frames, max_frames)
    """
    # Collect frame files
    frame_files = []
    for ext in IMAGE_EXTENSIONS:
        frame_files.extend(sorted(frames_dir.glob(f"*{ext}")))
    
    if not frame_files:
        return np.zeros((max_frames, 462), dtype=np.float32)
    
    # Uniformly sample if too many frames
    if len(frame_files) > max_frames:
        indices = np.linspace(0, len(frame_files) - 1, max_frames, dtype=int)
        frame_files = [frame_files[i] for i in indices]
    
    # Initialize MediaPipe (IMAGE mode for frames)
    BaseOptions = mp.tasks.BaseOptions
    Mode = mp.tasks.vision.RunningMode.IMAGE
    
    hand_opt = mp.tasks.vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(MODEL_ROOT, HAND_MODEL)),
        running_mode=Mode, num_hands=2)
    pose_opt = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(MODEL_ROOT, POSE_MODEL)),
        running_mode=Mode)
    face_opt = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(MODEL_ROOT, FACE_MODEL)),
        running_mode=Mode)
    
    clip = []
    
    with mp.tasks.vision.HandLandmarker.create_from_options(hand_opt) as hand_lm, \
         mp.tasks.vision.PoseLandmarker.create_from_options(pose_opt) as pose_lm, \
         mp.tasks.vision.FaceLandmarker.create_from_options(face_opt) as face_lm:
        
        for frame_path in frame_files[:max_frames]:
            img = cv2.imread(str(frame_path))
            if img is None:
                continue
            
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            hands = hand_lm.detect(mp_img)
            pose = pose_lm.detect(mp_img)
            face = face_lm.detect(mp_img)
            
            feat = extract_frame_landmarks(hands, pose, face)
            clip.append(feat)
    
    if not clip:
        return np.zeros((max_frames, 462), dtype=np.float32)
    
    clip = np.array(clip, dtype=np.float32)  # (T, 462)
    T = clip.shape[0]
    
    # Pad if needed
    if T < max_frames:
        pad = np.zeros((max_frames - T, 462), dtype=np.float32)
        clip = np.vstack([clip, pad])
    
    return clip  # (max_frames, 462)


# ──────────────────────────────────────────────────────────────────
# MAIN EXTRACTION PIPELINE
# ──────────────────────────────────────────────────────────────────

def extract_word_level_corpus():
    """
    Main pipeline: Extract all word-level clips from Frames_Word_Level.
    
    Output structure:
        OUTPUT_ROOT/
            word_name/
                take_name.npy    # (MAX_FRAMES, 462) float32
    """
    source_root = Path(SOURCE_FRAMES_ROOT)
    output_root = Path(OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Get word classes from CSV
    csv_path = Path(r"D:\mookvani\Backend\data\ISL_CSLRT_Corpus\corpus_csv_files\ISL_CSLRT_Corpus_word_details.xlsx")
    import pandas as pd
    df = pd.read_excel(csv_path)
    words = sorted(df['Word'].unique())
    print(f"Found {len(words)} word classes")
    
    # Process each word
    for word in words:
        # Map word to folder name (UPPERCASE_UNDERSCORES)
        folder_name = word.upper().replace(' ', '_')
        word_dir = Path(SOURCE_FRAMES_ROOT) / folder_name
        
        if not word_dir.exists():
            print(f"  Skip {word} (folder {folder_name} not found)")
            continue
        
        out_dir = output_root / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Find take directories
        takes = [d for d in word_dir.iterdir() if d.is_dir()]
        if not takes:
            # Flat structure: images directly in word folder
            clip = extract_video_clip(word_dir)
            out_path = out_dir / f"{folder_name}.npy"
            np.save(out_path, clip)
            print(f"  {folder_name}: 1 clip -> {out_path}")
            continue
        
        # Nested takes
        for take_dir in takes:
            clip = extract_video_clip(take_dir)
            out_path = out_dir / f"{take_dir.name}.npy"
            np.save(out_path, clip)
        
        print(f"  {folder_name}: {len(takes)} clips -> {out_dir}")


# ──────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import cv2
    import os
    
    print("=" * 60)
    print("Frame-wise Word-Level Feature Extraction")
    print(f"Source: {SOURCE_FRAMES_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Max frames: {MAX_FRAMES}")
    print(f"Output dims: {N_DIMS} (no flags)")
    print("=" * 60)
    
    extract_word_level_corpus()
    
    print("\nDone!")