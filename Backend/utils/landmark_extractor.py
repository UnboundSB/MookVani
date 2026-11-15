"""
MediaPipe Holistic Landmark Extraction Utility
===============================================

Reusable functions for frame-wise landmark extraction with auto-sharpening fallback.
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Optional, Tuple, List
import os

# ──────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────

MODEL_ROOT = r"D:\mookvani"
HOLISTIC_MODEL = "holistic_landmarker.task"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

MAX_SHARPEN_ATTEMPTS = 5
SHARPEN_KERNEL_BASE = np.array([[-1, -1, -1],
                                [-1,  9, -1],
                                [-1, -1, -1]], dtype=np.float32)

# MediaPipe Holistic landmark indices
N_HAND_LANDMARKS = 21
POSE_UPPER_INDICES = [11, 12, 13, 14, 15, 16, 23, 24]  # shoulders, elbows, wrists, hips
N_POSE_LANDMARKS = len(POSE_UPPER_INDICES)

FACE_BOUNDARY_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    33, 133, 362, 263,
    468, 469, 470, 471, 472, 473, 474, 475, 476, 477,
    70, 63, 105, 66, 107, 55, 193, 300, 293, 334, 296, 336, 285, 417,
    1, 2, 4, 5, 6, 19, 20,
    61, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
    234, 127, 162, 21, 54, 103, 67, 109, 454, 356, 389, 251, 284, 332, 297, 338,
]

N_HAND_LANDMARKS = 21
N_POSE_LANDMARKS = len(POSE_UPPER_INDICES)
N_FACE_LANDMARKS = len(FACE_BOUNDARY_INDICES)
N_TOTAL_LANDMARKS = 2 * 21 + 8 + 124
N_DIMS = (2 * 21 + 8 + 124) * 3

# ──────────────────────────────────────────────────────────────────
# MEDIAPIPE HOLISTIC INITIALIZATION
# ──────────────────────────────────────────────────────────────────

_holistic_landmarker = None

def get_holistic_landmarker(
    min_face_detection_confidence: float = 0.1,
    min_face_presence_confidence: float = 0.1,
    min_hand_landmarks_confidence: float = 0.1,
    min_pose_detection_confidence: float = 0.1,
    min_pose_presence_confidence: float = 0.1
) -> mp.tasks.vision.HolisticLandmarker:
    """Get or create cached HolisticLandmarker instance."""
    global _holistic_landmarker
    
    if _holistic_landmarker is None:
        BaseOptions = mp.tasks.BaseOptions
        Mode = mp.tasks.vision.RunningMode.IMAGE
        
        holistic_opt = mp.tasks.vision.HolisticLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=os.path.join(MODEL_ROOT, "holistic_landmarker.task")
            ),
            running_mode=Mode,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_hand_landmarks_confidence=min_hand_landmarks_confidence,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence)
        
        _holistic_landmarker = mp.tasks.vision.HolisticLandmarker.create_from_options(holistic_opt)
    
    return _holistic_landmarker


# ──────────────────────────────────────────────────────────────────
# SHARPENING FALLBACK
# ──────────────────────────────────────────────────────────────────

def sharpen_image(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Apply unsharp masking to enhance edges."""
    kernel = SHARPEN_KERNEL_BASE * strength
    # Normalize kernel to preserve brightness
    kernel = kernel / kernel.sum() if kernel.sum() != 0 else kernel
    sharpened = cv2.filter2D(image, -1, kernel)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def auto_sharpen_until_detected(
    frame: np.ndarray,
    holistic_lm: mp.tasks.vision.HolisticLandmarker,
    min_face_confidence: float = 0.1,
    max_attempts: int = MAX_SHARPEN_ATTEMPTS,
    verbose: bool = False
) -> Tuple[np.ndarray, mp.tasks.vision.HolisticLandmarkerResult, int]:
    """
    Try to detect landmarks, sharpening progressively until face is detected.
    
    Returns:
        sharpened_frame: The frame that worked (or last attempt)
        result: MediaPipe detection result
        attempts: Number of sharpening attempts made (0 = original worked)
    """
    current_frame = frame.copy()
    
    for attempt in range(max_attempts):
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB))
        
        result = holistic_lm.detect(mp_img)
        
        # Check if face detected
        face_detected = (result.face_landmarks is not None and 
                         len(result.face_landmarks) > 0)
        
        if face_detected:
            if verbose and attempt > 0:
                print(f"  ✓ Face detected after {attempt} sharpening attempt(s)")
            return current_frame, result, attempt
        
        # Not detected - sharpen and retry
        if attempt < max_attempts - 1:
            strength = 1.0 + attempt * 0.5  # 1.0, 1.5, 2.0, 2.5, 3.0
            current_frame = sharpen_image(current_frame, strength)
            if verbose:
                print(f"  Attempt {attempt + 1}: No face, sharpening (strength={strength})...")
    
    # Final attempt with max sharpening
    if verbose:
        print(f"  ⚠ No face detected after {max_attempts} attempts, returning last attempt")
    return current_frame, result, max_attempts


# ──────────────────────────────────────────────────────────────────
# MAIN EXTRACTION FUNCTION
# ──────────────────────────────────────────────────────────────────

def extract_landmarks_from_frame(
    frame: np.ndarray,
    holistic_lm: Optional[mp.tasks.vision.HolisticLandmarker] = None,
    auto_sharpen: bool = True,
    min_face_confidence: float = 0.1,
    verbose: bool = False
) -> Tuple[np.ndarray, mp.tasks.vision.HolisticLandmarkerResult, int]:
    """
    Extract landmarks from a single frame with auto-sharpening fallback.
    
    Args:
        frame: Input frame (BGR, H×W×3)
        holistic_lm: Pre-initialized HolisticLandmarker (creates default if None)
        auto_sharpen: If True, auto-sharpens until face detected
        min_face_confidence: Minimum face detection confidence
        verbose: Print progress
    
    Returns:
        landmarks: (522,) float32 array — 174 landmarks × 3 (x,y,z)
        result: Full MediaPipe result object
        sharpen_attempts: Number of sharpening attempts (0 = no sharpening needed)
    """
    if holistic_lm is None:
        holistic_lm = get_holistic_landmarker()
    
    if auto_sharpen:
        frame_used, result, attempts = auto_sharpen_until_detected(
            frame, holistic_lm, min_face_confidence, verbose=verbose)
    else:
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = holistic_lm.detect(mp_img)
        attempts = 0
    
    landmarks = extract_landmarks_array(result)
    return landmarks, result, attempts


def extract_landmarks_array(result: mp.tasks.vision.HolisticLandmarkerResult) -> np.ndarray:
    """Convert MediaPipe result to flat 522-dim array."""
    feat = np.zeros(522, dtype=np.float32)  # 174 × 3
    
    # Left hand (0-20)
    if result.left_hand_landmarks:
        for j, lm in enumerate(result.left_hand_landmarks):
            base = j * 3
            feat[base:base+3] = [lm.x, lm.y, lm.z]
    
    # Right hand (21-41)
    if result.right_hand_landmarks:
        for j, lm in enumerate(result.right_hand_landmarks):
            base = (21 + j) * 3
            feat[base:base+3] = [lm.x, lm.y, lm.z]
    
    # Pose upper body (8 landmarks)
    if result.pose_landmarks:
        plm = result.pose_landmarks
        for j, pose_idx in enumerate(POSE_UPPER_INDICES):
            lm = plm[pose_idx]
            base = (42 + j) * 3
            feat[base:base+3] = [lm.x, lm.y, lm.z]
    
    # Face boundary (124 landmarks)
    if result.face_landmarks and len(result.face_landmarks) > 0:
        flm = result.face_landmarks
        # Check if we have enough landmarks
        if len(flm) >= max(FACE_BOUNDARY_INDICES) + 1:
            for j, face_idx in enumerate(FACE_BOUNDARY_INDICES):
                lm = flm[face_idx]
                base = (42 + 8 + j) * 3
                feat[base:base+3] = [lm.x, lm.y, lm.z]
        else:
            print(f"Warning: Face landmarks only has {len(flm)} points, expected at least {max(FACE_BOUNDARY_INDICES)+1}")
    
    return feat


# ──────────────────────────────────────────────────────────────────
# VISUALIZATION
# ──────────────────────────────────────────────────────────────────

def draw_landmarks_on_frame(
    frame: np.ndarray,
    result: mp.tasks.vision.HolisticLandmarkerResult,
    draw_face: bool = True,
    draw_hands: bool = True,
    draw_pose: bool = True,
    face_color: Tuple[int, int, int] = (0, 255, 0),
    hand_color: Tuple[int, int, int] = (255, 0, 0),
    pose_color: Tuple[int, int, int] = (0, 0, 255),
    radius: int = 2
) -> np.ndarray:
    """Draw MediaPipe landmarks on frame for visualization."""
    vis = frame.copy()
    h, w = frame.shape[:2]
    
    def draw_points(landmarks, color):
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(vis, (x, y), radius, color, -1)
    
    if draw_face and result.face_landmarks:
        draw_points(result.face_landmarks, face_color)
    
    if draw_hands:
        if result.left_hand_landmarks:
            draw_points(result.left_hand_landmarks, hand_color)
        if result.right_hand_landmarks:
            draw_points(result.right_hand_landmarks, (0, 255, 255))  # Yellow for right
    
    if draw_pose and result.pose_landmarks:
        for idx in POSE_UPPER_INDICES:
            lm = result.pose_landmarks[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(vis, (x, y), radius + 1, pose_color, -1)
    
    return vis


def visualize_frame_with_landmarks(
    frame_path: str,
    holistic_lm: Optional[mp.tasks.vision.HolisticLandmarker] = None,
    output_path: Optional[str] = None,
    auto_sharpen: bool = True
) -> np.ndarray:
    """
    Load frame, extract landmarks with auto-sharpen, visualize and save.
    
    Returns:
        vis_frame: Annotated frame
    """
    if holistic_lm is None:
        holistic_lm = get_holistic_landmarker()
    
    frame = cv2.imread(frame_path)
    if frame is None:
        raise FileNotFoundError(f"Could not load: {frame_path}")
    
    # Extract with auto-sharpen
    landmarks, result, attempts = extract_landmarks_from_frame(
        frame, holistic_lm, auto_sharpen=auto_sharpen, verbose=True)
    
    print(f"Extraction: {attempts} sharpen attempts, landmarks shape: {landmarks.shape}")
    
    # Draw landmarks
    vis = draw_landmarks_on_frame(frame, result)
    
    # Add info text
    cv2.putText(vis, f"Sharpen attempts: {attempts}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis, f"Face: {'Yes' if result.face_landmarks else 'No'}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if result.face_landmarks else (0, 0, 255), 2)
    cv2.putText(vis, f"Pose: {'Yes' if result.pose_landmarks else 'No'}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0) if result.pose_landmarks else (0, 0, 255), 2)
    cv2.putText(vis, f"Hands: L{'Yes' if result.left_hand_landmarks else 'No'} R{'Yes' if result.right_hand_landmarks else 'No'}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    if output_path:
        cv2.imwrite(output_path, vis)
        print(f"Saved visualization to: {output_path}")
    
    return vis


# ──────────────────────────────────────────────────────────────────
# BATCH EXTRACTION (for pipeline use)
# ──────────────────────────────────────────────────────────────────

def extract_video_clip_landmarks(
    frames_dir: Path,
    holistic_lm: Optional[mp.tasks.vision.HolisticLandmarker] = None,
    max_frames: int = 120,
    auto_sharpen: bool = True
) -> np.ndarray:
    """
    Extract landmarks from all frames in a directory.
    
    Returns:
        clip: (max_frames, 522) float32
    """
    if holistic_lm is None:
        holistic_lm = get_holistic_landmarker()
    
    # Collect frame files
    frame_files = []
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        frame_files.extend(sorted(frames_dir.glob(f"*{ext}")))
    
    if not frame_files:
        return np.zeros((max_frames, 522), dtype=np.float32)
    
    # Uniformly sample if too many frames
    if len(frame_files) > max_frames:
        indices = np.linspace(0, len(frame_files) - 1, max_frames, dtype=int)
        frame_files = [frame_files[i] for i in indices]
    
    clip = []
    for frame_path in frame_files[:max_frames]:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        
        landmarks, _, _ = extract_landmarks_from_frame(
            frame, holistic_lm, auto_sharpen=auto_sharpen)
        clip.append(landmarks)
    
    if not clip:
        return np.zeros((max_frames, 522), dtype=np.float32)
    
    clip = np.array(clip, dtype=np.float32)
    T = clip.shape[0]
    
    if T < max_frames:
        pad = np.zeros((max_frames - T, 522), dtype=np.float32)
        clip = np.vstack([clip, pad])
    
    return clip  # (max_frames, 522)


# ──────────────────────────────────────────────────────────────────
# DEMO / TEST
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    
    # Test on a sample frame
    test_frame = r"D:\mookvani\Backend\data\ISL_CSLRT_Corpus\Frames_Word_Level\ABUSE\ABUSE (1).jpg"
    output_vis = r"D:\mookvani\landmark_visualization.jpg"
    
    print("Testing landmark extraction with visualization...")
    vis = visualize_frame_with_landmarks(
        test_frame, 
        output_path=output_vis,
        auto_sharpen=True)
    
    print(f"Visualization saved to: {output_vis}")
    
    # Also test batch extraction
    frames_dir = Path(r"D:\mookvani\Backend\data\ISL_CSLRT_Corpus\Frames_Word_Level\ABUSE")
    clip = extract_video_clip_landmarks(frames_dir)
    print(f"Batch clip shape: {clip.shape}")  # Should be (120, 522)