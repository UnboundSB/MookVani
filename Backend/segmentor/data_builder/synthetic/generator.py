"""
Synthetic Sentence Generator — Creates realistic ISL sentences from word .npy features.
Includes visualization to inspect joint connections.
"""

import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import cv2
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from scipy.interpolate import interp1d
import mediapipe as mp

# ────────────────────────── paths ──────────────────────────
PKG_DIR = Path(__file__).resolve().parent
APP_DIR = PKG_DIR.parent
FEATURES_DIR = APP_DIR / "conformer_features"
WORD_INVENTORY_JSON = APP_DIR / "word_inventory.json"
CLASSES_JSON = APP_DIR / "classes.json"
CLASSES_JSON_FALLBACK = Path(r"D:\MookVani\Backend\models\word_vocabulary\classes.json")

# ────────────────────────── constants ──────────────────────────
# Skeleton connections for visualization (MediaPipe 163-dim layout)
# Layout: [0:63] lh (21x3), [63:126] rh (21x3), [126:138] arms (4x3), [138:159] face (7x3), [159:163] flags
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # thumb
    (0,5),(5,6),(6,7),(7,8),        # index
    (0,9),(9,10),(10,11),(11,12),   # middle
    (0,13),(13,14),(14,15),(15,16), # ring
    (0,17),(17,18),(18,19),(19,20)  # pinky
]
ARM_CONNECTIONS = [(0,1), (1,3), (0,2), (2,3)]  # shoulders, elbows
FACE_CONNECTIONS = [(0,2), (1,2), (2,3), (2,4), (3,4), (5,6)]  # eyes, nose, mouth

# ────────────────────────── load resources ──────────────────────────
def load_word_inventory() -> dict:
    for p in (WORD_INVENTORY_JSON, Path(r"D:\MookVani\Backend\segmentor\data_builder\word_inventory.json")):
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError("word_inventory.json not found")

def load_classes() -> List[str]:
    for p in (CLASSES_JSON, CLASSES_JSON_FALLBACK):
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError("classes.json not found")

# ────────────────────────── core utilities ──────────────────────────
def find_motion_onset(features: np.ndarray, threshold_factor: float = 0.05) -> int:
    """Find first frame where hand velocity exceeds threshold."""
    # Hands: first 63 dims = 21 landmarks x 3
    hands = features[:, :63].reshape(-1, 21, 3)
    vel = np.linalg.norm(np.diff(hands, axis=1), axis=-1).mean(axis=1)
    if len(vel) == 0:
        return 0
    onset = np.where(vel > vel.max() * threshold_factor)[0]
    return onset[0] if len(onset) else 0

def time_warp(features: np.ndarray, sigma: float = 0.2) -> np.ndarray:
    """Apply Gaussian time warping to features."""
    T = len(features)
    if T < 2:
        return features
    t_orig = np.arange(T)
    t_warp = t_orig + np.random.normal(0, sigma * T, T)
    t_warp = np.clip(t_warp, 0, T - 1)
    # Interpolate each dimension
    f = interp1d(t_orig, features, axis=0, kind='linear', fill_value='extrapolate', assume_sorted=False)
    return f(t_warp).astype(np.float32)

def load_word_features(word: str) -> List[np.ndarray]:
    """Load all .npy files for a word from conformer_features."""
    word_dir = FEATURES_DIR / word.replace(" ", "_")
    if not word_dir.exists():
        return []
    files = list(word_dir.glob("*.npy"))
    feats = []
    for f in files:
        try:
            arr = np.load(f)
            if arr.ndim == 2 and arr.shape[1] == 163:
                feats.append(arr)
        except Exception:
            pass
    return feats

def extract_word_features_from_video(word: str, inventory: dict) -> List[np.ndarray]:
    """Extract MediaPipe features from archive word videos (fallback when no .npy exists)."""
    if word not in inventory:
        return []
    
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    # Load model (cached)
    if not hasattr(extract_word_features_from_video, '_landmarker'):
        model_path = PKG_DIR / "holistic_landmarker.task"
        if not model_path.exists():
            import urllib.request
            model_url = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task"
            print(f"[EXTRACT] Downloading MediaPipe model...")
            urllib.request.urlretrieve(model_url, model_path)
        
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.HolisticLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_face_detection_confidence=0.5,
            min_face_landmarks_confidence=0.5,
            min_pose_detection_confidence=0.5,
            min_pose_landmarks_confidence=0.5,
            min_hand_landmarks_confidence=0.5,
        )
        extract_word_features_from_video._landmarker = vision.HolisticLandmarker.create_from_options(options)
    
    landmarker = extract_word_features_from_video._landmarker
    
    video_paths = inventory[word]["videos"]
    all_feats = []
    
    for vp in video_paths[:3]:  # Max 3 videos per word
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            continue
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval_ms = int(1000 / fps)
        
        feats = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            # Timestamp must be monotonically increasing across ALL frames
            # Use a global frame counter
            if not hasattr(extract_word_features_from_video, '_global_frame_idx'):
                extract_word_features_from_video._global_frame_idx = 0
            timestamp_ms = extract_word_features_from_video._global_frame_idx * frame_interval_ms
            extract_word_features_from_video._global_frame_idx += 1
            
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            feat = _extract_frame_from_result(result)
            feats.append(feat)
            frame_idx += 1
        
        cap.release()
        if feats:
            all_feats.append(np.stack(feats, axis=0).astype(np.float32))
    
    return all_feats

def _extract_frame_from_result(result) -> np.ndarray:
    """Extract 163-dim feature from HolisticLandmarkerResult (same as extraction.py)."""
    lh = np.zeros((21, 3)); rh = np.zeros((21, 3))
    lh_flag = 0.0; rh_flag = 0.0
    
    if result.left_hand_landmarks:
        # result.left_hand_landmarks is a list of 21 NormalizedLandmark objects
        lh = np.array([[lm.x, lm.y, lm.z] for lm in result.left_hand_landmarks])
        lh_flag = 1.0
    if result.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in result.right_hand_landmarks])
        rh_flag = 1.0
    
    arms = np.zeros((4, 3)); arms_flag = 0.0
    if result.pose_landmarks:
        p = result.pose_landmarks  # list of 33 NormalizedLandmark
        arms = np.array([
            [p[11].x, p[11].y, p[11].z], [p[12].x, p[12].y, p[12].z],
            [p[13].x, p[13].y, p[13].z], [p[14].x, p[14].y, p[14].z],
        ])
        arms_flag = 1.0
    
    face_pts = np.zeros((7, 3)); face_flag = 0.0
    if result.face_landmarks:
        f = result.face_landmarks  # list of 468 NormalizedLandmark
        face_pts = np.array([
            [f[159].x, f[159].y, f[159].z], [f[386].x, f[386].y, f[386].z],
            [f[4].x, f[4].y, f[4].z],
            [f[234].x, f[234].y, f[234].z], [f[454].x, f[454].y, f[454].z],
            [f[61].x, f[61].y, f[61].z],    [f[291].x, f[291].y, f[291].z],
        ])
        face_flag = 1.0
    
    # Normalize (copied from extraction.py)
    anchor = np.array([0.5, 0.5, 0.0])
    if face_flag:
        anchor = face_pts[2]
    elif arms_flag:
        anchor = (arms[0] + arms[1]) / 2.0
    elif lh_flag:
        anchor = lh[0]
    elif rh_flag:
        anchor = rh[0]
    
    scale = 0.3
    if arms_flag:
        pose_scale = np.linalg.norm(arms[0] - arms[1])
        if pose_scale > 0.05:
            scale = pose_scale
    elif face_flag:
        face_scale = np.linalg.norm(face_pts[3] - face_pts[4]) * 3.0
        if face_scale > 0.05:
            scale = face_scale
    
    lh_out = (lh - anchor) / scale if lh_flag else np.ones((21, 3)) * np.array([-1.5 * scale, 3.0 * scale, 0.0])
    rh_out = (rh - anchor) / scale if rh_flag else np.ones((21, 3)) * np.array([1.5 * scale, 3.0 * scale, 0.0])
    arms_out = (arms - anchor) / scale if arms_flag else np.zeros((4, 3))
    face_out = (face_pts - anchor) / scale if face_flag else np.zeros((7, 3))
    
    return np.concatenate([lh_out.flatten(), rh_out.flatten(), arms_out.flatten(), face_out.flatten(),
                           [lh_flag, rh_flag, arms_flag, face_flag]]).astype(np.float32)

# ────────────────────────── augmentation pipeline ──────────────────────────
def trim_frames(sentence: np.ndarray, boundaries: List[int], 
                trim_start: int = 10, trim_end: int = 10) -> Tuple[np.ndarray, List[int]]:
    """Remove frames from start and end of sentence, adjust boundaries."""
    T = len(sentence)
    if trim_start + trim_end >= T:
        return sentence, boundaries
    new_sentence = sentence[trim_start:T-trim_end]
    new_boundaries = [b - trim_start for b in boundaries if trim_start <= b < T - trim_end]
    return new_sentence, new_boundaries

def horizontal_flip(sentence: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Flip skeleton horizontally (mirror left/right)."""
    if random.random() > p:
        return sentence
    
    flipped = sentence.copy()
    # Swap LH and RH hands (0:63 <-> 63:126)
    lh = flipped[:, 0:63].copy()
    rh = flipped[:, 63:126].copy()
    flipped[:, 0:63] = rh
    flipped[:, 63:126] = lh
    
    # Swap LH/RH flags (indices 159 <-> 160)
    flags = flipped[:, 159:163].copy()
    flipped[:, 159] = flags[:, 1]  # lh_flag = rh_flag
    flipped[:, 160] = flags[:, 0]  # rh_flag = lh_flag
    
    # Swap arms shoulders/elbows (0<->1, 2<->3)
    arms = flipped[:, 126:138].reshape(-1, 4, 3)
    arms[:, [0,1]] = arms[:, [1,0]]  # swap shoulders
    arms[:, [2,3]] = arms[:, [3,2]]  # swap elbows
    flipped[:, 126:138] = arms.reshape(-1, 12)
    
    # Mirror x-coordinates for all landmarks
    # LH x: flip sign (index 0 of each 3)
    flipped[:, 0:63:3] *= -1
    # RH x: flip sign
    flipped[:, 63:126:3] *= -1
    # Arms x: flip sign
    flipped[:, 126:138:3] *= -1
    # Face x: flip sign (indices 138, 141, 144, 147, 150, 153, 156)
    flipped[:, 138:159:3] *= -1
    
    return flipped

def horizontal_shift(sentence: np.ndarray, max_shift: float = 0.15, p: float = 0.5) -> np.ndarray:
    """Shift skeleton horizontally by random amount."""
    if random.random() > p:
        return sentence
    
    shifted = sentence.copy()
    shift = random.uniform(-max_shift, max_shift)
    
    # Shift x-coordinates for all landmarks
    shifted[:, 0:63:3] += shift
    shifted[:, 63:126:3] += shift
    shifted[:, 126:138:3] += shift
    shifted[:, 138:159:3] += shift
    
    return shifted

def vertical_shift(sentence: np.ndarray, max_shift: float = 0.1, p: float = 0.3) -> np.ndarray:
    """Shift skeleton vertically by random amount."""
    if random.random() > p:
        return sentence
    
    shifted = sentence.copy()
    shift = random.uniform(-max_shift, max_shift)
    
    # Shift y-coordinates for all landmarks
    shifted[:, 1:63:3] += shift
    shifted[:, 64:126:3] += shift
    shifted[:, 127:138:3] += shift
    shifted[:, 139:159:3] += shift
    
    return shifted

def apply_augmentations(sentence: np.ndarray, boundaries: List[int],
                        trim_range: Tuple[int, int] = (10, 15),
                        flip_p: float = 0.5,
                        hshift_p: float = 0.5,
                        vshift_p: float = 0.3) -> Tuple[np.ndarray, List[int]]:
    """Apply full augmentation pipeline to a sentence."""
    
    # 1. Trim frames from start and end
    trim_start = random.randint(*trim_range)
    trim_end = random.randint(*trim_range)
    sentence, boundaries = trim_frames(sentence, boundaries, trim_start, trim_end)
    
    # 2. Horizontal flip (mirror)
    sentence = horizontal_flip(sentence, flip_p)
    
    # 3. Horizontal shift
    sentence = horizontal_shift(sentence, p=hshift_p)
    
    # 4. Vertical shift
    sentence = vertical_shift(sentence, p=vshift_p)
    
    return sentence, boundaries

# ────────────────────────── sentence generation ──────────────────────────
def generate_synthetic_sentence(
    words: List[str],
    sigma: float = 0.2,
    prep_frames: int = 5,
    min_word_frames: int = 8,
    use_archive_fallback: bool = True,
    # Augmentation params
    trim_range: Tuple[int, int] = (10, 15),
    flip_p: float = 0.5,
    hshift_p: float = 0.5,
    vshift_p: float = 0.3
) -> Tuple[np.ndarray, List[int]]:
    """
    Generate a synthetic sentence from word features.
    
    Returns:
        sentence: (T, 163) array
        boundaries: list of frame indices where each word starts
    """
    inventory = load_word_inventory()
    warped_words = []
    boundaries = [0]
    
    for word in words:
        # Get available recordings for this word
        word_feats = load_word_features(word)
        
        # Fallback: extract from archive videos
        if not word_feats and use_archive_fallback:
            print(f"[INFO] No .npy for '{word}', extracting from archive...")
            word_feats = extract_word_features_from_video(word, inventory)
        
        if not word_feats:
            print(f"[WARN] No features for '{word}', skipping")
            continue
        
        # Pick random recording
        w = random.choice(word_feats)
        
        # Find motion onset and include prep frames
        onset = find_motion_onset(w)
        start = min(onset + prep_frames, len(w) - min_word_frames)
        w = w[start:]
        
        if len(w) < min_word_frames:
            continue
        
        # Time warp
        w = time_warp(w, sigma)
        
        warped_words.append(w)
        boundaries.append(boundaries[-1] + len(w))
    
    if not warped_words:
        raise ValueError("No valid word features found")
    
    sentence = np.vstack(warped_words)
    
    # Apply augmentations
    sentence, boundaries = apply_augmentations(
        sentence, boundaries[:-1],
        trim_range=trim_range,
        flip_p=flip_p,
        hshift_p=hshift_p,
        vshift_p=vshift_p
    )
    
    return sentence, boundaries

def generate_random_sentence(
    num_words: Optional[int] = None,
    sigma: float = 0.2,
    prep_frames: int = 5,
    trim_range: Tuple[int, int] = (10, 15),
    flip_p: float = 0.5,
    hshift_p: float = 0.5,
    vshift_p: float = 0.3
) -> Tuple[np.ndarray, List[int], List[str]]:
    """Generate a random sentence with 2-5 words from vocabulary."""
    classes = load_classes()
    inventory = load_word_inventory()
    available = [w for w in classes if w in inventory and load_word_features(w)]
    
    if num_words is None:
        num_words = random.randint(2, 5)
    
    words = random.sample(available, min(num_words, len(available)))
    sentence, boundaries = generate_synthetic_sentence(
        words, sigma, prep_frames, 
        trim_range=trim_range,
        flip_p=flip_p,
        hshift_p=hshift_p,
        vshift_p=vshift_p
    )
    return sentence, boundaries, words

# ────────────────────────── visualization ──────────────────────────
class SkeletonVisualizer:
    """Stateful skeleton visualizer that retains previous coordinates for missing landmarks."""
    
    def __init__(self, img_w: int = 1280, img_h: int = 720):
        self.img_w = img_w
        self.img_h = img_h
        self.prev_lh = None
        self.prev_rh = None
        self.prev_arms = None
        self.prev_face = None
        self.prev_flags = None
    
    def to_pixel(self, pts_3d):
        if len(pts_3d) == 0:
            return np.zeros((0, 2), dtype=int)
        x = np.clip((pts_3d[:, 0] + 2.0) / 4.0 * self.img_w, 0, self.img_w - 1).astype(int)
        y = np.clip((pts_3d[:, 1] + 1.0) / 6.0 * self.img_h, 0, self.img_h - 1).astype(int)
        return np.stack([x, y], axis=1)
    
    def update(self, features: np.ndarray, frame_idx: int) -> dict:
        """Extract keypoints, retaining previous coords for missing landmarks."""
        f = features[frame_idx]
        
        # Raw normalized coords
        lh_raw = f[0:63].reshape(21, 3)
        rh_raw = f[63:126].reshape(21, 3)
        arms_raw = f[126:138].reshape(4, 3)
        face_raw = f[138:159].reshape(7, 3)
        flags = f[159:163]  # [lh_flag, rh_flag, arms_flag, face_flag]
        
        # Update or retain previous
        lh_flag, rh_flag, arms_flag, face_flag = flags
        
        if lh_flag > 0.5:
            self.prev_lh = self.to_pixel(lh_raw)
        if rh_flag > 0.5:
            self.prev_rh = self.to_pixel(rh_raw)
        if arms_flag > 0.5:
            self.prev_arms = self.to_pixel(arms_raw)
        if face_flag > 0.5:
            self.prev_face = self.to_pixel(face_raw)
        
        self.prev_flags = flags
        
        # Return current or retained
        lh = self.prev_lh if self.prev_lh is not None else np.zeros((21, 2), dtype=int)
        rh = self.prev_rh if self.prev_rh is not None else np.zeros((21, 2), dtype=int)
        arms = self.prev_arms if self.prev_arms is not None else np.zeros((4, 2), dtype=int)
        face = self.prev_face if self.prev_face is not None else np.zeros((7, 2), dtype=int)
        
        return {'lh': lh, 'rh': rh, 'arms': arms, 'face': face, 'flags': flags}

def extract_keypoints_2d(features: np.ndarray, frame_idx: int, img_w: int = 1280, img_h: int = 720) -> dict:
    """Stateless extractor (for preview frames)."""
    # Create temporary visualizer for one-off extraction
    viz = SkeletonVisualizer(img_w, img_h)
    return viz.update(features, frame_idx)

def draw_skeleton(frame: np.ndarray, keypoints: dict, color: Tuple[int, int, int] = (0, 255, 0)):
    """Draw skeleton connections on frame."""
    flags = keypoints.get('flags', [1,1,1,1])
    lh_flag, rh_flag, arms_flag, face_flag = flags
    
    # Hands
    for name, pts, flag in [('lh', keypoints['lh'], lh_flag), ('rh', keypoints['rh'], rh_flag)]:
        if flag > 0.5 and len(pts) == 21:
            for a, b in HAND_CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    cv2.line(frame, tuple(pts[a]), tuple(pts[b]), color, 2)
            for pt in pts:
                cv2.circle(frame, tuple(pt), 3, color, -1)
    
    # Arms (shoulders, elbows) - indices: 0=L_shoulder, 1=R_shoulder, 2=L_elbow, 3=R_elbow
    arms = keypoints['arms']
    if arms_flag > 0.5 and len(arms) == 4:
        for a, b in ARM_CONNECTIONS:
            cv2.line(frame, tuple(arms[a]), tuple(arms[b]), color, 3)
        for pt in arms:
            cv2.circle(frame, tuple(pt), 5, color, -1)
        
        # Wrist-to-shoulder connections
        lh = keypoints['lh']
        rh = keypoints['rh']
        if lh_flag > 0.5 and len(lh) == 21:
            # Left wrist (index 0) to left shoulder (arms[0])
            cv2.line(frame, tuple(lh[0]), tuple(arms[0]), color, 2)
        if rh_flag > 0.5 and len(rh) == 21:
            # Right wrist (index 0) to right shoulder (arms[1])
            cv2.line(frame, tuple(rh[0]), tuple(arms[1]), color, 2)
    
    # Face
    face = keypoints['face']
    if face_flag > 0.5 and len(face) == 7:
        for a, b in FACE_CONNECTIONS:
            cv2.line(frame, tuple(face[a]), tuple(face[b]), color, 2)
        for pt in face:
            cv2.circle(frame, tuple(pt), 3, color, -1)

def visualize_sentence(
    sentence: np.ndarray,
    boundaries: List[int],
    words: List[str],
    fps: int = 25,
    window_name: str = "Synthetic Sentence"
):
    """Play synthetic sentence with skeleton overlay (stateful, retains coords)."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    delay = int(1000 / fps)
    T = len(sentence)
    boundary_set = set(boundaries)
    
    # Pre-compute word index for each frame
    word_for_frame = np.zeros(T, dtype=int)
    for j, b in enumerate(boundaries):
        if j + 1 < len(boundaries):
            word_for_frame[b:boundaries[j+1]] = j
        else:
            word_for_frame[b:] = j
    
    # Stateful visualizer - retains coords across frames
    viz = SkeletonVisualizer(1280, 720)
    
    print(f"[VIZ] Playing {T} frames at {fps} FPS. Controls: SPACE=next, R=restart, Q=quit")
    
    for i in range(T):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        try:
            kpts = viz.update(sentence, i)
            draw_skeleton(frame, kpts, (0, 255, 0))
        except Exception as e:
            cv2.putText(frame, f"Viz Error: {e}", (30, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        current_word = words[word_for_frame[i]]
        
        # Overlay info (lighter weight)
        cv2.putText(frame, f"Word: {current_word}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.putText(frame, f"Sentence: {' '.join(words)}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {i}/{T}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Highlight boundary frames
        if i in boundary_set:
            cv2.rectangle(frame, (0, 0), (1280, 720), (0, 0, 255), 3)
            cv2.putText(frame, "BOUNDARY", (500, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        cv2.imshow(window_name, frame)
        
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('r') or key == ord('R'):
            cv2.destroyWindow(window_name)
            return 'restart'
        elif key == 32:  # SPACE
            continue
    
    cv2.destroyWindow(window_name)
    return 'next'

# ────────────────────────── main demo ──────────────────────────
def demo_random_sentences(num_demos: int = 5):
    """Generate and visualize random synthetic sentences."""
    print(f"[DEMO] Generating {num_demos} random synthetic sentences...")
    
    for i in range(num_demos):
        try:
            sentence, boundaries, words = generate_random_sentence()
            print(f"\n  Demo {i+1}: {' '.join(words)}")
            print(f"    Frames: {len(sentence)}, Boundaries: {boundaries}")
            
            action = visualize_sentence(sentence, boundaries, words)
            if action == 'restart':
                continue
        except Exception as e:
            print(f"[ERROR] {e}")
            continue

def demo_specific_sentence(words: List[str]):
    """Visualize a specific sentence."""
    try:
        sentence, boundaries = generate_synthetic_sentence(words)
        print(f"Visualizing: {' '.join(words)}")
        print(f"Frames: {len(sentence)}, Boundaries: {boundaries}")
        visualize_sentence(sentence, boundaries, words)
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Demo specific words: python generator.py "hello" "world"
        words = sys.argv[1:]
        demo_specific_sentence(words)
    else:
        demo_random_sentences(10)