"""
YAP Segmentor Fine-tune v4 — Self-Supervised/Semi-Supervised
===============================================================
Blank class (0) = non-signing/rejected | Boundary refinement | 30 epochs
"""

import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import csv
import json
import random
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from yap_conformer import YAPConfig, YAPpingConformer, ConformerConfig, hand_part

# ────────────────────────── paths ──────────────────────────
SEG_CSV = Path(r"D:\MookVani\backend\segmentor\data_builder\segmentation_clean.csv")
SENTENCES_CSV = Path(r"D:\MookVani\backend\segmentor\data_builder\sentences.csv")
WORD_INVENTORY = Path(r"D:\MookVani\backend\segmentor\data_builder\word_inventory.json")
CLASSES_JSON = Path(r"D:\MookVani\Backend\models\word_vocabulary\classes.json")

# Unlabeled data for semi-supervised
UNLABELED_DIR = Path(r"D:\MookVani\backend\data\ISL_CSLRT_Corpus\Videos_Sentence_Level")

# Output
CHECKPOINT_DIR = Path(r"D:\MookVani\Backend\models\yap_segmentor_v4")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Feature cache
FEATURES_CACHE_DIR = Path(r"D:\MookVani\backend\segmentor\features_cache")
FEATURES_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ────────────────────────── constants ──────────────────────────
FPS = 25
MAX_LEN = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
END_THRESHOLD = 0.45
MIN_SEGMENT_FRAMES = 5
print(f"[INFO] Device: {DEVICE}")

# ────────────────────────── MediaPipe feature extractor ──────────────────────────
class MediaPipeFeatureExtractor:
    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        model_path = Path(__file__).resolve().parent / "holistic_landmarker.task"
        if not model_path.exists():
            import urllib.request
            model_url = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task"
            print(f"[EXTRACT] Downloading MediaPipe holistic model...")
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
        self.landmarker = vision.HolisticLandmarker.create_from_options(options)
    
    def extract_video(self, video_path: Path) -> np.ndarray:
        import mediapipe as mp
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or FPS
        frame_interval_ms = int(1000 / fps)
        
        features_list = []
        if not hasattr(self, '_global_frame_idx'):
            self._global_frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = self._global_frame_idx * frame_interval_ms
            self._global_frame_idx += 1
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
            feat = self._extract_frame(result)
            features_list.append(feat)
        cap.release()
        return np.stack(features_list, axis=0).astype(np.float32)
    
    def _extract_frame(self, result):
        lh = np.zeros((21, 3)); rh = np.zeros((21, 3))
        lh_flag = 0.0; rh_flag = 0.0
        if result.left_hand_landmarks:
            lh = np.array([[lm.x, lm.y, lm.z] for lm in result.left_hand_landmarks])
            lh_flag = 1.0
        if result.right_hand_landmarks:
            rh = np.array([[lm.x, lm.y, lm.z] for lm in result.right_hand_landmarks])
            rh_flag = 1.0
        arms = np.zeros((4, 3)); arms_flag = 0.0
        if result.pose_landmarks:
            p = result.pose_landmarks
            arms = np.array([
                [p[11].x, p[11].y, p[11].z], [p[12].x, p[12].y, p[12].z],
                [p[13].x, p[13].y, p[13].z], [p[14].x, p[14].y, p[14].z],
            ])
            arms_flag = 1.0
        face_pts = np.zeros((7, 3)); face_flag = 0.0
        if result.face_landmarks:
            f = result.face_landmarks
            face_pts = np.array([
                [f[159].x, f[159].y, f[159].z], [f[386].x, f[386].y, f[386].z],
                [f[4].x, f[4].y, f[4].z],
                [f[234].x, f[234].y, f[234].z], [f[454].x, f[454].y, f[454].z],
                [f[61].x, f[61].y, f[61].z], [f[291].x, f[291].y, f[291].z],
            ])
            face_flag = 1.0
        anchor = np.array([0.5, 0.5, 0.0])
        if face_flag: anchor = face_pts[2]
        elif arms_flag: anchor = (arms[0] + arms[1]) / 2.0
        elif lh_flag: anchor = lh[0]
        elif rh_flag: anchor = rh[0]
        scale = 0.3
        if arms_flag:
            pose_scale = np.linalg.norm(arms[0] - arms[1])
            if pose_scale > 0.05: scale = pose_scale
        elif face_flag:
            face_scale = np.linalg.norm(face_pts[3] - face_pts[4]) * 3.0
            if face_scale > 0.05: scale = face_scale
        lh_out = (lh - anchor) / scale if lh_flag else np.ones((21, 3)) * np.array([-1.5 * scale, 3.0 * scale, 0.0])
        rh_out = (rh - anchor) / scale if rh_flag else np.ones((21, 3)) * np.array([1.5 * scale, 3.0 * scale, 0.0])
        arms_out = (arms - anchor) / scale if arms_flag else np.zeros((4, 3))
        face_out = (face_pts - anchor) / scale if face_flag else np.zeros((7, 3))
        return np.concatenate([lh_out.flatten(), rh_out.flatten(), arms_out.flatten(), face_out.flatten(),
                               [lh_flag, rh_flag, arms_flag, face_flag]]).astype(np.float32)
    
    def close(self):
        self.landmarker.close()


# ────────────────────────── Feature Cache ──────────────────────────
FEATURES_CACHE_DIR = Path(r"D:\MookVani\backend\segmentor\features_cache")
FEATURES_CACHE_DIR.mkdir(parents=True, exist_ok=True)

class FeatureCache:
    def __init__(self, cache_dir: Path = FEATURES_CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _cache_path(self, video_path: Path) -> Path:
        import hashlib
        key = hashlib.md5(str(video_path).encode()).hexdigest()
        return self.cache_dir / f"{key}.npz"
    
    def get(self, video_path: Path) -> Optional[Dict]:
        cache_path = self._cache_path(video_path)
        if cache_path.exists():
            data = np.load(cache_path, allow_pickle=True)
            return {
                'features': data['features'],
                'boundaries': data['boundaries'].tolist() if 'boundaries' in data else [],
                'sentence': str(data['sentence']) if 'sentence' in data else '',
                'sentence_id': str(data['sentence_id']) if 'sentence_id' in data else '',
            }
        return None
    
    def put(self, video_path: Path, features: np.ndarray, boundaries: List[Dict], sentence: str, sentence_id: str):
        cache_path = self._cache_path(video_path)
        np.savez_compressed(
            cache_path,
            features=features,
            boundaries=np.array(boundaries, dtype=object),
            sentence=sentence,
            sentence_id=sentence_id,
        )
    
    def has(self, video_path: Path) -> bool:
        return self._cache_path(video_path).exists()


# ────────────────────────── Augmentation Pipeline ──────────────────────────
def trim_frames(sentence: np.ndarray, boundaries: List[int], 
                trim_start: int = 5, trim_end: int = 5) -> Tuple[np.ndarray, List[int]]:
    T = len(sentence)
    if trim_start + trim_end >= T:
        return sentence, boundaries
    new_sentence = sentence[trim_start:T-trim_end]
    new_boundaries = [b - trim_start for b in boundaries if trim_start <= b < T - trim_end]
    return new_sentence, new_boundaries

def horizontal_flip(sentence: np.ndarray, p: float = 0.5) -> np.ndarray:
    if random.random() > p:
        return sentence
    flipped = sentence.copy()
    lh = flipped[:, 0:63].copy()
    rh = flipped[:, 63:126].copy()
    flipped[:, 0:63] = rh
    flipped[:, 63:126] = lh
    flags = flipped[:, 159:163].copy()
    flipped[:, 159] = flags[:, 1]
    flipped[:, 160] = flags[:, 0]
    arms = flipped[:, 126:138].reshape(-1, 4, 3)
    arms[:, [0,1]] = arms[:, [1,0]]
    arms[:, [2,3]] = arms[:, [3,2]]
    flipped[:, 126:138] = arms.reshape(-1, 12)
    flipped[:, 0:63:3] *= -1
    flipped[:, 63:126:3] *= -1
    flipped[:, 126:138:3] *= -1
    flipped[:, 138:159:3] *= -1
    return flipped

def horizontal_shift(sentence: np.ndarray, max_shift: float = 0.15, p: float = 0.5) -> np.ndarray:
    if random.random() > p:
        return sentence
    shifted = sentence.copy()
    shift = random.uniform(-max_shift, max_shift)
    shifted[:, 0:63:3] += shift
    shifted[:, 63:126:3] += shift
    shifted[:, 126:138:3] += shift
    shifted[:, 138:159:3] += shift
    return shifted

def vertical_shift(sentence: np.ndarray, max_shift: float = 0.1, p: float = 0.3) -> np.ndarray:
    if random.random() > p:
        return sentence
    shifted = sentence.copy()
    shift = random.uniform(-max_shift, max_shift)
    shifted[:, 1:63:3] += shift
    shifted[:, 64:126:3] += shift
    shifted[:, 127:138:3] += shift
    shifted[:, 139:159:3] += shift
    return shifted

def apply_augmentations(sentence: np.ndarray, boundaries: List[int],
                        trim_range: Tuple[int, int] = (5, 10),
                        flip_p: float = 0.5, hshift_p: float = 0.5, vshift_p: float = 0.3) -> Tuple[np.ndarray, List[int]]:
    trim_start = random.randint(*trim_range)
    trim_end = random.randint(*trim_range)
    sentence, boundaries = trim_frames(sentence, boundaries, trim_start, trim_end)
    sentence = horizontal_flip(sentence, flip_p)
    sentence = horizontal_shift(sentence, p=hshift_p)
    sentence = vertical_shift(sentence, p=vshift_p)
    return sentence, boundaries


# ────────────────────────── Boundary Post-Processing ──────────────────────────
def nms_boundaries(boundaries: np.ndarray, min_distance: int = 5) -> np.ndarray:
    """Non-maximum suppression on boundaries - keep strongest within window."""
    if len(boundaries) == 0:
        return boundaries
    sorted_idx = np.argsort(boundaries)
    kept = []
    for b in boundaries[sorted_idx]:
        if not kept or b - kept[-1] >= 5:
            kept.append(b)
    return np.array(kept)

def smooth_boundaries(boundaries: np.ndarray, window: int = 3) -> np.ndarray:
    """Smooth boundary positions by median filtering."""
    if len(boundaries) < 3:
        return boundaries
    from scipy.ndimage import median_filter
    return median_filter(boundaries, size=window)

def enforce_min_segment(boundaries: np.ndarray, T: int, min_frames: int = 5) -> np.ndarray:
    """Remove boundaries that create segments shorter than min_frames."""
    if len(boundaries) == 0:
        return boundaries
    all_bounds = np.concatenate([[0], boundaries, [T]])
    diffs = np.diff(all_bounds)
    keep = []
    for i, b in enumerate(boundaries):
        if diffs[i] >= 5 and diffs[i+1] >= 5:
            keep.append(b)
    return np.array(keep)


# ────────────────────────── Loss Functions ──────────────────────────
def focal_presence_loss(presence_logit: torch.Tensor, presence_target: torch.Tensor, 
                        alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    prob = torch.sigmoid(presence_logit)
    ce = F.binary_cross_entropy_with_logits(presence_logit, presence_target, reduction='none')
    pt = prob * presence_target + (1 - prob) * (1 - presence_target)
    focal_weight = alpha * presence_target + (1 - alpha) * (1 - presence_target)
    loss = focal_weight * (1 - pt) ** gamma * ce
    return loss.mean()

def end_bce_loss(end_logit: torch.Tensor, end_target: torch.Tensor) -> torch.Tensor:
    pos_weight = torch.tensor([10.0], device=end_logit.device)
    return F.binary_cross_entropy_with_logits(end_logit, end_target, pos_weight=pos_weight)

def blank_class_loss(presence_logit: torch.Tensor, presence_target: torch.Tensor) -> torch.Tensor:
    """Class 0 = blank (non-signing), Class 1 = signing."""
    # presence_target: 0 = blank/idle, 1 = signing
    # This is just BCE with class weights
    pos_weight = torch.tensor([2.0], device=presence_logit.device)  # signing less frequent
    return F.binary_cross_entropy_with_logits(presence_logit, presence_target, pos_weight=pos_weight)

def temporal_smooth_loss(end_logit: torch.Tensor, lambda_smooth: float = 0.1) -> torch.Tensor:
    """Penalize rapid changes in end probability - encourages smooth boundaries."""
    end_prob = torch.sigmoid(end_logit)
    diff = end_prob[:, 1:] - end_prob[:, :-1]
    return lambda_smooth * (diff ** 2).mean()

def boundary_nms_penalty(end_logit: torch.Tensor, min_distance: int = 5, penalty: float = 0.01) -> torch.Tensor:
    """Gentle penalty for boundaries that are too close - only activates when boundaries are too close."""
    end_prob = torch.sigmoid(end_logit)
    T = end_logit.shape[1]
    penalty_loss = 0.0
    
    # Find peaks in end_prob - compare each point with neighbors
    # Pad with zeros on both sides
    end_prob_padded = F.pad(end_prob, (1, 1), mode='constant', value=0)  # (B, T+2)
    
    # is_peak[i] = end_prob[i] > end_prob[i-1] AND end_prob[i] > end_prob[i+1]
    left = end_prob_padded[:, :-2]   # (B, T) - left neighbor
    center = end_prob_padded[:, 1:-1]  # (B, T) - center
    right = end_prob_padded[:, 2:]   # (B, T) - right neighbor
    
    is_peak = (center > left) & (center > right)
    
    # Only penalize if two peaks are closer than min_distance
    peak_indices = is_peak.nonzero(as_tuple=True)
    if len(peak_indices[0]) > 1:
        batch_indices, peak_positions = peak_indices
        for b in range(end_logit.shape[0]):
            batch_peaks = peak_positions[batch_indices == b]
            if len(batch_peaks) > 1:
                sorted_peaks = batch_peaks.sort().values
                distances = sorted_peaks[1:] - sorted_peaks[:-1]
                too_close = distances < min_distance
                if too_close.any():
                    # Penalize based on how close they are
                    penalty_loss += penalty * (min_distance - distances[too_close].float()).mean()
    
    return penalty_loss * 0.01  # Very small weight


# ────────────────────────── Total Loss ──────────────────────────
def compute_losses(model_out: Dict, targets: Dict, epoch: int = 0) -> Tuple[torch.Tensor, Dict]:
    presence_logit = model_out['presence_logit']      # (B, T)
    end_logit = model_out['end_logit']                # (B, T)
    
    presence_target = targets['presence']   # 1 = signing, 0 = blank/idle
    end_target = targets['end_target']      # 1 = boundary
    
    lengths = targets['lengths']
    B, T = presence_logit.shape
    mask = torch.arange(T, device=presence_logit.device).unsqueeze(0) < lengths.unsqueeze(1)
    
    # Presence loss (blank class: 0=idle, 1=signing)
    presence_loss = F.binary_cross_entropy_with_logits(
        presence_logit[mask], 
        presence_target[mask],
        pos_weight=torch.tensor([2.0], device=DEVICE)
    )
    
    # End boundary loss (weighted BCE)
    end_loss = F.binary_cross_entropy_with_logits(
        end_logit[mask], 
        end_target[mask],
        pos_weight=torch.tensor([10.0], device=DEVICE)
    )
    
    # Temporal smoothing (very gentle)
    smooth_weight = min(0.02 * (epoch / 10), 0.02)
    smooth_loss = temporal_smooth_loss(end_logit, lambda_smooth=smooth_weight)
    
    # Min segment penalty (very gentle)
    seg_penalty_weight = min(0.05 * (epoch / 20), 0.05)
    seg_penalty = boundary_nms_penalty(end_logit, min_distance=5, penalty=seg_penalty_weight)
    
    total_loss = presence_loss + end_loss + smooth_weight * smooth_loss + seg_penalty
    
    return total_loss, {
        'total': total_loss.item(),
        'presence': presence_loss.item(),
        'end': end_loss.item(),
        'smooth': smooth_loss.item() if isinstance(smooth_loss, torch.Tensor) else smooth_loss,
        'seg_penalty': seg_penalty.item() if isinstance(seg_penalty, torch.Tensor) else seg_penalty,
    }


# ────────────────────────── Semi-Supervised Pseudo-Labeling ──────────────────────────
@torch.no_grad()
def generate_pseudo_labels(model, unlabeled_videos: List[Path], 
                           extractor: MediaPipeFeatureExtractor,
                           presence_thresh: float = 0.9, end_thresh: float = 0.8) -> List[Dict]:
    """Generate pseudo-labels for unlabeled videos."""
    model.eval()
    pseudo_data = []
    
    for video_path in unlabeled_videos:
        try:
            features = extractor.extract_video(video_path)
            if len(features) == 0:
                continue
            
            features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(DEVICE)
            B, T, _ = features_tensor.shape
            
            pose = features_tensor[:, :, :126].reshape(1, T, 42, 3)
            conf = torch.ones(1, T, 42, device=DEVICE)
            valid = torch.ones(1, T, 42, dtype=torch.bool, device=DEVICE)
            lh_flag = features_tensor[:, :, 159] > 0.5
            rh_flag = features_tensor[:, :, 160] > 0.5
            valid[:, :, :21] = lh_flag.unsqueeze(-1).expand(-1, -1, 21)
            valid[:, :, 21:42] = rh_flag.unsqueeze(-1).expand(-1, -1, 21)
            
            t = torch.arange(T, device=DEVICE).float().unsqueeze(0) / FPS
            
            with torch.no_grad():
                yap_out = model.yap.forward_frames(
                    pose=features_tensor[:, :, :126].reshape(1, T, 42, 3),
                    conf=torch.ones(1, T, 42, device=DEVICE),
                    valid=torch.ones(1, T, 42, dtype=torch.bool, device=DEVICE),
                    t=torch.arange(T, device=DEVICE).float().unsqueeze(0) / FPS,
                    reset_mask=None,
                )
            
            presence = torch.sigmoid(yap_out['presence'][0]).cpu().numpy()
            end = torch.sigmoid(yap_out['end'][0]).cpu().numpy()
            
            # High confidence pseudo-labels
            pred_boundaries = np.where(end > 0.8)[0]
            presence_high = presence > 0.9
            
            if len(pred_boundaries) > 0 and presence_high.mean() > 0.7:
                pseudo_data.append({
                    'video_path': video_path,
                    'features': features,
                    'boundaries': [{'frame_index': int(b), 'timestamp': float(b/FPS)} for b in pred_boundaries],
                    'sentence': f"pseudo_{video_path.stem}",
                    'sentence_id': f"pseudo_{video_path.stem}",
                    'confidence': float(end[pred_boundaries].mean()) if len(pred_boundaries) > 0 else 0.0
                })
        except Exception as e:
            print(f"[PSEUDO] Error on {video_path}: {e}")
    
    model.train()
    return pseudo_data


# ────────────────────────── Datasets ──────────────────────────
class LabeledDataset(Dataset):
    """Labeled real sentences with manual boundaries."""
    
    def __init__(self, max_len: int = 512, augment: bool = True):
        self.max_len = max_len
        self.augment = augment
        self.extractor = MediaPipeFeatureExtractor()
        self.cache = FeatureCache()
        
        self.sentence_groups = self._load_segmentation()
        self.sentence_ids = list(self.sentence_groups.keys())
        print(f"[INFO] Labeled: {len(self.sentence_ids)} sentences")
    
    def _load_segmentation(self) -> Dict[str, List[Dict]]:
        groups = {}
        with open(SEG_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sent_id = row['sentence_id']
                if sent_id not in groups:
                    groups[sent_id] = []
                groups[sent_id].append({
                    'frame_index': int(row['frame_index']),
                    'timestamp': float(row['timestamp']),
                    'sentence': row['sentence'],
                    'sentence_video': row['sentence_video'],
                })
        return groups
    
    def __len__(self):
        return len(self.sentence_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        sent_id = self.sentence_ids[idx]
        boundaries = self.sentence_groups[sent_id]
        video_path = Path(boundaries[0]['sentence_video'])
        
        cached = self.cache.get(video_path)
        if cached is not None:
            features = cached['features']
            boundaries = cached['boundaries']
            sentence = cached['sentence']
        else:
            features = self.extractor.extract_video(video_path)
            sentence = boundaries[0]['sentence']
            self.cache.put(video_path, features, boundaries, sentence, sent_id)
        
        T = len(features)
        presence = np.ones(T, dtype=np.float32)
        end_target = np.zeros(T, dtype=np.float32)
        
        boundary_frames = sorted([b['frame_index'] for b in boundaries if 0 <= b['frame_index'] < T])
        for bf in boundary_frames:
            end_target[bf] = 1.0
        
        transition_window = 2
        for bf in boundary_frames:
            start = max(0, bf - 2)
            end = min(T, bf + 3)
            presence[start:end] = 0.0
        
        if self.augment:
            features, boundary_frames = apply_augmentations(features, boundary_frames)
        
        T = len(features)
        # Recompute presence and end_target after augmentations
        presence = np.ones(T, dtype=np.float32)
        end_target = np.zeros(T, dtype=np.float32)
        boundary_frames = sorted([b for b in boundary_frames if 0 <= b < T])
        for bf in boundary_frames:
            end_target[bf] = 1.0
        transition_window = 2
        for bf in boundary_frames:
            start = max(0, bf - 2)
            end = min(T, bf + 3)
            presence[start:end] = 0.0
        
        # Pad/truncate to MAX_LEN
        if T > MAX_LEN:
            start = random.randint(0, T - MAX_LEN)
            features = features[start:start + MAX_LEN]
            presence = presence[start:start + MAX_LEN]
            end_target = end_target[start:start + MAX_LEN]
            T = MAX_LEN
        elif T < MAX_LEN:
            pad_len = MAX_LEN - T
            features = np.pad(features, ((0, pad_len), (0, 0)), mode='constant')
            presence = np.pad(presence, (0, pad_len), mode='constant')
            end_target = np.pad(end_target, (0, pad_len), mode='constant')
        
        return {
            'features': torch.from_numpy(features).float(),
            'presence': torch.from_numpy(presence).float(),
            'end_target': torch.from_numpy(end_target).float(),
            'lengths': torch.tensor(T, dtype=torch.long),
            'sentence_id': sent_id,
        }


class UnlabeledDataset(Dataset):
    """Unlabeled videos for semi-supervised pseudo-labeling."""
    
    def __init__(self, max_len: int = 512, max_videos: int = 100):
        self.max_len = max_len
        self.max_videos = max_videos
        self.extractor = MediaPipeFeatureExtractor()
        self.cache = FeatureCache()
        
        # Get all ISL_CSLRT videos
        self.video_paths = list(UNLABELED_DIR.rglob("*.mp4")) + list(UNLABELED_DIR.rglob("*.MP4"))
        self.video_paths = [p for p in self.video_paths if p.exists()]
        random.shuffle(self.video_paths)
        self.video_paths = self.video_paths[:max_videos]
        
        print(f"[INFO] Unlabeled: {len(self.video_paths)} videos")
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx: int) -> Dict:
        video_path = self.video_paths[idx]
        
        cached = self.cache.get(video_path)
        if cached is not None:
            features = cached['features']
        else:
            features = self.extractor.extract_video(video_path)
            self.cache.put(video_path, features, [], video_path.stem, video_path.stem)
        
        T = len(features)
        features = features[:MAX_LEN] if len(features) > MAX_LEN else np.pad(features, ((0, MAX_LEN - len(features)), (0, 0)))
        T = min(len(features), MAX_LEN)
        
        return {
            'features': torch.from_numpy(features).float(),
            'video_path': str(video_path),
            'lengths': torch.tensor(T, dtype=torch.long),
        }


# ────────────────────────── Training Loop ──────────────────────────
def train_v4(
    model: nn.Module,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epochs: int = 30,
    pseudo_label_every: int = 5,
    save_every: int = 5,
):
    model.train()
    model.to(DEVICE)
    
    print(f"[INFO] Starting v4 fine-tune for {epochs} epochs")
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Labeled batches/epoch: {len(labeled_loader)}")
    print(f"[INFO] Unlabeled batches/epoch: {len(unlabeled_loader)}")
    
    # Unlabeled iterator for semi-supervised
    unlabeled_iter = iter(unlabeled_loader)
    
    for epoch in range(1, epochs + 1):
        epoch_losses = []
        
        for batch_idx, batch in enumerate(labeled_loader):
            # Get unlabeled batch for consistency loss
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)
            
            features = batch['features'].to(DEVICE)
            presence = batch['presence'].to(DEVICE)
            end_target = batch['end_target'].to(DEVICE)
            lengths = batch['lengths'].to(DEVICE)
            
            # Labeled forward
            B, T, _ = features.shape
            pose = features[:, :, :126].reshape(B, T, 42, 3)
            conf = torch.ones(B, T, 42, device=DEVICE)
            valid = torch.ones(B, T, 42, dtype=torch.bool, device=DEVICE)
            
            lh_flag = features[:, :, 159] > 0.5
            rh_flag = features[:, :, 160] > 0.5
            valid = torch.ones(B, T, 42, dtype=torch.bool, device=DEVICE)
            valid[:, :, :21] = lh_flag.unsqueeze(-1).expand(-1, -1, 21)
            valid[:, :, 21:42] = rh_flag.unsqueeze(-1).expand(-1, -1, 21)
            
            t = torch.arange(T, device=DEVICE).float().unsqueeze(0).repeat(B, 1) / FPS
            lengths_tensor = torch.full((B,), T, dtype=torch.long, device=DEVICE)
            
            yap_out = model.yap.forward_frames(
                pose=features[:, :, :126].reshape(B, T, 42, 3),
                conf=conf, valid=valid, t=t, reset_mask=None,
            )
            
            presence_logit = yap_out['presence']
            end_logit = yap_out['end']
            
            presence_gt = batch['presence'].to(DEVICE)
            end_target_gt = batch['end_target'].to(DEVICE)
            
            model_out = {'presence_logit': presence_logit, 'end_logit': end_logit}
            targets_dict = {
                'presence': presence_gt,  # Already (B, T)
                'end_target': end_target_gt,
                'lengths': lengths,
            }
            
            loss, loss_dict = compute_losses(model_out, targets_dict, epoch)
            
            # Semi-supervised consistency loss on unlabeled
            if unlabeled_loader is not None:
                try:
                    unsup_batch = next(unlabeled_iter)
                except StopIteration:
                    unsup_iter = iter(unlabeled_loader)
                    unsup_batch = next(unsup_iter)
                
                unsup_features = unsup_batch['features'].to(DEVICE)
                unsup_B, unsup_T, _ = unsup_features.shape
                
                with torch.no_grad():
                    # Teacher predictions (EMA model would be better)
                    unsup_pose = unsup_features[:, :, :126].reshape(unsup_B, unsup_T, 42, 3)
                    unsup_conf = torch.ones(unsup_B, unsup_T, 42, device=DEVICE)
                    unsup_valid = torch.ones(unsup_B, unsup_T, 42, dtype=torch.bool, device=DEVICE)
                    unsup_t = torch.arange(unsup_T, device=DEVICE).float().unsqueeze(0).repeat(unsup_B, 1) / FPS
                    
                    with torch.no_grad():
                        unsup_out = model.yap.forward_frames(
                            pose=unsup_pose,
                            conf=unsup_conf,
                            valid=unsup_valid,
                            t=unsup_t,
                            reset_mask=None,
                        )
                    
                    pseudo_presence = torch.sigmoid(unsup_out['presence']).detach()
                    pseudo_end = torch.sigmoid(unsup_out['end']).detach()
                    
                    # Consistency loss: student should match teacher on unlabeled
                    student_out = model.yap.forward_frames(
                        pose=unsup_features[:, :, :126].reshape(unsup_B, unsup_T, 42, 3),
                        conf=torch.ones(unsup_B, unsup_T, 42, device=DEVICE),
                        valid=torch.ones(unsup_B, unsup_T, 42, dtype=torch.bool, device=DEVICE),
                        t=torch.arange(unsup_T, device=DEVICE).float().unsqueeze(0).repeat(unsup_B, 1) / FPS,
                        reset_mask=None,
                    )
                    
                    consistency_loss = F.mse_loss(
                        torch.sigmoid(student_out['presence']), 
                        pseudo_presence
                    ) + F.mse_loss(
                        torch.sigmoid(student_out['end']), 
                        pseudo_end
                    )
                    
                    loss = loss + 0.1 * consistency_loss
                loss = loss.requires_grad_()
                loss_dict['consistency'] = consistency_loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(loss_dict)
            
            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch} Batch {batch_idx}/{len(labeled_loader)}: "
                      f"loss={loss_dict['total']:.4f} "
                      f"(pres={loss_dict.get('presence', 0):.4f} "
                      f"end={loss_dict.get('end', 0):.4f} "
                      f"smooth={loss_dict.get('smooth', 0):.4f} "
                      f"seg_pen={loss_dict.get('seg_penalty', 0):.4f})")
        
        avg_loss = np.mean([l['total'] for l in epoch_losses])
        print(f"Epoch {epoch}/{epochs} - Avg Loss: {avg_loss:.4f}")
        
        if scheduler:
            scheduler.step()
        
        # Pseudo-labeling every N epochs
        if epoch % pseudo_label_every == 0:
            print(f"[INFO] Generating pseudo-labels at epoch {epoch}...")
            # Add pseudo-labeled data to training set (simplified)
            # In practice, you'd add to dataset and recreate loader
        
        if epoch % save_every == 0:
            ckpt_path = CHECKPOINT_DIR / f"yap_v4_epoch{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            print(f"[INFO] Saved checkpoint: {ckpt_path}")
    
    final_path = CHECKPOINT_DIR / "yap_v4_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"[INFO] Saved final model: {final_path}")


def main():
    print("=" * 60)
    print("YAP Segmentor v4 — Self-Supervised Fine-tune")
    print("=" * 60)
    
    # Datasets
    labeled_dataset = LabeledDataset(max_len=MAX_LEN, augment=True)
    unlabeled_dataset = UnlabeledDataset(max_len=MAX_LEN, max_videos=50)
    
    labeled_loader = DataLoader(labeled_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    
    # Model: YAP only (2 hands = 42 joints)
    cfg = YAPConfig(parts=[hand_part("left_hand"), hand_part("right_hand")])
    conformer_cfg = ConformerConfig(num_words=228)
    model = YAPpingConformer(cfg, conformer_cfg)
    
    # Load pretrained segmentor weights
    ckpt_path = Path(r"D:\MookVani\Backend\models\yap_segmentor\yap_segmentor_final.pth")
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE), strict=False)
        print(f"[INFO] Loaded pretrained segmentor weights")
    
    # Freeze Conformer blocks + head, train only YAP backbone
    for param in model.blocks.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = False
    for param in model.yap.parameters():
        param.requires_grad = True
    
    print(f"[INFO] Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=2e-5,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    train_v4(model, labeled_loader, unlabeled_loader, optimizer, scheduler, epochs=30)
    
    # Cleanup
    # (cleanup handled in dataset __del__ or manually)


if __name__ == "__main__":
    main()