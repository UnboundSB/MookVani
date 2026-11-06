"""
Movement-based video segmenter — Stage 0 preprocessing for YAP segmentor.

Runs MediaPipe every frame on raw sentence video, detects sign onset / offset
via wrist-velocity hysteresis, applies pre-roll / post-roll context sweep, and
extracts + caches each detected segment as a (T, 163) .pt tensor.

Output layout (generated, not committed):
    segmentor/data/<class_name>/<video_stem>/<seg_idx>/segment.pt
    segmentor/data/<class_name>/<video_stem>/manifest.json

Usage (from D:\MookVani\Backend):
    python segmentor\data_builder\segment_videos.py \
        --video Backend\data\ISL_CSLRT_Corpus\Videos_Sentence_Level\are you free today\are you free today.mp4 \
        --class-name hello \
        --out-dir segmentor\data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

BACKEND_DIR = Path(__file__).resolve().parent.parent
SEGMENTOR_DIR = Path(__file__).resolve().parent
for _p in (str(BACKEND_DIR), str(SEGMENTOR_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
FPS = 25
HI_VELOCITY = 0.012   # normalized wrist displacement per frame to trigger onset
LO_VELOCITY = 0.006   # lower threshold to confirm offset
DEBOUNCE_FRAMES = 3   # frames above HI to confirm onset / below LO to confirm offset
MIN_SEGMENT_FRAMES = 12
PRE_ROLL = 5          # frames before onset (anticipation / transition from idle)
POST_ROLL = 8         # frames after offset (hold release / transition to next sign)
BLANK_GAP_MIN = 10    # min frames between signs to count as a blank gap

# 163-dim layout (hands 126 + arms 12 + face 21 + flags 4)
WRIST_L_IDX = 7   # index 7 of left hand (wrist)
WRIST_R_IDX = 28  # index 28 of right hand (wrist)


# --------------------------------------------------------------------------- #
# MediaPipe per-frame extraction
# --------------------------------------------------------------------------- #
class FrameExtractor:
    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        model_path = SEGMENTOR_DIR / "holistic_landmarker.task"
        if not model_path.exists():
            import urllib.request
            url = ("https://storage.googleapis.com/mediapipe-models/"
                   "holistic_landmarker/holistic_landmarker/float16/1/"
                   "holistic_landmarker.task")
            print(f"[EXTRACT] Downloading MediaPipe holistic model...")
            urllib.request.urlretrieve(url, model_path)

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
        self._global_frame_idx = 0

    def extract_frame(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        import mediapipe as mp
        frame = cv2.resize(frame_bgr, (640, 480))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = self._global_frame_idx * int(1000 / FPS)
        self._global_frame_idx += 1
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        return self._encode(result)

    def _encode(self, result) -> np.ndarray:
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
            s = np.linalg.norm(arms[0] - arms[1])
            if s > 0.05: scale = s
        elif face_flag:
            s = np.linalg.norm(face_pts[3] - face_pts[4]) * 3.0
            if s > 0.05: scale = s

        lh_out = (lh - anchor) / scale if lh_flag else np.ones((21, 3)) * np.array([-1.5 * scale, 3.0 * scale, 0.0])
        rh_out = (rh - anchor) / scale if rh_flag else np.ones((21, 3)) * np.array([1.5 * scale, 3.0 * scale, 0.0])
        arms_out = (arms - anchor) / scale if arms_flag else np.zeros((4, 3))
        face_out = (face_pts - anchor) / scale if face_flag else np.zeros((7, 3))
        return np.concatenate([
            lh_out.flatten(), rh_out.flatten(), arms_out.flatten(), face_out.flatten(),
            [lh_flag, rh_flag, arms_flag, face_flag],
        ]).astype(np.float32)

    def close(self):
        self.landmarker.close()


def wrist_velocity(feat_t: np.ndarray, feat_tm1: np.ndarray) -> float:
    """Normalized wrist displacement between consecutive frames."""
    if feat_t is None or feat_tm1 is None:
        return 0.0
    wl_t = feat_t[WRIST_L_IDX * 3 : WRIST_L_IDX * 3 + 3]
    wr_t = feat_t[21 * 3 + WRIST_R_IDX * 3 : 21 * 3 + WRIST_R_IDX * 3 + 3]
    wl_p = feat_tm1[WRIST_L_IDX * 3 : WRIST_L_IDX * 3 + 3]
    wr_p = feat_tm1[21 * 3 + WRIST_R_IDX * 3 : 21 * 3 + WRIST_R_IDX * 3 + 3]
    return float(np.linalg.norm(wl_t - wl_p) + np.linalg.norm(wr_t - wr_p))


def detect_sign_segments(features: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (start_frame, end_frame) inclusive sign segments."""
    T = features.shape[0]
    if T < 2:
        return []

    vel = np.zeros(T, dtype=np.float32)
    for t in range(1, T):
        vel[t] = wrist_velocity(features[t], features[t - 1])

    # EMA smoothing to kill MediaPipe jitter
    smooth = np.zeros(T, dtype=np.float32)
    alpha = 0.7
    smooth[0] = vel[0]
    for t in range(1, T):
        smooth[t] = alpha * vel[t] + (1 - alpha) * smooth[t - 1]

    segments = []
    t = 0
    while t < T:
        # onset: HI crossed and sustained for DEBOUNCE_FRAMES
        if smooth[t] > HI_VELOCITY:
            onset = t
            sustained = 0
            while t < T and smooth[t] > HI_VELOCITY:
                sustained += 1
                t += 1
            if sustained < DEBOUNCE_FRAMES:
                continue
            # offset: drop below LO and stay there for DEBOUNCE_FRAMES
            offset = t - 1
            below = 0
            while t < T:
                if smooth[t] < LO_VELOCITY:
                    below += 1
                    if below >= DEBOUNCE_FRAMES:
                        offset = t - DEBOUNCE_FRAMES
                        break
                else:
                    below = 0
                t += 1
            else:
                offset = T - 1

            if offset - onset + 1 >= MIN_SEGMENT_FRAMES:
                segments.append((onset, offset))
        else:
            t += 1
    return segments


# --------------------------------------------------------------------------- #
# Context sweep + blank gaps
# --------------------------------------------------------------------------- #
def apply_context(segments: list[tuple[int, int]], T: int) -> list[tuple[int, int]]:
    """Add pre-roll / post-roll around each segment, clamp to [0, T-1], merge overlaps."""
    expanded = []
    for s, e in segments:
        ns = max(0, s - PRE_ROLL)
        ne = min(T - 1, e + POST_ROLL)
        expanded.append((ns, ne))
    # merge overlaps
    expanded.sort()
    merged = []
    for s, e in expanded:
        if merged and s <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def blank_gaps(segments: list[tuple[int, int]], T: int) -> list[tuple[int, int]]:
    """Idle gaps between signs long enough to count as blank class."""
    gaps = []
    prev_end = -1
    for s, e in segments:
        gap_start = prev_end + 1
        gap_end = s - 1
        if gap_end - gap_start + 1 >= BLANK_GAP_MIN:
            gaps.append((gap_start, gap_end))
        prev_end = e
    if prev_end + 1 <= T - 1 and (T - 1 - (prev_end + 1) + 1) >= BLANK_GAP_MIN:
        gaps.append((prev_end + 1, T - 1))
    return gaps


# --------------------------------------------------------------------------- #
# Tensor extraction + caching
# --------------------------------------------------------------------------- #
def extract_segment_tensor(features: np.ndarray, s: int, e: int) -> np.ndarray:
    """Slice features[s:e+1] -> (T, 163)."""
    return features[s:e + 1].copy()


def save_segment(out_root: Path, class_name: str, video_stem: str,
                 seg_idx: int, tensor: np.ndarray, is_blank: bool,
                 s: int, e: int) -> dict:
    seg_dir = out_root / class_name / video_stem / f"seg_{seg_idx:04d}"
    seg_dir.mkdir(parents=True, exist_ok=True)
    np.save(seg_dir / "segment.pt", tensor)
    meta = {
        "class": class_name,
        "video_stem": video_stem,
        "seg_idx": seg_idx,
        "is_blank": is_blank,
        "start_frame": s,
        "end_frame": e,
        "frames": int(tensor.shape[0]),
    }
    with open(seg_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def segment_video(video_path: str | Path, class_name: str,
                  out_root: str | Path = "segmentor/data",
                  cache_features: bool = True) -> dict:
    video_path = Path(video_path)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    extractor = FrameExtractor()
    all_features = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        feat = extractor.extract_frame(frame)
        if feat is not None:
            all_features.append(feat)
    cap.release()
    extractor.close()

    if not all_features:
        return {"video": str(video_path), "segments": [], "blanks": [], "error": "no frames"}

    features = np.stack(all_features, axis=0).astype(np.float32)
    T = features.shape[0]

    sign_segments = detect_sign_segments(features)
    sign_segments = apply_context(sign_segments, T)
    blank_segments = blank_gaps(sign_segments, T)

    video_stem = video_path.stem
    saved = []
    seg_idx = 0

    for s, e in sign_segments:
        tensor = extract_segment_tensor(features, s, e)
        meta = save_segment(out_root, class_name, video_stem, seg_idx,
                            tensor, is_blank=False, s=s, e=e)
        saved.append(meta)
        seg_idx += 1

    for s, e in blank_segments:
        tensor = extract_segment_tensor(features, s, e)
        meta = save_segment(out_root, class_name, video_stem, seg_idx,
                            tensor, is_blank=True, s=s, e=e)
        saved.append(meta)
        seg_idx += 1

    manifest = {
        "video": str(video_path),
        "video_stem": video_stem,
        "class": class_name,
        "total_frames": T,
        "fps": FPS,
        "num_segments": len(saved),
        "segments": saved,
    }
    manifest_path = out_root / class_name / video_stem / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[SEGMENT] {video_stem}: {len(sign_segments)} signs, "
          f"{len(blank_segments)} blanks, {len(saved)} tensors saved")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Movement-based video segmenter")
    parser.add_argument("--video", required=True, help="Path to raw .mp4 video")
    parser.add_argument("--class-name", required=True, help="Target class name (or 'blank')")
    parser.add_argument("--out-dir", default="segmentor/data",
                        help="Output root for segment tensors")
    args = parser.parse_args()

    segment_video(args.video, args.class_name, args.out_dir)


if __name__ == "__main__":
    main()

