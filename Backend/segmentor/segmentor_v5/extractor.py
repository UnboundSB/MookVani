"""
MediaPipe Holistic feature extractor for the YAP segmentor.

Extracts a 163-dim per-frame feature vector:
    left_hand  (21 x 3 = 63)
    right_hand (21 x 3 = 63)
    pose arms  (4  x 3 = 12)   [shoulders + elbows]
    face       (7  x 3 = 21)   [eyes, nose, cheeks, mouth corners]
    flags      (4)             [lh, rh, arms, face present]

Total = 63 + 63 + 12 + 21 + 4 = 163.

Coordinates are re-anchored (face anchor preferred) and scaled by pose/face
extent so the vector is roughly translation/scale invariant.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Optional


class MediaPipeFeatureExtractor:
    """Extract 163-dim features from video using MediaPipe Holistic."""

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        model_path = Path(__file__).resolve().parent.parent / "holistic_landmarker.task"
        if not model_path.exists():
            import urllib.request
            model_url = (
                "https://storage.googleapis.com/mediapipe-models/"
                "holistic_landmarker/holistic_landmarker/float16/1/"
                "holistic_landmarker.task"
            )
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
        self._global_frame_idx = 0

    def extract_video(self, video_path: Path) -> np.ndarray:
        import mediapipe as mp

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_interval_ms = int(1000 / fps)

        features_list = []
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
            features_list.append(self._extract_frame(result))

        cap.release()
        if not features_list:
            raise RuntimeError(f"No frames extracted from {video_path}")
        return np.stack(features_list, axis=0).astype(np.float32)

    # ------------------------------------------------------------------ frame
    def _extract_frame(self, result) -> np.ndarray:
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

        return np.concatenate([
            lh_out.flatten(), rh_out.flatten(), arms_out.flatten(), face_out.flatten(),
            [lh_flag, rh_flag, arms_flag, face_flag],
        ]).astype(np.float32)

    def close(self):
        self.landmarker.close()
