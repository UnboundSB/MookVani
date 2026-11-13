"""
Backend/scripts/stable_extraction.py

Stable 163-dim MediaPipe landmark extraction for the iSign word-level dataset.

Coordinate stability:
  Each landmark coordinate (x, y, z) is tracked independently. When a frame's
  coordinate reads exactly 0.0 (dropout / reset), the previous non-zero coordinate
  is retained instead. This prevents the "down then up at infinite speed" jitter
  caused by coordinates flickering between a real value and 0.

Layout (163 dims):
  [0:63]    lh  (21 landmarks x 3)
  [63:126]  rh  (21 landmarks x 3)
  [126:138] arms (4 landmarks x 3: L-shoulder, R-shoulder, L-elbow, R-elbow)
  [138:159] face (7 landmarks x 3: L-eye, R-eye, nose, L-cheek, R-cheek, L-mouth, R-mouth)
  [159:163] flags [lh_flag, rh_flag, arms_flag, face_flag]
"""

from __future__ import annotations

import numpy as np


class CoordTracker:
    """
    Tracks the last known good value for every coordinate independently.

    On each frame, any coordinate that is exactly 0.0 is replaced by the last
    retained non-zero value. Coordinates that have never been seen non-zero stay 0.
    """

    def __init__(self, n_coords: int, n_dims: int = 3):
        self.last = np.zeros((n_coords, n_dims), dtype=np.float32)
        self.seen = np.zeros((n_coords, n_dims), dtype=bool)

    def update(self, coords: np.ndarray) -> np.ndarray:
        """
        coords: (n_coords, 3) array for this frame. Zeros = missing / reset.
        Returns stable (n_coords, 3) with zeros replaced by last good value.
        """
        coords = np.asarray(coords, dtype=np.float32)
        mask = coords == 0.0
        out = coords.copy()
        out[mask] = self.last[mask]
        # update last for non-zero coords
        nz = ~mask
        self.last[nz] = coords[nz]
        self.seen[nz] = True
        # coords never seen stay 0 (out already 0 there)
        return out

    def reset(self):
        self.last[:] = 0.0
        self.seen[:] = False


def _normalize_skeleton(lh, rh, arms, face_pts, lh_flag, rh_flag, arms_flag, face_flag):
    """Core robust cascade normalization (same as extraction.py canonical)."""
    anchor = np.array([0.5, 0.5, 0.0])
    if face_flag:
        anchor = face_pts[2]  # Nose
    elif arms_flag:
        anchor = (arms[0] + arms[1]) / 2.0  # Neck
    elif lh_flag:
        anchor = lh[0]  # Left wrist
    elif rh_flag:
        anchor = rh[0]  # Right wrist

    scale = 0.3
    if arms_flag:
        pose_scale = np.linalg.norm(arms[0] - arms[1])
        if pose_scale > 0.05:
            scale = pose_scale
    elif face_flag:
        face_scale = np.linalg.norm(face_pts[3] - face_pts[4]) * 3.0
        if face_scale > 0.05:
            scale = face_scale

    lh_out = lh - anchor if lh_flag else np.ones((21, 3)) * np.array([-1.5 * scale, 3.0 * scale, 0.0])
    rh_out = rh - anchor if rh_flag else np.ones((21, 3)) * np.array([1.5 * scale, 3.0 * scale, 0.0])
    arms_out = arms - anchor if arms_flag else np.zeros((4, 3))
    face_out = face_pts - anchor if face_flag else np.zeros((7, 3))

    lh_out = lh_out / scale
    rh_out = rh_out / scale
    arms_out = arms_out / scale
    face_out = face_out / scale
    return lh_out, rh_out, arms_out, face_out


# MediaPipe landmark indices (matches extraction.py canonical)
HAND_IDX = slice(0, 21)          # 21 hand landmarks
POSE_ARMS_IDX = [11, 12, 13, 14] # L-shoulder, R-shoulder, L-elbow, R-elbow
FACE_IDX = [159, 386, 4, 234, 454, 61, 291]  # L-eye, R-eye, nose, L-cheek, R-cheek, L-mouth, R-mouth


def extract_stable_frame(hands_result, pose_result, face_result,
                         lh_tracker, rh_tracker, arms_tracker, face_tracker):
    """
    Extract one 163-dim frame with per-coordinate carry-forward stability.

    Returns (frame_data: np.ndarray (163,), trackers unchanged - callers keep them).
    """
    # --- Hands (21 x 3 each) ---
    lh_raw = np.zeros((21, 3), dtype=np.float32)
    rh_raw = np.zeros((21, 3), dtype=np.float32)
    lh_flag = 0.0
    rh_flag = 0.0
    if hands_result and hands_result.hand_landmarks:
        for idx, handedness in enumerate(hands_result.handedness):
            label = handedness[0].category_name
            pts = np.array([[lm.x, lm.y, lm.z] for lm in hands_result.hand_landmarks[idx]],
                           dtype=np.float32)
            if label == 'Left':
                lh_raw = pts
                lh_flag = 1.0
            else:
                rh_raw = pts
                rh_flag = 1.0

    lh = lh_tracker.update(lh_raw) if lh_flag else lh_tracker.last.copy()
    rh = rh_tracker.update(rh_raw) if rh_flag else rh_tracker.last.copy()

    # --- Arms (4 x 3) ---
    arms_raw = np.zeros((4, 3), dtype=np.float32)
    arms_flag = 0.0
    if pose_result and pose_result.pose_landmarks:
        p = pose_result.pose_landmarks[0]
        arms_raw = np.array([
            [p[11].x, p[11].y, p[11].z],
            [p[12].x, p[12].y, p[12].z],
            [p[13].x, p[13].y, p[13].z],
            [p[14].x, p[14].y, p[14].z],
        ], dtype=np.float32)
        arms_flag = 1.0
    arms = arms_tracker.update(arms_raw) if arms_flag else arms_tracker.last.copy()

    # --- Face (7 x 3) ---
    face_raw = np.zeros((7, 3), dtype=np.float32)
    face_flag = 0.0
    if face_result and face_result.face_landmarks:
        f = face_result.face_landmarks[0]
        face_raw = np.array([
            [f[159].x, f[159].y, f[159].z],
            [f[386].x, f[386].y, f[386].z],
            [f[4].x, f[4].y, f[4].z],
            [f[234].x, f[234].y, f[234].z],
            [f[454].x, f[454].y, f[454].z],
            [f[61].x, f[61].y, f[61].z],
            [f[291].x, f[291].y, f[291].z],
        ], dtype=np.float32)
        face_flag = 1.0
    face = face_tracker.update(face_raw) if face_flag else face_tracker.last.copy()

    # --- Normalization ---
    lh_out, rh_out, arms_out, face_out = _normalize_skeleton(
        lh, rh, arms, face, lh_flag, rh_flag, arms_flag, face_flag
    )

    frame_data = np.concatenate([
        lh_out.flatten(), rh_out.flatten(), arms_out.flatten(), face_out.flatten(),
        [lh_flag, rh_flag, arms_flag, face_flag]
    ]).astype(np.float32)  # (163,)

    return frame_data


def new_trackers():
    """Create fresh trackers for a new video (resets carry-forward state)."""
    return (CoordTracker(21, 3),   # lh
            CoordTracker(21, 3),   # rh
            CoordTracker(4, 3),    # arms
            CoordTracker(7, 3))    # face
