"""
Backend/scripts/visualize_tensor.py

Play a real video with the extracted (T,163) tensor overlaid as a stick figure.

Tensor layout (163 dims per frame):
  [0:63]    lh  (21 x 3)
  [63:126]  rh  (21 x 3)
  [126:138] arms (4 x 3: L-shoulder, R-shoulder, L-elbow, R-elbow)
  [138:159] face (7 x 3: L-eye, R-eye, nose, L-cheek, R-cheek, L-mouth, R-mouth)
  [159:163] flags [lh, rh, arms, face]

Usage:
    .\venv\Scripts\python.exe Backend\scripts\visualize_tensor.py ^
        --video F:\dataset\archive_extracted\...\1. loud\MVI_1234.MOV ^
        --tensor Backend\data\tensors_include50_word_level\loud\loud_01.pt ^
        --scale 400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Stick-figure connections (indices into the flattened 163-dim vector)
# Arms:  [0]=L-shoulder, [1]=R-shoulder, [2]=L-elbow, [3]=R-elbow  (offset 126)
# Face:  [0]=L-eye, [1]=R-eye, [2]=nose, [3]=L-cheek, [4]=R-cheek, [5]=L-mouth, [6]=R-mouth  (offset 138)
# Hands: 21 landmarks each (offset 0 and 63)

ARMS_OFFSET = 126
FACE_OFFSET = 138
LH_OFFSET = 0
RH_OFFSET = 63

# Limb connections: (start_idx, end_idx, color BGR)
ARM_CONNECTIONS = [
    (0, 1, (0, 255, 0)),   # L-shoulder -> R-shoulder (across top)
    (0, 2, (0, 255, 0)),   # L-shoulder -> L-elbow
    (1, 3, (0, 255, 0)),   # R-shoulder -> R-elbow
]

FACE_CONNECTIONS = [
    (0, 1, (255, 0, 0)),   # L-eye -> R-eye
    (0, 2, (255, 0, 0)),   # L-eye -> nose
    (1, 2, (255, 0, 0)),   # R-eye -> nose
    (3, 4, (255, 0, 0)),   # L-cheek -> R-cheek
    (3, 5, (255, 0, 0)),   # L-cheek -> L-mouth
    (4, 6, (255, 0, 0)),   # R-cheek -> R-mouth
    (5, 6, (255, 0, 0)),   # L-mouth -> R-mouth
    (2, 5, (255, 0, 0)),   # nose -> L-mouth
    (2, 6, (255, 0, 0)),   # nose -> R-mouth
]

# Hand connections (MediaPipe hand skeleton)
HAND_CONNECTIONS = [
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),   # wrist to each finger base
    (1, 2), (2, 3), (3, 4),      # thumb
    (5, 6), (6, 7), (7, 8),      # index
    (9, 10), (10, 11), (11, 12), # middle
    (13, 14), (14, 15), (15, 16),# ring
    (17, 18), (18, 19), (19, 20),# pinky
]
HAND_COLOR = (0, 0, 255)


def denorm_to_display(pts: np.ndarray, scale: float, cx: int, cy: int) -> np.ndarray:
    """
    Map normalized coords (relative to anchor, divided by scale) to display pixels.
    pts: (N, 3) normalized. Returns (N, 2) int pixels.

    MediaPipe normalized coords: origin top-left, y increases downward.
    After normalization (pts - anchor), positive y = below anchor.
    So display_y = cy + y * scale keeps positive y below center (correct).
    """
    xy = pts[:, :2].copy()
    xy[:, 0] = cx + xy[:, 0] * scale   # x: right is positive
    xy[:, 1] = cy + xy[:, 1] * scale   # y: down is positive (NO flip)
    return xy.astype(np.int32)


def draw_skeleton(frame, tensor_frame, scale, cx, cy):
    """Draw stick figure on frame from one 163-dim frame."""
    h, w = frame.shape[:2]
    if cx is None:
        cx = w // 2
    if cy is None:
        cy = h // 2

    flags = tensor_frame[159:163]
    lh_flag, rh_flag, arms_flag, face_flag = flags

    # --- Arms ---
    if arms_flag > 0.5:
        arms = tensor_frame[ARMS_OFFSET:ARMS_OFFSET + 12].reshape(4, 3)
        pts = denorm_to_display(arms, scale, cx, cy)
        for i, j, color in ARM_CONNECTIONS:
            cv2.line(frame, tuple(pts[i]), tuple(pts[j]), color, 3)
        for p in pts:
            cv2.circle(frame, tuple(p), 4, (0, 255, 0), -1)

    # --- Face ---
    if face_flag > 0.5:
        face = tensor_frame[FACE_OFFSET:FACE_OFFSET + 21].reshape(7, 3)
        pts = denorm_to_display(face, scale, cx, cy)
        for i, j, color in FACE_CONNECTIONS:
            cv2.line(frame, tuple(pts[i]), tuple(pts[j]), color, 2)
        for p in pts:
            cv2.circle(frame, tuple(p), 3, (255, 0, 0), -1)

    # --- Hands ---
    for hand_offset, flag in ((LH_OFFSET, lh_flag), (RH_OFFSET, rh_flag)):
        if flag > 0.5:
            hand = tensor_frame[hand_offset:hand_offset + 63].reshape(21, 3)
            pts = denorm_to_display(hand, scale, cx, cy)
            # Draw connections
            for i, j in HAND_CONNECTIONS:
                cv2.line(frame, tuple(pts[i]), tuple(pts[j]), HAND_COLOR, 2)
            # Draw joints
            for p in pts:
                cv2.circle(frame, tuple(p), 3, HAND_COLOR, -1)

    # Label
    labels = []
    if lh_flag > 0.5: labels.append("LH")
    if rh_flag > 0.5: labels.append("RH")
    if arms_flag > 0.5: labels.append("ARMS")
    if face_flag > 0.5: labels.append("FACE")
    cv2.putText(frame, f"Frame flags: {', '.join(labels)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Visualize extracted tensor as stick figure on video")
    parser.add_argument("--video", type=Path, required=True, help="Path to input video")
    parser.add_argument("--tensor", type=Path, required=True, help="Path to (T,163) .pt tensor")
    parser.add_argument("--scale", type=float, default=300, help="Display scale for normalized coords")
    parser.add_argument("--cx", type=int, default=None, help="Center X (default: video width/2)")
    parser.add_argument("--cy", type=int, default=None, help="Center Y (default: video height/2)")
    parser.add_argument("--fps-delay", type=int, default=30, help="Delay ms between frames")
    args = parser.parse_args()

    # Load tensor
    tensor = torch.load(args.tensor, weights_only=False)  # (T, 163)
    tensor = tensor.numpy()
    print(f"Tensor shape: {tensor.shape}  (T={tensor.shape[0]}, dims={tensor.shape[1]})")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"Cannot open video: {args.video}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video frames: {total_frames}")

    frame_idx = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx < tensor.shape[0]:
                t_frame = tensor[frame_idx]
                frame = draw_skeleton(frame, t_frame, args.scale, args.cx, args.cy)
                cv2.putText(frame, f"{frame_idx+1}/{tensor.shape[0]}", (10, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "Tensor ended", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("iSign Stick Figure", frame)
            key = cv2.waitKey(args.fps_delay) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # pause
                cv2.waitKey(0)

            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
