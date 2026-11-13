"""
Backend/scripts/visualize_webcam.py

Real-time stick figure visualization on your webcam feed.
Shows exactly what the extraction pipeline captures - hands, arms, face - so you
can verify it looks right before re-extracting the dataset.

Controls:
  q  - quit
  s  - save current frame as image
  c  - toggle confidence text on/off
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MODEL_DIR = Path(r"D:\mookvani")

# Connections (same as visualize_tensor.py)
ARM_CONNECTIONS = [
    (0, 1, (0, 255, 0)),   # L-shoulder -> R-shoulder
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

HAND_CONNECTIONS = [
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
]

FACE_IDX = [159, 386, 4, 234, 454, 61, 291]   # L-eye, R-eye, nose, L-cheek, R-cheek, L-mouth, R-mouth
POSE_ARMS_IDX = [11, 12, 13, 14]             # L-sh, R-sh, L-el, R-el


def main():
    BaseOptions = mp.tasks.BaseOptions
    VM = mp.tasks.vision.RunningMode.VIDEO

    hand_lm = mp.tasks.vision.HandLandmarker.create_from_options(
        mp.tasks.vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_DIR / "hand_landmarker.task")),
            running_mode=VM, num_hands=2))
    pose_lm = mp.tasks.vision.PoseLandmarker.create_from_options(
        mp.tasks.vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_DIR / "pose_landmarker.task")),
            running_mode=VM))
    face_lm = mp.tasks.vision.FaceLandmarker.create_from_options(
        mp.tasks.vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_DIR / "face_landmarker.task")),
            running_mode=VM,
            min_face_detection_confidence=0.1,
            min_face_presence_confidence=0.1,
            min_tracking_confidence=0.1,
            output_face_blendshapes=False))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    show_conf = True
    frame_index = 0

    print("Webcam stick figure running. Press 'q' to quit, 's' to save frame, 'c' to toggle confidence text.")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            f_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=f_rgb)
            timestamp_ms = frame_index

            hands = hand_lm.detect_for_video(mp_image, timestamp_ms)
            pose = pose_lm.detect_for_video(mp_image, timestamp_ms)
            face = face_lm.detect_for_video(mp_image, timestamp_ms)

            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            scale = min(w, h) * 0.45

            # --- Draw arms (shoulders + elbows) ---
            arms_flag = False
            if pose and pose.pose_landmarks:
                p = pose.pose_landmarks[0]
                arms_pts = []
                for idx in POSE_ARMS_IDX:
                    lm = p[idx]
                    arms_pts.append([lm.x, lm.y, lm.z])
                arms_pts = np.array(arms_pts, dtype=np.float32)
                arms_flag = True
                xy = arms_pts[:, :2] * np.array([w, h]) + np.array([cx, cy])
                xy = xy.astype(np.int32)
                for i, j, color in ARM_CONNECTIONS:
                    cv2.line(frame, tuple(xy[i]), tuple(xy[j]), color, 3)
                for pt in xy:
                    cv2.circle(frame, tuple(pt), 5, (0, 255, 0), -1)

            # --- Draw face (7 pts) ---
            face_flag = False
            if face and face.face_landmarks:
                f = face.face_landmarks[0]
                face_pts = np.array([[f[i].x, f[i].y, f[i].z] for i in FACE_IDX],
                                    dtype=np.float32)
                face_flag = True
                xy = face_pts[:, :2] * np.array([w, h]) + np.array([cx, cy])
                xy = xy.astype(np.int32)
                for i, j, color in FACE_CONNECTIONS:
                    cv2.line(frame, tuple(xy[i]), tuple(xy[j]), color, 2)
                for pt in xy:
                    cv2.circle(frame, tuple(pt), 4, (255, 0, 0), -1)

            # --- Draw hands ---
            for hand_color in [(0, 0, 255), (0, 128, 255)]:
                pass
            if hands and hands.hand_landmarks:
                for idx, handedness in enumerate(hands.handedness):
                    label = handedness[0].category_name
                    pts = np.array([[lm.x, lm.y, lm.z] for lm in hands.hand_landmarks[idx]],
                                   dtype=np.float32)
                    xy = pts[:, :2] * np.array([w, h]) + np.array([cx, cy])
                    xy = xy.astype(np.int32)
                    for i, j in HAND_CONNECTIONS:
                        cv2.line(frame, tuple(xy[i]), tuple(xy[j]), (0, 0, 255), 2)
                    for pt in xy:
                        cv2.circle(frame, tuple(pt), 3, (0, 0, 255), -1)
                    # Label hand
                    cv2.putText(frame, label, tuple(xy[0]), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 255), 2)

            # Confidence text
            if show_conf:
                info = []
                if face_flag: info.append("FACE: OK")
                else: info.append("FACE: none")
                if arms_flag: info.append("ARMS: OK")
                else: info.append("ARMS: none")
                if hands and hands.hand_landmarks: info.append("HANDS: OK")
                else: info.append("HANDS: none")
                cv2.putText(frame, " | ".join(info), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.putText(frame, f"Frame {frame_index}", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("iSign Stick Figure - Webcam", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite("webcam_stick_figure.png", frame)
                print("Saved webcam_stick_figure.png")
            elif key == ord('c'):
                show_conf = not show_conf

            frame_index += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
