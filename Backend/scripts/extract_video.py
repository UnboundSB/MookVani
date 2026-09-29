import os
import cv2
import torch
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.extract_features import extract_frame_features_video_mode

BaseOptions = mp.tasks.BaseOptions
VisionMode = mp.tasks.vision.RunningMode.VIDEO

hand_options_video = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="../hand_landmarker.task"),
    running_mode=VisionMode, num_hands=2)
pose_options_video = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="../pose_landmarker.task"),
    running_mode=VisionMode)
face_options_video = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="../face_landmarker.task"),
    running_mode=VisionMode)

def extract_video_features(vid_path):
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print(f"Failed to open {vid_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0

    frames_data = []

    last_lh = np.zeros((21, 3))
    last_rh = np.zeros((21, 3))
    last_arms = np.zeros((4, 3))
    last_face = np.zeros((7, 3))

    with vision.HandLandmarker.create_from_options(hand_options_video) as hand_lm, \
         vision.PoseLandmarker.create_from_options(pose_options_video) as pose_lm, \
         vision.FaceLandmarker.create_from_options(face_options_video) as face_lm:

        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            f_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=f_rgb)
            timestamp_ms = int(frame_index * (1000 / fps))

            hands = hand_lm.detect_for_video(mp_image, timestamp_ms)
            pose = pose_lm.detect_for_video(mp_image, timestamp_ms)
            face = face_lm.detect_for_video(mp_image, timestamp_ms)

            # Reconstruct handedness mappings for video mode output
            hands_lh = np.zeros((21, 3))
            hands_rh = np.zeros((21, 3))
            lh_flag, rh_flag = False, False

            if hands and hands.hand_landmarks:
                for idx, handedness in enumerate(hands.handedness):
                    label = handedness[0].category_name
                    pts = np.array([[lm.x, lm.y, lm.z] for lm in hands.hand_landmarks[idx]])
                    if label == 'Left':
                        hands_lh = pts; lh_flag = True
                    else:
                        hands_rh = pts; rh_flag = True

            pose_lms = pose.pose_landmarks if pose and pose.pose_landmarks else None
            face_lms = face.face_landmarks if face and face.face_landmarks else None

            feat_vector, last_lh, last_rh, last_arms, last_face = extract_frame_features_video_mode(
                hands_lh, hands_rh, lh_flag, rh_flag,
                pose_lms, face_lms,
                last_lh, last_rh, last_arms, last_face
            )

            frames_data.append(feat_vector)
            frame_index += 1

    cap.release()
    if len(frames_data) == 0:
        return None

    tensor_data = torch.tensor(np.array(frames_data), dtype=torch.float32)  # (T, 163)
    return tensor_data
