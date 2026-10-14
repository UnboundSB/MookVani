"""Unified video feature extraction script for isolated word and continuous sentence videos.

Part of the 3-file extraction suite:
1. extraction.py     - Canonical landmark normalization & 163-dim feature formulation
2. extract_video.py  - Video stream feature extraction (word and sentence level)
3. extract_images.py - Image frame sequence feature extraction (e.g. ISL_CSLRT Frames_Word_Level)
"""

import os
import re
import cv2
import glob
import torch
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from extraction import extract_frame_features_video_mode, clean_word

BaseOptions = mp.tasks.BaseOptions
VisionMode = mp.tasks.vision.RunningMode.VIDEO

# Resolve absolute paths to MediaPipe models at project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_hand_path = os.path.join(ROOT_DIR, "hand_landmarker.task")
_pose_path = os.path.join(ROOT_DIR, "pose_landmarker.task")
_face_path = os.path.join(ROOT_DIR, "face_landmarker.task")

hand_options_video = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=_hand_path),
    running_mode=VisionMode, num_hands=2)
pose_options_video = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=_pose_path),
    running_mode=VisionMode)
face_options_video = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=_face_path),
    running_mode=VisionMode)


def extract_video_features(vid_path):
    """Extract canonical 163-dim features from a video file into a (T, 163) PyTorch tensor."""
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print(f"[WARN] Failed to open {vid_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0

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

    return torch.tensor(np.array(frames_data), dtype=torch.float32)  # (T, 163)


def process_single_video_word(vid_path, output_dir):
    """Processes a single word video and saves tensor under output_dir/class_name/vid_name.pt."""
    word_dir_name = os.path.basename(os.path.dirname(vid_path))
    class_name = clean_word(word_dir_name)
    if not class_name:
        return False

    img_name = os.path.splitext(os.path.basename(vid_path))[0]
    save_dir = os.path.join(output_dir, class_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{img_name}.pt")

    if os.path.exists(save_path):
        return True

    tensor_data = extract_video_features(vid_path)
    if tensor_data is not None:
        torch.save(tensor_data, save_path)
        return True
    return False


def extract_word_videos_directory(source_dir, output_dir, max_workers=None):
    """Extract features from isolated word video directories using multiprocessing."""
    os.makedirs(output_dir, exist_ok=True)
    all_videos = []
    for ext in ('*.mp4', '*.MP4', '*.mov', '*.MOV', '*.avi', '*.AVI'):
        all_videos.extend(glob.glob(os.path.join(source_dir, "**", ext), recursive=True))

    print(f"[INFO] Found {len(all_videos)} word videos in {source_dir}.")
    if not all_videos:
        return 0

    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() // 2)

    success_count = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_video_word, vid, output_dir): vid for vid in all_videos}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting word videos"):
            if future.result():
                success_count += 1

    print(f"[SUCCESS] Extracted {success_count}/{len(all_videos)} word videos.")
    return success_count


def extract_sentence_videos_directory(source_dir, output_dir, label="SENTENCE DATASET"):
    """Extract features from continuous sentence videos."""
    os.makedirs(output_dir, exist_ok=True)
    all_videos = []
    for ext in ('*.mp4', '*.avi', '*.mov'):
        all_videos.extend(glob.glob(os.path.join(source_dir, "**", ext), recursive=True))

    print(f"[INFO] [{label}] Found {len(all_videos)} continuous videos in {source_dir}.")
    if not all_videos:
        return 0

    success_count = 0
    for vid_path in tqdm(all_videos, desc=label):
        class_name = os.path.basename(os.path.dirname(vid_path))
        vid_name = os.path.splitext(os.path.basename(vid_path))[0]
        save_dir = os.path.join(output_dir, class_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{vid_name}.pt")

        if os.path.exists(save_path):
            success_count += 1
            continue

        tensor_data = extract_video_features(vid_path)
        if tensor_data is not None:
            torch.save(tensor_data, save_path)
            success_count += 1

    print(f"[SUCCESS] [{label}] {success_count}/{len(all_videos)} sentence videos processed.")
    return success_count


if __name__ == "__main__":
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="Extract MediaPipe features from videos")
    parser.add_argument("--source", type=str, default="", help="Input directory of video files")
    parser.add_argument("--output", type=str, default="", help="Output directory for saved .pt tensors")
    parser.add_argument("--mode", type=str, choices=["word", "sentence"], default="word")
    args = parser.parse_args()

    if args.source and args.output:
        if args.mode == "word":
            extract_word_videos_directory(args.source, args.output)
        else:
            extract_sentence_videos_directory(args.source, args.output)
    else:
        print("[INFO] Unified extract_video.py ready. Run with --source and --output, or import as module.")
