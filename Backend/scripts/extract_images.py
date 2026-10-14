"""Extract MediaPipe landmark features from image sequence datasets (e.g. ISL_CSLRT Frames_Word_Level).

Part of the 3-file extraction suite:
1. extraction.py     - Canonical landmark normalization & 163-dim feature formulation
2. extract_video.py  - Video stream feature extraction
3. extract_images.py - Image frame sequence feature extraction
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
import argparse

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from extraction import extract_frame_features, clean_word

BaseOptions = mp.tasks.BaseOptions
VisionMode = mp.tasks.vision.RunningMode.IMAGE

# Resolve absolute paths to MediaPipe models at project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_hand_path = os.path.join(ROOT_DIR, "hand_landmarker.task")
_pose_path = os.path.join(ROOT_DIR, "pose_landmarker.task")
_face_path = os.path.join(ROOT_DIR, "face_landmarker.task")

hand_options_image = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=_hand_path),
    running_mode=VisionMode, num_hands=2)
pose_options_image = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=_pose_path),
    running_mode=VisionMode)
face_options_image = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=_face_path),
    running_mode=VisionMode)


def natural_sort_key(s):
    """Sort strings containing numbers in natural ascending numerical order."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def extract_sequence_from_images(image_paths, hand_lm, pose_lm, face_lm):
    """Processes a sorted list of image file paths into a (T, 163) tensor."""
    frames_data = []

    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        f_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=f_rgb)

        hands_res = hand_lm.detect(mp_image)
        pose_res = pose_lm.detect(mp_image)
        face_res = face_lm.detect(mp_image)

        feat = extract_frame_features(hands_res, pose_res, face_res)
        frames_data.append(feat)

    if not frames_data:
        return None

    return torch.tensor(np.array(frames_data), dtype=torch.float32)  # (T, 163)


def extract_images_dataset(source_dir, output_dir):
    """Traverses all class subdirectories in source_dir, extracts features, and saves tensors in output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    class_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    class_dirs.sort(key=natural_sort_key)

    print(f"[INFO] Found {len(class_dirs)} class folders in {source_dir}")
    print(f"[INFO] Destination directory: {output_dir}")

    success_count = 0
    skipped_count = 0

    with vision.HandLandmarker.create_from_options(hand_options_image) as hand_lm, \
         vision.PoseLandmarker.create_from_options(pose_options_image) as pose_lm, \
         vision.FaceLandmarker.create_from_options(face_options_image) as face_lm:

        for c_name in tqdm(class_dirs, desc="Extracting ISL Word Frames"):
            clean_cls = clean_word(c_name)
            if not clean_cls:
                continue

            cls_input_dir = os.path.join(source_dir, c_name)
            cls_output_dir = os.path.join(output_dir, clean_cls)
            os.makedirs(cls_output_dir, exist_ok=True)
            save_path = os.path.join(cls_output_dir, f"{clean_cls}_01.pt")

            # Skip if already extracted
            if os.path.exists(save_path):
                skipped_count += 1
                success_count += 1
                continue

            # Find and naturally sort all images
            imgs = [f for f in os.listdir(cls_input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            imgs.sort(key=natural_sort_key)

            if not imgs:
                continue

            img_full_paths = [os.path.join(cls_input_dir, f) for f in imgs]
            tensor_seq = extract_sequence_from_images(img_full_paths, hand_lm, pose_lm, face_lm)

            if tensor_seq is not None:
                torch.save(tensor_seq, save_path)
                success_count += 1

    print(f"[SUCCESS] Extracted {success_count}/{len(class_dirs)} classes (skipped {skipped_count} existing).")
    return success_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract MediaPipe features from image frame sequences")
    parser.add_argument(
        "--source",
        type=str,
        default=r"d:\MookVani\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Frames_Word_Level",
        help="Source directory containing word frame folders"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=r"d:\MookVani\Backend\data\tensors_isl_csltr_word_level",
        help="Output directory for saved tensors"
    )
    args = parser.parse_args()

    extract_images_dataset(args.source, args.output)
