import os
import cv2
import glob
import torch
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.extract_features import extract_frame_features

BaseOptions = mp.tasks.BaseOptions
VisionMode = mp.tasks.vision.RunningMode.IMAGE

hand_options = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path="../hand_landmarker.task"),
                                             running_mode=VisionMode, num_hands=2)
pose_options = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path="../pose_landmarker.task"),
                                             running_mode=VisionMode)
face_options = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path="../face_landmarker.task"),
                                             running_mode=VisionMode)

def extract_sentence_directory(source_dir, output_dir, label):
    os.makedirs(output_dir, exist_ok=True)
    all_videos = glob.glob(os.path.join(source_dir, "*.mp4"))

    print(f"⚙️ [{label}] Found {len(all_videos)} continuous videos. Extracting frames...")
    success_count = 0

    with vision.HandLandmarker.create_from_options(hand_options) as hand_lm, \
         vision.PoseLandmarker.create_from_options(pose_options) as pose_lm, \
         vision.FaceLandmarker.create_from_options(face_options) as face_lm:

        for vid_path in tqdm(all_videos, desc=label):
            vid_name = os.path.splitext(os.path.basename(vid_path))[0]
            save_path = os.path.join(output_dir, f"{vid_name}.pt")

            # SKIP if already exists to save massive time
            if os.path.exists(save_path):
                success_count += 1
                continue

            cap = cv2.VideoCapture(vid_path)
            if not cap.isOpened():
                continue

            frames_data = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # NO AGGRESSIVE SHARPENING RETRIES FOR VIDEO TO SAVE TIME
                f_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=f_rgb)

                hands = hand_lm.detect(mp_image)
                pose = pose_lm.detect(mp_image)
                face = face_lm.detect(mp_image)

                feats = extract_frame_features(hands, pose, face)
                frames_data.append(feats)

            cap.release()

            if len(frames_data) > 0:
                tensor_data = torch.tensor(np.array(frames_data), dtype=torch.float32) # (T, 163)
                torch.save(tensor_data, save_path)
                success_count += 1

    print(f"✅ [{label}] {success_count}/{len(all_videos)} videos processed.")
    return success_count

if __name__ == "__main__":
    RAW_SENTENCE_TRAIN_DIR = "../data/ISL_CSLRT_Corpus/Videos_Sentence_Level/Train"
    RAW_SENTENCE_TEST_DIR = "../data/ISL_CSLRT_Corpus/Videos_Sentence_Level/Test"

    OUTPUT_DIR_SENTENCE_TRAIN = "../data/tensors_sentence_level_163_train"
    OUTPUT_DIR_SENTENCE_TEST = "../data/tensors_sentence_level_163_test"

    train_success = extract_sentence_directory(RAW_SENTENCE_TRAIN_DIR, OUTPUT_DIR_SENTENCE_TRAIN, "SENTENCE TRAIN")
    test_success = extract_sentence_directory(RAW_SENTENCE_TEST_DIR, OUTPUT_DIR_SENTENCE_TEST, "SENTENCE TEST")

    print(f"\n✅ Sentence-level extraction complete. Train: {train_success}, Test: {test_success}")
