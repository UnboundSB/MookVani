import os
import cv2
import glob
import torch
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm
import urllib.request
from data.extraction import extract_frame_features, extract_frame_features_video_mode

def download_models():
    models = {
        "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        "pose_landmarker.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        "holistic_landmarker.task": "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
    }
    print("Downloading MediaPipe task models...")
    for name, url in models.items():
        if not os.path.exists(name):
            print(f"Fetching {name}...")
            urllib.request.urlretrieve(url, name)
    print("All models downloaded!")

def process_static_image(img_path, hand_lm, pose_lm, face_lm):
    frame = cv2.imread(img_path)
    if frame is None:
        return None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    hands = hand_lm.detect(mp_image)
    pose = pose_lm.detect(mp_image)
    face = face_lm.detect(mp_image)

    feats = extract_frame_features(hands, pose, face)  # (163,)
    return torch.tensor(np.expand_dims(feats, axis=0), dtype=torch.float32)  # (1, 163)

def extract_word_level(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    BaseOptions = mp.tasks.BaseOptions
    VisionMode = mp.tasks.vision.RunningMode.IMAGE

    hand_options = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
                                                 running_mode=VisionMode, num_hands=2)
    pose_options = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path="pose_landmarker.task"),
                                                 running_mode=VisionMode)
    face_options = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path="face_landmarker.task"),
                                                 running_mode=VisionMode)

    all_images = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG'):
        all_images.extend(glob.glob(f"{data_dir}/**/{ext}", recursive=True))

    print(f"Found {len(all_images)} images. Starting word-level extraction...")
    success_count = 0

    with vision.HandLandmarker.create_from_options(hand_options) as hand_lm, \
         vision.PoseLandmarker.create_from_options(pose_options) as pose_lm, \
         vision.FaceLandmarker.create_from_options(face_options) as face_lm:

        for img_path in tqdm(all_images):
            class_name = os.path.basename(os.path.dirname(img_path))
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            save_dir = os.path.join(output_dir, class_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{img_name}.pt")

            if os.path.exists(save_path):
                success_count += 1
                continue

            tensor_data = process_static_image(img_path, hand_lm, pose_lm, face_lm)
            if tensor_data is not None:
                torch.save(tensor_data, save_path)
                success_count += 1

    print(f"Word-level extraction complete! {success_count}/{len(all_images)} images processed.")

def process_sentence_repetition_folder(folder_path, hand_lm, pose_lm, face_lm):
    frame_paths = sorted(glob.glob(os.path.join(folder_path, "*.jpg")) +
                          glob.glob(os.path.join(folder_path, "*.png")))
    if not frame_paths:
        return None

    seq = []
    for fp in frame_paths:
        frame = cv2.imread(fp)
        if frame is None:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        hands = hand_lm.detect(mp_image)
        pose = pose_lm.detect(mp_image)
        face = face_lm.detect(mp_image)
        feats = extract_frame_features(hands, pose, face)
        seq.append(feats)

    if not seq:
        return None
    return torch.tensor(np.array(seq), dtype=torch.float32)

def extract_sentence_level(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    BaseOptions = mp.tasks.BaseOptions
    VisionMode = mp.tasks.vision.RunningMode.IMAGE

    hand_options = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
                                                 running_mode=VisionMode, num_hands=2)
    pose_options = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path="pose_landmarker.task"),
                                                 running_mode=VisionMode)
    face_options = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path="face_landmarker.task"),
                                                 running_mode=VisionMode)

    if os.path.exists(data_dir):
        phrase_dirs = [d for d in glob.glob(f"{data_dir}/*") if os.path.isdir(d)]
        rep_folders = []
        for phrase_dir in phrase_dirs:
            phrase_name = os.path.basename(phrase_dir)
            for rep_dir in glob.glob(f"{phrase_dir}/*"):
                if os.path.isdir(rep_dir):
                    rep_folders.append((phrase_name, rep_dir))

        print(f"Found {len(phrase_dirs)} sentence phrases, {len(rep_folders)} total repetition folders. Extracting...")
        success_count = 0

        with vision.HandLandmarker.create_from_options(hand_options) as hand_lm, \
             vision.PoseLandmarker.create_from_options(pose_options) as pose_lm, \
             vision.FaceLandmarker.create_from_options(face_options) as face_lm:

            for phrase_name, rep_dir in tqdm(rep_folders):
                rep_id = os.path.basename(rep_dir)
                save_dir = os.path.join(output_dir, phrase_name)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{rep_id}.pt")

                if os.path.exists(save_path):
                    success_count += 1
                    continue

                tensor_data = process_sentence_repetition_folder(rep_dir, hand_lm, pose_lm, face_lm)
                if tensor_data is not None:
                    torch.save(tensor_data, save_path)
                    success_count += 1

        print(f"Sentence-level extraction complete! {success_count}/{len(rep_folders)} repetitions processed.")
    else:
        print(f"{data_dir} not found.")

def process_video(video_path, landmarker):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sequence = []
    frame_index = 0

    last_lh = np.zeros((21, 3)); last_rh = np.zeros((21, 3))
    last_arms = np.zeros((4, 3)); last_face = np.zeros((7, 3))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int((frame_index * 1000) / fps)
        frame_index += 1

        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        lh_raw = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks]) \
                 if results.left_hand_landmarks else np.zeros((21, 3))
        rh_raw = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks]) \
                 if results.right_hand_landmarks else np.zeros((21, 3))
        lh_flag = 1.0 if results.left_hand_landmarks else 0.0
        rh_flag = 1.0 if results.right_hand_landmarks else 0.0

        pose_landmarks = results.pose_landmarks if results.pose_landmarks else None
        face_landmarks = results.face_landmarks if results.face_landmarks else None

        feats, last_lh, last_rh, last_arms, last_face = extract_frame_features_video_mode(
            lh_raw, rh_raw, lh_flag, rh_flag, pose_landmarks, face_landmarks,
            last_lh, last_rh, last_arms, last_face
        )
        sequence.append(feats)

    cap.release()
    if not sequence:
        return None
    return torch.tensor(np.array(sequence), dtype=torch.float32)

def extract_video_level(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    video_options = vision.HolisticLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="holistic_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO
    )

    print("Extracting video tensors...")
    video_files = glob.glob(f"{data_dir}/**/*.mp4", recursive=True)
    success_count = 0

    for vid_path in tqdm(video_files):
        relative_path = os.path.relpath(vid_path, data_dir)
        save_dir = os.path.join(output_dir, os.path.dirname(relative_path))
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, os.path.basename(vid_path).replace(".mp4", ".pt"))

        if os.path.exists(save_path):
            success_count += 1
            continue

        with vision.HolisticLandmarker.create_from_options(video_options) as landmarker:
            tensor_data = process_video(vid_path, landmarker)
            if tensor_data is not None:
                torch.save(tensor_data, save_path)
                success_count += 1

    print(f"Video extraction complete! {success_count}/{len(video_files)} videos processed.")

if __name__ == "__main__":
    download_models()
    
    # Update these paths to match your local dataset location
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(BASE_DIR, "data", "isl_csltr_dataset")
    WORD_LEVEL_DIR = os.path.join(DATASET_DIR, "ISL_CSLRT_Corpus", "ISL_CSLRT_Corpus", "Frames_Word_Level")
    SENTENCE_LEVEL_DIR = os.path.join(DATASET_DIR, "ISL_CSLRT_Corpus", "ISL_CSLRT_Corpus", "Frames_Sentence_Level")
    
    OUTPUT_WORD = os.path.join(BASE_DIR, "data", "tensors_word_level_163")
    OUTPUT_SENTENCE = os.path.join(BASE_DIR, "data", "tensors_sentence_level_163")
    OUTPUT_VIDEO = os.path.join(BASE_DIR, "data", "tensors_video_163")
    
    extract_word_level(WORD_LEVEL_DIR, OUTPUT_WORD)
    extract_sentence_level(SENTENCE_LEVEL_DIR, OUTPUT_SENTENCE)
    extract_video_level(DATASET_DIR, OUTPUT_VIDEO)
