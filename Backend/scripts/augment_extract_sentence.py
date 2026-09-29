import os
import cv2
import glob
import random
import torch
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm
import sys
from collections import defaultdict

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

def get_random_geometric_matrix(w, h, max_shift_frac=0.06, max_rotation_deg=8, allow_flip=True):
    """Returns a (flip_flag, rot_mat) to be applied consistently across all frames of a sequence."""
    do_flip = allow_flip and random.random() < 0.5
    angle = random.uniform(-max_rotation_deg, max_rotation_deg)
    shift_x = random.uniform(-max_shift_frac, max_shift_frac) * w
    shift_y = random.uniform(-max_shift_frac, max_shift_frac) * h

    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rot_mat[0, 2] += shift_x
    rot_mat[1, 2] += shift_y

    return do_flip, rot_mat

def apply_geometric_matrix(img, do_flip, rot_mat):
    h, w = img.shape[:2]
    out = img.copy()
    if do_flip:
        out = cv2.flip(out, 1)
    out = cv2.warpAffine(out, rot_mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return out

def extract_and_save_sequence(frame_paths, save_path, do_flip, rot_mat, hand_lm, pose_lm, face_lm):
    """Extracts MediaPipe landmarks from a sequence of images and saves the tensor."""
    if os.path.exists(save_path):
        return True # Skip if already processed

    frames_data = []
    
    for f_path in frame_paths:
        img = cv2.imread(f_path)
        if img is None:
            continue
            
        if rot_mat is not None: # Apply augmentation
            img = apply_geometric_matrix(img, do_flip, rot_mat)
            
        f_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=f_rgb)

        hands = hand_lm.detect(mp_image)
        pose = pose_lm.detect(mp_image)
        face = face_lm.detect(mp_image)

        feats = extract_frame_features(hands, pose, face)
        frames_data.append(feats)

    if len(frames_data) > 0:
        tensor_data = torch.tensor(np.array(frames_data), dtype=torch.float32)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(tensor_data, save_path)
        return True
    
    return False

def process_sentence_dataset(raw_dir, train_out_dir, val_out_dir, target_train_count=25, val_split_ratio=0.2):
    os.makedirs(train_out_dir, exist_ok=True)
    os.makedirs(val_out_dir, exist_ok=True)

    # 1. Map classes to physical sample folder paths
    class_dirs = [d for d in glob.glob(f"{raw_dir}/*") if os.path.isdir(d)]
    class_to_samples = defaultdict(list)
    
    for c_dir in class_dirs:
        cls_name = os.path.basename(c_dir)
        sample_dirs = [d for d in glob.glob(f"{c_dir}/*") if os.path.isdir(d)]
        for s_dir in sample_dirs:
            frames = sorted(glob.glob(f"{s_dir}/*.jpg"))
            if frames:
                class_to_samples[cls_name].append(s_dir) # STORE DIRECTORY PATH, not just frames yet
                
    print(f"Found {len(class_to_samples)} sentence classes.")

    # 2. Split into Train/Val by checking sample directories
    train_samples_map = defaultdict(list)
    val_samples_map = defaultdict(list)
    
    for cls, samples in class_to_samples.items():
        random.shuffle(samples)
        split_idx = max(1, int(len(samples) * (1 - val_split_ratio))) if len(samples) > 1 else 1
        train_samples_map[cls] = samples[:split_idx]
        val_samples_map[cls] = samples[split_idx:] if len(samples) > 1 else []

    # =========================================================================
    # LEAKAGE VERIFICATION: Explicitly assert that train and val sets are disjoint
    # =========================================================================
    for cls in class_to_samples.keys():
        train_set = set(train_samples_map[cls])
        val_set = set(val_samples_map[cls])
        assert train_set.isdisjoint(val_set), f"DATA LEAKAGE DETECTED in class {cls}!"
    print("✅ Leakage Verification Passed: Zero overlap between Train and Val sequences.")

    # Calculate exact number of augmentations needed per class to hit target
    class_aug_distribution = {}
    total_train_target = 0
    
    for cls, train_dirs in train_samples_map.items():
        base_count = len(train_dirs)
        augs_needed = max(0, target_train_count - base_count)
        
        # Distribute the augs_needed evenly across the base_count videos
        augs_per_video = [0] * base_count
        for i in range(augs_needed):
            augs_per_video[i % base_count] += 1
            
        class_aug_distribution[cls] = augs_per_video
        total_train_target += base_count + augs_needed

    total_val = sum(len(s) for s in val_samples_map.values())
    print(f"Extraction Plan: {total_train_target} Train Tensors (Dynamic Target={target_train_count}), {total_val} Val Tensors.")

    # 3. Process and Extract
    with vision.HandLandmarker.create_from_options(hand_options) as hand_lm, \
         vision.PoseLandmarker.create_from_options(pose_options) as pose_lm, \
         vision.FaceLandmarker.create_from_options(face_options) as face_lm:
         
         # Process Train
         train_pbar = tqdm(total=total_train_target, desc="Extracting Train")
         for cls, train_dirs in train_samples_map.items():
             augs_per_video = class_aug_distribution[cls]
             
             for s_idx, s_dir in enumerate(train_dirs):
                 frames = sorted(glob.glob(f"{s_dir}/*.jpg"))
                 
                 # Base (unaugmented)
                 save_path = os.path.join(train_out_dir, cls, f"{os.path.basename(s_dir)}_base.pt")
                 extract_and_save_sequence(frames, save_path, False, None, hand_lm, pose_lm, face_lm)
                 train_pbar.update(1)
                 
                 # Augmented copies dynamically generated to hit target
                 num_augs_for_this_video = augs_per_video[s_idx]
                 if len(frames) > 0 and num_augs_for_this_video > 0:
                     test_img = cv2.imread(frames[0])
                     h, w = test_img.shape[:2]
                     
                     for aug_idx in range(num_augs_for_this_video):
                         save_path = os.path.join(train_out_dir, cls, f"{os.path.basename(s_dir)}_aug{aug_idx}.pt")
                         do_flip, rot_mat = get_random_geometric_matrix(w, h, allow_flip=True)
                         extract_and_save_sequence(frames, save_path, do_flip, rot_mat, hand_lm, pose_lm, face_lm)
                         train_pbar.update(1)
         train_pbar.close()
         
         # Process Val
         val_pbar = tqdm(total=total_val, desc="Extracting Val")
         for cls, val_dirs in val_samples_map.items():
             for s_dir in val_dirs:
                 frames = sorted(glob.glob(f"{s_dir}/*.jpg"))
                 
                 # Base only (NO augmentation for validation)
                 save_path = os.path.join(val_out_dir, cls, f"{os.path.basename(s_dir)}_base.pt")
                 extract_and_save_sequence(frames, save_path, False, None, hand_lm, pose_lm, face_lm)
                 val_pbar.update(1)
         val_pbar.close()

if __name__ == "__main__":
    RAW_SENTENCE_DIR = "d:/MookVani/Backend/data/isl_csltr_dataset/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/Frames_Sentence_Level"
    TRAIN_OUT_DIR = "d:/MookVani/Backend/data/tensors_sentence_level_163_train"
    VAL_OUT_DIR = "d:/MookVani/Backend/data/tensors_sentence_level_163_val"
    
    process_sentence_dataset(RAW_SENTENCE_DIR, TRAIN_OUT_DIR, VAL_OUT_DIR, target_train_count=25)
    print("✅ Dynamic Extraction and Verification complete!")
