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

MIN_HAND_CONFIDENCE = 0.6

def detect_with_retries(frame, hand_lm, pose_lm, face_lm):
    """Tries increasingly aggressive sharpening ONLY when needed, and picks the result by
    detection CONFIDENCE (not just hand count)."""
    best_hands, best_pose, best_face = None, None, None
    best_score, best_num_hands, best_level = -1.0, 0, -1

    for level in range(4):
        f = frame.copy()
        if level == 1:
            f = cv2.addWeighted(f, 1.5, cv2.GaussianBlur(f, (0, 0), 3), -0.5, 0)
        elif level == 2:
            f = cv2.addWeighted(f, 2.0, cv2.GaussianBlur(f, (0, 0), 5), -1.0, 0)
        elif level == 3:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            f = cv2.filter2D(f, -1, kernel)

        f_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=f_rgb)

        hands = hand_lm.detect(mp_image)
        pose = pose_lm.detect(mp_image)
        face = face_lm.detect(mp_image)

        if hands and hands.handedness:
            confidences = [h[0].score for h in hands.handedness]
            score = sum(confidences) / len(confidences)
            num_hands = len(confidences)
        else:
            score, num_hands = 0.0, 0

        is_better = (
            best_level == -1 or
            (num_hands > best_num_hands and score >= MIN_HAND_CONFIDENCE) or
            (num_hands == best_num_hands and score > best_score)
        )
        if is_better:
            best_hands, best_pose, best_face = hands, pose, face
            best_score, best_num_hands, best_level = score, num_hands, level

        if best_score >= MIN_HAND_CONFIDENCE and best_num_hands == 2:
            break

    return best_hands, best_pose, best_face, best_level, best_score

def process_static_image(img_path, hand_lm, pose_lm, face_lm):
    frame = cv2.imread(img_path)
    if frame is None:
        return None, -1, 0.0

    hands, pose, face = None, None, None
    hands, pose, face, level, score = detect_with_retries(frame, hand_lm, pose_lm, face_lm)

    feats = extract_frame_features(hands, pose, face)  # (163,)
    tensor = torch.tensor(np.expand_dims(feats, axis=0), dtype=torch.float32)  # (1, 163)
    return tensor, level, score

def extract_directory(source_dir, output_dir, label):
    all_images = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG'):
        all_images.extend(glob.glob(f"{source_dir}/**/{ext}", recursive=True))

    print(f"⚙️ [{label}] Found {len(all_images)} images. Extracting...")
    success_count = 0
    level_counts = {0: 0, 1: 0, 2: 0, 3: 0, -1: 0}

    with vision.HandLandmarker.create_from_options(hand_options) as hand_lm, \
         vision.PoseLandmarker.create_from_options(pose_options) as pose_lm, \
         vision.FaceLandmarker.create_from_options(face_options) as face_lm:

        for img_path in tqdm(all_images, desc=label):
            class_name = os.path.basename(os.path.dirname(img_path))
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            save_dir = os.path.join(output_dir, class_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{img_name}.pt")

            tensor_data, level, score = process_static_image(img_path, hand_lm, pose_lm, face_lm)
            if tensor_data is not None:
                torch.save(tensor_data, save_path)
                success_count += 1
                level_counts[level] = level_counts.get(level, 0) + 1

    print(f"✅ [{label}] {success_count}/{len(all_images)} images processed.")
    print(f"   Enhancement level usage: raw={level_counts[0]}, mild_sharpen={level_counts[1]}, "
          f"aggressive_sharpen={level_counts[2]}, kernel_sharpen={level_counts[3]}, "
          f"no_hands_detected={level_counts[-1]}")
    if level_counts[3] > 0.15 * max(1, success_count):
        print(f"   ⚠️ Harshest sharpening level chosen on {level_counts[3]/max(1,success_count):.1%} "
              f"of images — worth spot-checking a sample of these against the source image to "
              f"confirm hand poses are genuinely correct, not hallucinated.")
    return success_count

if __name__ == "__main__":
    AUGMENTED_WORD_LEVEL_DIR = "d:/MookVani/Backend/data/word_level_images_augmented"
    VAL_HOLDOUT_DIR = "d:/MookVani/Backend/data/word_level_images_val_holdout"
    OUTPUT_DIR_WORD_TRAIN = "d:/MookVani/Backend/data/tensors_word_level_163_train"
    OUTPUT_DIR_WORD_VAL = "d:/MookVani/Backend/data/tensors_word_level_163_val"
    os.makedirs(OUTPUT_DIR_WORD_TRAIN, exist_ok=True)
    os.makedirs(OUTPUT_DIR_WORD_VAL, exist_ok=True)

    train_success = extract_directory(AUGMENTED_WORD_LEVEL_DIR, OUTPUT_DIR_WORD_TRAIN, "TRAIN (real+augmented)")
    val_success = extract_directory(VAL_HOLDOUT_DIR, OUTPUT_DIR_WORD_VAL, "VAL (real only)")

    print(f"\n✅ Word-level extraction complete. Train: {train_success}, Val: {val_success}")
