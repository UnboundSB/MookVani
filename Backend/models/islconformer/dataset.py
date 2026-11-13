"""
Dataset loader for F:\datasets\mookvani_dataset (ISL CSLRT Corpus)

Expected structure:
  Frames_Word_Level/<class>/<take>/*.jpg
  Videos_Sentence_Level/<sentence>/<take>.mp4
  corpus_csv_files/ISL_CSLRT_Corpus_word_details.xlsx
  corpus_csv_files/ISL_CSLRT_Corpus_frame_details.xlsx

Extracts MediaPipe landmarks → 163-dim tensors.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import mediapipe as mp


# ──────────────────────────────────────────────────────────────────
# MediaPipe Feature Extraction (163-dim canonical layout)
# ──────────────────────────────────────────────────────────────────

LH_IDX = list(range(21))          # Left hand 21 landmarks
RH_IDX = list(range(21, 42))      # Right hand 21
ARMS_IDX = [11, 12, 13, 14]       # Shoulders, elbows (Pose)
FACE_IDX = [159, 386, 4, 234, 454, 61, 291]  # FaceMesh

def extract_frame_landmarks(hands_result, pose_result, face_result):
    """Extract 163-dim feature vector from MediaPipe results."""
    # 163 = 63*2 (hands) + 12 (arms) + 21 (face) + 4 (flags)
    feat = np.zeros(163, dtype=np.float32)
    lh_flag = rh_flag = arms_flag = face_flag = 0.0

    # Hands
    if hands_result and hands_result.hand_landmarks:
        for i, handedness in enumerate(hands_result.handedness):
            label = handedness[0].category_name
            lms = np.array([[lm.x, lm.y, lm.z] for lm in hands_result.hand_landmarks[i]])
            conf = getattr(handedness[0], 'score', 1.0)
            if label == "Left":
                feat[0:63] = lms.flatten()
                lh_flag = conf
            else:
                feat[63:126] = lms.flatten()
                rh_flag = conf

    # Arms (shoulders + elbows)
    if pose_result and pose_result.pose_landmarks:
        plm = pose_result.pose_landmarks[0]
        arms = []
        for idx in ARMS_IDX:
            lm = plm[idx]
            arms.extend([lm.x, lm.y, lm.z])
        feat[126:138] = arms
        arms_flag = 1.0

    # Face (7 points × 3 = 21)
    if face_result and face_result.face_landmarks:
        flm = face_result.face_landmarks[0]
        face = []
        for idx in FACE_IDX:
            lm = flm[idx]
            face.extend([lm.x, lm.y, lm.z])
        feat[138:159] = face
        face_flag = 1.0

    # Flags
    feat[159] = lh_flag
    feat[160] = rh_flag
    feat[161] = arms_flag
    feat[162] = face_flag

    return feat


def extract_video_landmarks(video_path, max_frames=300):
    """Extract 163-dim sequence from video file."""
    BaseOptions = mp.tasks.BaseOptions
    Mode = mp.tasks.vision.RunningMode.VIDEO

    hand_opt = mp.tasks.vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=r"D:\mookvani\hand_landmarker.task"),
        running_mode=Mode, num_hands=2)
    pose_opt = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=r"D:\mookvani\pose_landmarker.task"),
        running_mode=Mode)
    face_opt = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=r"D:\mookvani\face_landmarker.task"),
        running_mode=Mode)

    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    ts_ms = 0

    with mp.tasks.vision.HandLandmarker.create_from_options(hand_opt) as hand_lm, \
         mp.tasks.vision.PoseLandmarker.create_from_options(pose_opt) as pose_lm, \
         mp.tasks.vision.FaceLandmarker.create_from_options(face_opt) as face_lm:

        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            hands = hand_lm.detect_for_video(mp_img, ts_ms)
            pose = pose_lm.detect_for_video(mp_img, ts_ms)
            face = face_lm.detect_for_video(mp_img, ts_ms)
            feat = extract_frame_landmarks(hands, pose, face)
            frames.append(feat)
            ts_ms += int(1000 / fps)

    cap.release()
    return np.array(frames, dtype=np.float32) if frames else None


# ──────────────────────────────────────────────────────────────────
# Word-Level Dataset (Frames_Word_Level)
class WordFramesDataset(Dataset):
    """
    Loads word-level frames, extracts MediaPipe landmarks on-the-fly.
    For pre-extracted tensors, use WordTensorDataset below.
    """
    def __init__(self, root, classes, max_len=120, transform=None):
        self.max_frames = max_len
        self.root = Path(root)
        self.classes = classes
        # Map class names to folder names (lowercase spaces -> UPPERCASE_UNDERSCORES)
        self.class_to_folder = {c: c.upper().replace(' ', '_') for c in classes}
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform

        self.samples = []
        for cls in classes:
            folder = self.class_to_folder[cls]
            cls_dir = self.root / folder
            if not cls_dir.exists():
                continue

            # Check for nested take folders first
            take_dirs = [d for d in cls_dir.iterdir() if d.is_dir()]
            if take_dirs:
                # Nested structure: class/take/frames.jpg
                for take_dir in take_dirs:
                    frames = sorted(take_dir.glob("*.jpg")) + sorted(take_dir.glob("*.png"))
                    if frames:
                        self.samples.append((frames, self.class_to_idx[cls]))
            else:
                # Flat structure: class/frames.jpg
                frames = sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.png"))
                if frames:
                    self.samples.append((frames, self.class_to_idx[cls]))

        print(f"[WordFramesDataset] {len(self.samples)} clips from {len(classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frames, label = self.samples[idx]
        # Limit frames
        if len(frames) > self.max_frames:
            step = len(frames) / self.max_frames
            frames = [frames[int(i * step)] for i in range(self.max_frames)]

        # Extract MediaPipe landmarks
        seq = []
        BaseOptions = mp.tasks.BaseOptions
        Mode = mp.tasks.vision.RunningMode.IMAGE
        hand_opt = mp.tasks.vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=r"D:\mookvani\hand_landmarker.task"),
            running_mode=Mode, num_hands=2)
        pose_opt = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=r"D:\mookvani\pose_landmarker.task"),
            running_mode=Mode)
        face_opt = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=r"D:\mookvani\face_landmarker.task"),
            running_mode=Mode)

        with mp.tasks.vision.HandLandmarker.create_from_options(hand_opt) as hand_lm, \
             mp.tasks.vision.PoseLandmarker.create_from_options(pose_opt) as pose_lm, \
             mp.tasks.vision.FaceLandmarker.create_from_options(face_opt) as face_lm:

            import cv2
            for frame_path in frames:
                img = cv2.imread(str(frame_path))
                if img is None:
                    continue
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                hands = hand_lm.detect(mp_img)
                pose = pose_lm.detect(mp_img)
                face = face_lm.detect(mp_img)
                feat = extract_frame_landmarks(hands, pose, face)
                seq.append(feat)

        if not seq:
            return torch.zeros(self.max_frames, 163), label, 0

        seq = np.array(seq[:self.max_frames], dtype=np.float32)
        T = seq.shape[0]
        if T < self.max_frames:
            pad = np.zeros((self.max_frames - T, 163), dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=0)

        return torch.from_numpy(seq), label, T


# ──────────────────────────────────────────────────────────────────
# Pre-extracted Tensor Dataset (for faster training)
# ──────────────────────────────────────────────────────────────────

class WordTensorDataset(Dataset):
    """Loads pre-extracted 163-dim .pt tensors."""
    def __init__(self, root, classes, max_len=120):
        self.root = Path(root)
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.max_len = max_len
        self.samples = []

        for cls in classes:
            folder = cls.upper().replace(' ', '_')
            cls_dir = self.root / folder
            if not cls_dir.exists():
                continue
            for pt_file in cls_dir.glob("*.pt"):
                self.samples.append((pt_file, self.class_to_idx[cls]))

        print(f"[WordTensorDataset] {len(self.samples)} clips from {len(self.classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        clip = torch.load(path, weights_only=True).float()
        if clip.ndim == 3:
            clip = clip[0]  # (1, T, 163) -> (T, 163)
        T = min(clip.shape[0], self.max_len)
        clip = clip[:T]
        # Pad
        if T < self.max_len:
            pad = torch.zeros(self.max_len - T, 163)
            clip = torch.cat([clip, pad], dim=0)
        return clip, label, T


def collate_word_batch(batch):
    """Pad sequences to max length in batch."""
    clips, labels, lengths = zip(*batch)
    clips = torch.stack(clips)  # (B, T, 163)
    return clips, torch.tensor(labels), torch.tensor(lengths)


def build_word_dataloaders(data_root, classes, batch_size=32, max_len=120,
                           num_workers=4, val_split=0.2, use_tensors=True):
    """Build train/val DataLoaders."""
    DatasetClass = WordTensorDataset if use_tensors else WordFramesDataset
    dataset = DatasetClass(data_root, classes, max_len=max_len)

    # Stratified split
    from collections import defaultdict
    by_class = defaultdict(list)
    for i, (_, label) in enumerate(dataset.samples):
        by_class[label].append(i)

    train_idx, val_idx = [], []
    for indices in by_class.values():
        np.random.shuffle(indices)
        n_val = max(1, int(len(indices) * val_split))
        # Ensure at least 1 train sample per class
        if len(indices) - n_val == 0:
            n_val = max(0, len(indices) - 1)
        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])

    # If no validation samples, move some from train to val
    if len(val_idx) == 0 and len(train_idx) > 0:
        val_idx.append(train_idx.pop())

    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_word_batch, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_word_batch, num_workers=num_workers)

    print(f"Train: {len(train_set)} | Val: {len(val_set)}")
    return train_loader, val_loader


# ──────────────────────────────────────────────────────────────────
# Sentence-Level Dataset (Videos_Sentence_Level)
# ──────────────────────────────────────────────────────────────────

class SentenceVideoDataset(Dataset):
    """Loads sentence videos, extracts landmarks for CTC training."""
    def __init__(self, root, gloss_csv, classes, max_frames=300):
        import pandas as pd
        self.root = Path(root)
        self.max_frames = max_frames
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        df = pd.read_csv(gloss_csv)
        self.samples = []
        for _, row in df.iterrows():
            sent = row['Sentence']
            gloss = row['SIGN GLOSSES']
            # Map gloss words to class indices
            words = gloss.split()
            target = [self.class_to_idx.get(w, -1) for w in words]
            target = [t for t in target if t >= 0]
            if target:
                # Find video
                vid_dir = self.root / sent.replace(" ", "_").replace("?", "").replace(".", "")
                vids = list(vid_dir.glob("*.mp4"))
                if vids:
                    self.samples.append((vids[0], target))

        print(f"[SentenceVideoDataset] {len(self.samples)} sentences")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, target = self.samples[idx]
        seq = extract_video_landmarks(video_path, self.max_frames)
        if seq is None:
            return torch.zeros(self.max_frames, 163), torch.tensor(target), 0, len(target)
        T = seq.shape[0]
        return torch.from_numpy(seq).float(), torch.tensor(target), T, len(target)


def collate_sentence_batch(batch):
    clips, targets, lengths, target_lens = zip(*batch)
    clips = torch.stack(clips)
    targets = torch.cat(targets)
    return clips, torch.tensor(lengths), targets, torch.tensor(target_lens)


if __name__ == "__main__":
    # Quick test
    classes = ["hello", "good", "bad", "you", "i"]
    train_loader, val_loader = build_word_dataloaders(
        r"F:\datasets\mookvani_dataset\ISL_CSLRT_Corpus\Frames_Word_Level",
        classes, batch_size=4, use_tensors=False)
    for x, y, l in train_loader:
        print(f"Batch: {x.shape}, labels: {y}, lengths: {l}")
        break