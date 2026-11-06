"""
Datasets for YAP segmentor v5.

- ``LabeledDataset``: real sentence videos with boundary annotations from the
  segmentation CSV. Produces presence (1 = signing, 0 = transition) and
  end_target (1 at boundary frames) per frame. Supports train/val split.
- ``UnlabeledDataset``: raw sentence-level videos without annotations, used for
  Mean-Teacher consistency loss.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import FeatureCache
from .config import CFG
from .extractor import MediaPipeFeatureExtractor


# ------------------------------------------------------------------ labeled
class LabeledDataset(Dataset):
    """Sentence videos with frame-level boundary annotations."""

    def __init__(self, max_len: int = 512, augment: bool = True,
                 split: str = "train", val_split: float = 0.2, seed: int = 42):
        self.max_len = max_len
        self.augment = augment
        self.split = split
        self.val_split = val_split
        self.extractor = MediaPipeFeatureExtractor()
        self.cache = FeatureCache()

        self.sentence_groups = self._load_segmentation()
        self.sentence_ids = list(self.sentence_groups.keys())

        # Deterministic train/val split
        rng = random.Random(seed)
        rng.shuffle(self.sentence_ids)
        split_idx = int(len(self.sentence_ids) * (1 - val_split))
        if split == "train":
            self.sentence_ids = self.sentence_ids[:split_idx]
        else:
            self.sentence_ids = self.sentence_ids[split_idx:]

        print(f"[INFO] Labeled {split}: {len(self.sentence_ids)} sentences")

    def _load_segmentation(self) -> Dict[str, List[Dict]]:
        groups: Dict[str, List[Dict]] = {}
        with open(CFG.SEG_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sent_id = row["sentence_id"]
                groups.setdefault(sent_id, []).append({
                    "frame_index": int(row["frame_index"]),
                    "timestamp": float(row["timestamp"]),
                    "sentence": row["sentence"],
                    "sentence_video": row["sentence_video"],
                })
        return groups

    def __len__(self) -> int:
        return len(self.sentence_ids)

    def __getitem__(self, idx: int) -> Dict:
        sent_id = self.sentence_ids[idx]
        boundaries = self.sentence_groups[sent_id]
        video_path = Path(boundaries[0]["sentence_video"])

        cached = self.cache.get(video_path)
        if cached is not None:
            features = cached["features"]
            boundaries = cached["boundaries"]
            sentence = cached["sentence"]
        else:
            features = self.extractor.extract_video(video_path)
            sentence = boundaries[0]["sentence"]
            self.cache.put(video_path, features, boundaries, sentence, sent_id)

        T = len(features)
        presence = np.ones(T, dtype=np.float32)
        end_target = np.zeros(T, dtype=np.float32)

        boundary_frames = sorted(
            b["frame_index"] for b in boundaries if 0 <= b["frame_index"] < T
        )
        for bf in boundary_frames:
            end_target[bf] = 1.0

        # Transition windows around boundaries are "idle" (presence = 0).
        transition_window = 2
        for bf in boundary_frames:
            start = max(0, bf - transition_window)
            end = min(T, bf + transition_window + 1)
            presence[start:end] = 0.0

        # Pad / truncate to max_len.
        if T > self.max_len:
            start = random.randint(0, T - self.max_len)
            features = features[start:start + self.max_len]
            presence = presence[start:start + self.max_len]
            end_target = end_target[start:start + self.max_len]
            T = self.max_len
        elif T < self.max_len:
            pad_len = self.max_len - T
            features = np.pad(features, ((0, pad_len), (0, 0)), mode="constant")
            presence = np.pad(presence, (0, pad_len), mode="constant")
            end_target = np.pad(end_target, (0, pad_len), mode="constant")

        # Augmentation.
        if self.augment:
            from .augment import augment_features
            if random.random() < 0.5:
                features = augment_features(features)

        return {
            "features": torch.from_numpy(features).float(),
            "presence": torch.from_numpy(presence).float(),
            "end_target": torch.from_numpy(end_target).float(),
            "lengths": torch.tensor(T, dtype=torch.long),
            "sentence_id": sent_id,
        }


# ------------------------------------------------------------------ unlabeled
class UnlabeledDataset(Dataset):
    """Raw sentence-level videos without boundary annotations (semi-supervised)."""

    def __init__(self, max_len: int = 512, max_videos: int = 50):
        self.max_len = max_len
        self.max_videos = max_videos
        self.extractor = MediaPipeFeatureExtractor()
        self.cache = FeatureCache()

        self.video_paths = list(
            Path(r"D:\MookVani\backend\data\ISL_CSLRT_Corpus\Videos_Sentence_Level").rglob("*.mp4")
        )
        self.video_paths = [p for p in self.video_paths if p.exists()]
        random.shuffle(self.video_paths)
        self.video_paths = self.video_paths[:max_videos]
        print(f"[INFO] Unlabeled: {len(self.video_paths)} videos")

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Dict:
        video_path = self.video_paths[idx]

        cached = self.cache.get(video_path)
        if cached is not None:
            features = cached["features"]
        else:
            features = self.extractor.extract_video(video_path)
            self.cache.put(video_path, features, [], video_path.stem, video_path.stem)

        T = len(features)
        features = (features[:512] if T > 512
                    else np.pad(features, ((0, max(0, 512 - T)), (0, 0)), mode="constant"))
        T = min(T, 512)

        return {
            "features": torch.from_numpy(features).float(),
            "video_path": str(video_path),
            "lengths": torch.tensor(T, dtype=torch.long),
        }
