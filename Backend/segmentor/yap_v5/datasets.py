"""
Datasets for YAP Segmentor
"""

import csv
import random
import hashlib
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from .config import cfg
from .extractor import MediaPipeFeatureExtractor
from .augmentation import augment_features


class FeatureCache:
    """Cache extracted features to avoid re-extraction."""
    
    def __init__(self, cache_dir: Path = Path(r"D:\MookVani\backend\segmentor\features_cache")):
        self.cache_dir = Path(r"D:\MookVani\backend\segmentor\features_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _cache_path(self, video_path: Path) -> Path:
        key = hashlib.md5(str(video_path).encode()).hexdigest()
        return self.cache_dir / f"{key}.npz"
    
    def get(self, video_path: Path) -> Optional[Dict]:
        cache_path = self._cache_path(video_path)
        if cache_path.exists():
            data = np.load(cache_path, allow_pickle=True)
            return {
                'features': data['features'],
                'boundaries': data['boundaries'].tolist() if 'boundaries' in data else [],
                'sentence': str(data['sentence']) if 'sentence' in data else '',
                'sentence_id': str(data['sentence_id']) if 'sentence_id' in data else '',
            }
        return None
    
    def put(self, video_path: Path, features: np.ndarray, boundaries: List[Dict], sentence: str, sentence_id: str):
        cache_path = self._cache_path(video_path)
        np.savez_compressed(
            cache_path,
            features=features,
            boundaries=np.array(boundaries, dtype=object),
            sentence=sentence,
            sentence_id=sentence_id,
        )
    
    def has(self, video_path: Path) -> bool:
        return self._cache_path(video_path).exists()


class LabeledDataset(Dataset):
    """Labeled real sentences with manual boundaries."""
    
    def __init__(self, max_len: int = 512, augment: bool = True, split: str = 'train', val_split: float = 0.2, seed: int = 42):
        self.max_len = 512
        self.augment = augment
        self.split = split
        self.val_split = val_split
        
        from .extractor import MediaPipeFeatureExtractor
        self.extractor = MediaPipeFeatureExtractor()
        self.cache = FeatureCache()
        
        self.sentence_groups = self._load_segmentation()
        self.sentence_ids = list(self.sentence_groups.keys())
        
        # Train/val split
        random.seed(42)
        random.shuffle(self.sentence_ids)
        split_idx = int(len(self.sentence_ids) * (1 - val_split))
        if split == 'train':
            self.sentence_ids = self.sentence_ids[:split_idx]
        else:
            self.sentence_ids = self.sentence_ids[split_idx:]
        
        print(f"[INFO] Labeled {split}: {len(self.sentence_ids)} sentences")
    
    def _load_segmentation(self) -> Dict[str, List[Dict]]:
        from .config import cfg
        groups = {}
        with open(cfg.SEG_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sent_id = row['sentence_id']
                if sent_id not in groups:
                    groups[sent_id] = []
                groups[sent_id].append({
                    'frame_index': int(row['frame_index']),
                    'timestamp': float(row['timestamp']),
                    'sentence': row['sentence'],
                    'sentence_video': row['sentence_video'],
                })
        return groups
    
    def __len__(self):
        return len(self.sentence_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        sent_id = self.sentence_ids[idx]
        boundaries = self.sentence_groups[sent_id]
        video_path = Path(boundaries[0]['sentence_video'])
        
        if not video_path.exists():
            raise RuntimeError(f"Video not found: {video_path}")
        
        # Try cache first
        cached = self.cache.get(video_path)
        if cached is not None:
            features = cached['features']
            boundaries = cached['boundaries']
            sentence = cached['sentence']
        else:
            from .extractor import MediaPipeFeatureExtractor
            extractor = MediaPipeFeatureExtractor()
            features = extractor.extract_video(video_path)
            sentence = boundaries[0]['sentence']
            self.cache.put(video_path, features, boundaries, sentence, sent_id)
        
        T = len(features)
        
        # Create ground truth
        presence = np.ones(T, dtype=np.float32)
        end_target = np.zeros(T, dtype=np.float32)
        
        boundary_frames = sorted([b['frame_index'] for b in boundaries if 0 <= b['frame_index'] < T])
        for bf in boundary_frames:
            end_target[bf] = 1.0
        
        # Presence = 0 at boundary frames and transition zones
        transition_window = 2
        for bf in boundary_frames:
            start = max(0, bf - 2)
            end = min(T, bf + 3)
            presence[start:end] = 0.0
        
        # Pad/truncate to max_len
        if T > 512:
            start = random.randint(0, T - 512)
            features = features[start:start + 512]
            presence = presence[start:start + 512]
            end_target = end_target[start:start + 512]
            T = 512
        elif T < 512:
            pad_len = 512 - T
            features = np.pad(features, ((0, pad_len), (0, 0)), mode='constant')
            presence = np.pad(presence, (0, pad_len), mode='constant')
            end_target = np.pad(end_target, (0, pad_len), mode='constant')
        
        # Augmentation
        if self.augment:
            features = augment_features(features)
        
        return {
            'features': torch.from_numpy(features).float(),
            'presence': torch.from_numpy(presence).float(),
            'end_target': torch.from_numpy(end_target).float(),
            'lengths': torch.tensor(T, dtype=torch.long),
            'sentence_id': sent_id,
        }


class UnlabeledDataset(Dataset):
    """Unlabeled videos for semi-supervised pseudo-labeling."""
    
    def __init__(self, max_len: int = 512, max_videos: int = 50):
        self.max_len = max_len
        self.max_videos = max_videos
        from .extractor import MediaPipeFeatureExtractor
        self.extractor = MediaPipeFeatureExtractor()
        self.cache = FeatureCache()
        
        from .config import cfg
        self.video_paths = list(cfg.UNLABELED_DIR.rglob("*.mp4")) + list(cfg.UNLABELED_DIR.rglob("*.MP4"))
        self.video_paths = [p for p in self.video_paths if p.exists()]
        random.shuffle(self.video_paths)
        self.video_paths = self.video_paths[:max_videos]
        
        print(f"[INFO] Unlabeled: {len(self.video_paths)} videos")
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx: int) -> Dict:
        video_path = self.video_paths[idx]
        
        cached = self.cache.get(video_path)
        if cached is not None:
            features = cached['features']
        else:
            features = self.extractor.extract_video(video_path)
            self.cache.put(video_path, features, [], video_path.stem, video_path.stem)
        
        T = len(features)
        features = features[:512] if T > 512 else np.pad(features, ((0, max(0, 512 - T)), (0, 0)), mode='constant')
        T = min(T, 512)
        
        return {
            'features': torch.from_numpy(features).float(),
            'video_path': str(video_path),
            'lengths': torch.tensor(T, dtype=torch.long),
        }


def create_dataloaders(
    labeled_batch_size: int = 8,
    unlabeled_batch_size: int = 8,
    val_split: float = 0.2,
    num_workers: int = 0,
    pin_memory: bool = True
):
    """Create data loaders for training."""
    from torch.utils.data import DataLoader
    
    # Full labeled dataset (will be split)
    full_labeled = LabeledDataset(max_len=512, augment=True, split='train', val_split=0.2)
    val_dataset = LabeledDataset(max_len=512, augment=False, split='val', val_split=0.2)
    unlabeled_dataset = UnlabeledDataset(max_len=512, max_videos=50)
    
    labeled_loader = DataLoader(
        LabeledDataset(max_len=512, augment=True, split='train', val_split=0.2),
        batch_size=8, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        LabeledDataset(max_len=512, augment=False, split='val', val_split=0.2),
        batch_size=4, shuffle=False, num_workers=0, pin_memory=True
    )
    unlabeled_loader = DataLoader(
        UnlabeledDataset(max_len=512, max_videos=50),
        batch_size=8, shuffle=True, num_workers=0, pin_memory=True
    )
    
    return labeled_loader, val_loader, unlabeled_loader