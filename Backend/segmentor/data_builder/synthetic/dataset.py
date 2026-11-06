"""
Synthetic Data Generator for Segmentor Training
Generates batches of (sequence, boundary_labels) for training.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random
import numpy as np
from pathlib import Path
from typing import List, Tuple, Iterator, Optional
from synthetic.generator import (
    generate_synthetic_sentence, 
    generate_random_sentence,
    load_classes,
    load_word_inventory,
    FEATURES_DIR,
    apply_augmentations
)

class SyntheticSegmentorDataset:
    """
    Dataset that generates synthetic sentences with perfect boundary labels.
    Used for training the conformer segmentor.
    """
    
    def __init__(
        self,
        max_seq_len: int = 512,
        sigma: float = 0.2,
        prep_frames: int = 5,
        use_archive_fallback: bool = True,
        vocab: Optional[List[str]] = None,
        # Augmentation params
        trim_range: Tuple[int, int] = (10, 15),
        flip_p: float = 0.5,
        hshift_p: float = 0.5,
        vshift_p: float = 0.3,
    ):
        self.max_seq_len = max_seq_len
        self.sigma = sigma
        self.prep_frames = prep_frames
        self.use_archive_fallback = use_archive_fallback
        
        # Augmentation params
        self.trim_range = trim_range
        self.flip_p = flip_p
        self.hshift_p = hshift_p
        self.vshift_p = vshift_p
        
        self.classes = load_classes()
        self.inventory = load_word_inventory()
        self.vocab = vocab or self.classes
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        
        # Cache loaded features
        self._feature_cache = {}
        
    def _load_word_features(self, word: str) -> List[np.ndarray]:
        """Load features for a word with caching."""
        if word in self._feature_cache:
            return self._feature_cache[word]
        
        word_dir = FEATURES_DIR / word.replace(" ", "_")
        feats = []
        if word_dir.exists():
            for f in word_dir.glob("*.npy"):
                try:
                    arr = np.load(f)
                    if arr.ndim == 2 and arr.shape[1] == 163:
                        feats.append(arr)
                except Exception:
                    pass
        
        if not feats and self.use_archive_fallback:
            from synthetic.generator import extract_word_features_from_video
            feats = extract_word_features_from_video(word, self.inventory)
        
        self._feature_cache[word] = feats
        return feats
    
    def _generate_one(self, words: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Generate one synthetic sentence with boundary labels."""
        if words is None:
            # Random words from vocabulary that have features
            available = [w for w in self.classes if self._load_word_features(w)]
            num_words = random.randint(2, 5)
            words = random.sample(available, min(num_words, len(available)))
        
        # Filter to only words with features
        words = [w for w in words if self._load_word_features(w)]
        if len(words) < 2:
            # Fallback to random
            return self._generate_one()
        
        sentence, boundaries = generate_synthetic_sentence(
            words, 
            sigma=self.sigma,
            prep_frames=self.prep_frames,
            use_archive_fallback=self.use_archive_fallback,
            trim_range=self.trim_range,
            flip_p=self.flip_p,
            hshift_p=self.hshift_p,
            vshift_p=self.vshift_p
        )
        
        # Create boundary labels (1 at boundary frames, 0 elsewhere)
        T = len(sentence)
        boundary_labels = np.zeros(T, dtype=np.float32)
        for b in boundaries:
            if b < T:
                boundary_labels[b] = 1.0
        
        # Pad or truncate to max_seq_len
        if T > self.max_seq_len:
            # Truncate (keep first max_seq_len frames)
            sentence = sentence[:self.max_seq_len]
            boundary_labels = boundary_labels[:self.max_seq_len]
            # Adjust boundaries
            boundaries = [b for b in boundaries if b < self.max_seq_len]
        elif T < self.max_seq_len:
            # Pad with zeros
            pad_len = self.max_seq_len - T
            sentence = np.pad(sentence, ((0, pad_len), (0, 0)), mode='constant')
            boundary_labels = np.pad(boundary_labels, (0, pad_len), mode='constant')
        
        return sentence, boundary_labels, words
    
    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray, List[str]]]:
        """Infinite iterator for training."""
        while True:
            yield self._generate_one()
    
    def batch_generator(self, batch_size: int = 32) -> Iterator[Tuple[np.ndarray, np.ndarray, List[List[str]]]]:
        """Generate batches for training."""
        batch_x, batch_y, batch_words = [], [], []
        for x, y, words in self:
            batch_x.append(x)
            batch_y.append(y)
            batch_words.append(words)
            if len(batch_x) >= batch_size:
                yield np.stack(batch_x), np.stack(batch_y), batch_words
                batch_x, batch_y, batch_words = [], [], []

def preview_dataset():
    """Show what the dataset generates."""
    print("=" * 60)
    print("SYNTHETIC SEGMENTOR DATASET PREVIEW")
    print("=" * 60)
    
    ds = SyntheticSegmentorDataset(max_seq_len=256, sigma=0.2)
    
    print(f"\nVocabulary size: {len(ds.vocab)}")
    print(f"Words with features: {len([w for w in ds.classes if ds._load_word_features(w)])}")
    
    print("\n--- Single Sample ---")
    x, y, words = ds._generate_one()
    print(f"Words: {words}")
    print(f"Sequence shape: {x.shape}")  # (T, 163)
    print(f"Boundary labels shape: {y.shape}")  # (T,)
    print(f"Boundary frames: {np.where(y > 0)[0].tolist()}")
    print(f"Positive labels: {y.sum():.0f}")
    print(f"Feature range: [{x.min():.3f}, {x.max():.3f}], mean: {x.mean():.3f}")
    
    print("\n--- Batch ---")
    batch_x, batch_y, batch_words = next(ds.batch_generator(batch_size=4))
    print(f"Batch X shape: {batch_x.shape}")  # (B, T, 163)
    print(f"Batch Y shape: {batch_y.shape}")  # (B, T)
    for i, (w, b) in enumerate(zip(batch_words, np.where(batch_y > 0)[1])):
        print(f"  Sample {i}: {w} -> boundaries at {np.where(batch_y[i] > 0)[0].tolist()}")
    
    print("\n--- Feature breakdown (first sample) ---")
    print(f"  LH hands (0:63):     {x[0, :63].shape}")
    print(f"  RH hands (63:126):   {x[0, 63:126].shape}")
    print(f"  Arms (126:138):      {x[0, 126:138].shape}")
    print(f"  Face (138:159):      {x[0, 138:159].shape}")
    print(f"  Flags (159:163):     {x[0, 159:163].shape}")

if __name__ == "__main__":
    preview_dataset()