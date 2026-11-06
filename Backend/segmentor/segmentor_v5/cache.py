"""
NPZ feature cache keyed by the MD5 hash of the video path.

Each cached entry stores: features, boundaries, sentence, sentence_id.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class FeatureCache:
    def __init__(self, cache_dir: Path | None = None):
        from .config import CFG
        self.cache_dir = Path(cache_dir) if cache_dir is not None else CFG.FEATURES_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ paths
    def _cache_path(self, video_path: Path) -> Path:
        key = hashlib.md5(str(video_path).encode()).hexdigest()
        return self.cache_dir / f"{key}.npz"

    # ------------------------------------------------------------------ api
    def get(self, video_path: Path) -> Optional[Dict]:
        cache_path = self._cache_path(video_path)
        if cache_path.exists():
            data = np.load(cache_path, allow_pickle=True)
            return {
                "features": data["features"],
                "boundaries": data["boundaries"].tolist() if "boundaries" in data else [],
                "sentence": str(data["sentence"]) if "sentence" in data else "",
                "sentence_id": str(data["sentence_id"]) if "sentence_id" in data else "",
            }
        return None

    def put(self, video_path: Path, features: np.ndarray, boundaries: List[Dict],
            sentence: str, sentence_id: str) -> None:
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
