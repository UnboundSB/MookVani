"""
Boundary post-processing helpers: NMS, median smoothing, min-segment enforcement.
"""

from __future__ import annotations

from typing import List

import numpy as np


def nms_boundaries(boundaries: List[int], min_distance: int = 5) -> List[int]:
    """Keep only boundaries separated by at least ``min_distance`` frames."""
    if not boundaries:
        return []
    boundaries = sorted(boundaries)
    kept = [boundaries[0]]
    for b in boundaries[1:]:
        if b - kept[-1] >= min_distance:
            kept.append(b)
    return kept


def smooth_boundaries(boundaries: np.ndarray, window: int = 3) -> np.ndarray:
    """Median-filter a boundary index array."""
    if len(boundaries) < 3:
        return boundaries
    from scipy.ndimage import median_filter
    return median_filter(boundaries.astype(float), size=window).astype(int)


def enforce_min_segment(boundaries: np.ndarray, T: int, min_frames: int = 5) -> np.ndarray:
    """Drop boundaries that would create segments shorter than ``min_frames``."""
    if len(boundaries) == 0:
        return boundaries
    all_bounds = np.concatenate([[0], boundaries, [T]])
    diffs = np.diff(all_bounds)
    keep = []
    for i, b in enumerate(boundaries):
        if diffs[i] >= min_frames and diffs[i + 1] >= min_frames:
            keep.append(b)
    return np.array(keep)
