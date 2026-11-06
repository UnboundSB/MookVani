"""
Data augmentations applied to 163-dim feature sequences (T, 163).

Layout offsets (53 joints x 3 = 159 coords + 4 flags):
    hands   0:126   (left 0:63, right 63:126)
    arms    126:138 (4 x 3)
    face    138:159 (7 x 3)
    flags   159:163 (lh, rh, arms, face)
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


# ------------------------------------------------------------------ time warp
def time_warp(features: np.ndarray, sigma: float = 0.15) -> np.ndarray:
    """Elastically warp the time axis of a feature sequence."""
    T = len(features)
    if T < 2:
        return features
    t_orig = np.arange(T)
    t_warp = t_orig + np.random.normal(0, sigma * T, T)
    t_warp = np.clip(t_warp, 0, T - 1)
    from scipy.interpolate import interp1d
    f = interp1d(t_orig, features, axis=0, kind="linear",
                 fill_value="extrapolate", assume_sorted=False)
    return f(t_warp).astype(np.float32)


# ------------------------------------------------------------------ flips / shifts
def horizontal_flip(features: np.ndarray) -> np.ndarray:
    """Swap left/right hands and mirror x-coordinates."""
    flipped = features.copy()
    lh = flipped[:, 0:63].copy()
    rh = flipped[:, 63:126].copy()
    flipped[:, 0:63] = rh
    flipped[:, 63:126] = lh
    flags = flipped[:, 159:163].copy()
    flipped[:, 159] = flags[:, 1]
    flipped[:, 160] = flags[:, 0]
    arms = flipped[:, 126:138].reshape(-1, 4, 3)
    arms[:, [0, 1]] = arms[:, [1, 0]]
    arms[:, [2, 3]] = arms[:, [3, 2]]
    flipped[:, 126:138] = arms.reshape(-1, 12)
    flipped[:, 0:63:3] *= -1
    flipped[:, 63:126:3] *= -1
    flipped[:, 126:138:3] *= -1
    flipped[:, 138:159:3] *= -1
    return flipped


def random_shift(features: np.ndarray, max_shift: float = 0.1) -> np.ndarray:
    """Add a small constant offset to every x coordinate."""
    shifted = features.copy()
    shift = random.uniform(-max_shift, max_shift)
    shifted[:, 0:63:3] += shift
    shifted[:, 63:126:3] += shift
    shifted[:, 126:138:3] += shift
    shifted[:, 138:159:3] += shift
    return shifted


# ------------------------------------------------------------------ noise / dropout
def gaussian_noise(features: np.ndarray, std: float = 0.01) -> np.ndarray:
    noise = np.random.normal(0, std, features.shape).astype(np.float32)
    return features + noise


def random_dropout(features: np.ndarray, p: float = 0.1) -> np.ndarray:
    mask = np.random.random(features.shape) > p
    return features * mask


# ------------------------------------------------------------------ composed
def augment_features(features: np.ndarray, aug_prob: float = 0.5) -> np.ndarray:
    """Apply a random subset of augmentations with independent probabilities."""
    if random.random() > aug_prob:
        return features

    feats = features.copy()
    if random.random() < 0.3:
        feats = time_warp(feats, sigma=0.15)
    if random.random() < 0.5:
        feats = horizontal_flip(feats)
    if random.random() < 0.3:
        feats = random_shift(feats, max_shift=0.1)
    if random.random() < 0.2:
        feats = gaussian_noise(feats, std=0.01)
    if random.random() < 0.1:
        feats = random_dropout(feats, p=0.1)
    return feats


# ------------------------------------------------------------------ mixup
def apply_mixup(x1: torch.Tensor, x2: torch.Tensor,
                y1: torch.Tensor, y2: torch.Tensor,
                alpha: float = 0.2):
    """MixUp augmentation for features and labels."""
    if alpha <= 0:
        return x1, y1
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    mixed_x = lam * x1 + (1 - lam) * x2
    mixed_y = lam * y1 + (1 - lam) * y2
    return mixed_x, mixed_y
