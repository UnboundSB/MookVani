"""
Data Augmentation for YAP Segmentor
"""

import random
import numpy as np
from scipy.interpolate import interp1d


def time_warp(features: np.ndarray, sigma: float = 0.15) -> np.ndarray:
    """Apply Gaussian time warping to features."""
    from scipy.interpolate import interp1d
    T = len(features)
    if T < 2:
        return features
    t_orig = np.arange(T)
    t_warp = t_orig + np.random.normal(0, sigma * T, T)
    t_warp = np.clip(t_warp, 0, T - 1)
    f = interp1d(t_orig, features, axis=0, kind='linear', fill_value='extrapolate', assume_sorted=False)
    return f(t_warp).astype(np.float32)


def horizontal_flip(features: np.ndarray) -> np.ndarray:
    """Flip left/right hands and mirror x-coordinates."""
    flipped = features.copy()
    # Swap left and right hands
    lh = features[:, 0:63].copy()
    rh = features[:, 63:126].copy()
    features[:, 0:63] = rh
    features[:, 63:126] = lh
    
    # Swap hand flags
    flags = features[:, 159:163].copy()
    features[:, 159] = flags[:, 1]
    features[:, 160] = flags[:, 0]
    
    # Swap arms (shoulders and elbows)
    arms = features[:, 126:138].reshape(-1, 4, 3)
    arms[:, [0, 1]] = arms[:, [1, 0]]
    arms[:, [2, 3]] = arms[:, [3, 2]]
    features[:, 126:138] = arms.reshape(-1, 12)
    
    # Mirror x-coordinates
    features[:, 0:63:3] *= -1
    features[:, 63:126:3] *= -1
    features[:, 126:138:3] *= -1
    features[:, 138:159:3] *= -1
    
    return features


def random_shift(features: np.ndarray, max_shift: float = 0.1) -> np.ndarray:
    """Apply random horizontal shift."""
    shifted = features.copy()
    shift = np.random.uniform(-max_shift, max_shift)
    shifted[:, 0:63:3] += shift
    shifted[:, 63:126:3] += shift
    shifted[:, 126:138:3] += shift
    shifted[:, 138:159:3] += shift
    return shifted


def vertical_shift(features: np.ndarray, max_shift: float = 0.1) -> np.ndarray:
    """Apply random vertical shift."""
    shifted = features.copy()
    shift = np.random.uniform(-max_shift, max_shift)
    shifted[:, 1:63:3] += shift
    shifted[:, 64:126:3] += shift
    shifted[:, 127:138:3] += shift
    shifted[:, 139:159:3] += shift
    return shifted


def gaussian_noise(features: np.ndarray, std: float = 0.01) -> np.ndarray:
    """Add Gaussian noise to features."""
    noise = np.random.normal(0, std, features.shape).astype(np.float32)
    return features + noise


def random_dropout(features: np.ndarray, p: float = 0.1) -> np.ndarray:
    """Random dropout of features."""
    mask = np.random.random(features.shape) > p
    return features * mask


def mixup_features(x1: np.ndarray, x2: np.ndarray, y1: np.ndarray, y2: np.ndarray, alpha: float = 0.2):
    """Apply MixUp augmentation to features and labels."""
    if alpha <= 0:
        return x1, y1
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    mixed_x = lam * x1 + (1 - lam) * x2
    mixed_y = lam * y1 + (1 - lam) * y2
    return mixed_x, mixed_y


def augment_features(features: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Apply random augmentations to features."""
    if np.random.random() > 0.5:
        return features
    
    feats = features.copy()
    
    if np.random.random() < 0.3:
        feats = time_warp(feats, sigma=0.15)
    if np.random.random() < 0.5:
        feats = horizontal_flip(feats)
    if np.random.random() < 0.3:
        feats = random_shift(feats, max_shift=0.1)
    if np.random.random() < 0.2:
        feats = gaussian_noise(feats, std=0.01)
    if np.random.random() < 0.1:
        feats = random_dropout(feats, p=0.1)
    
    return feats


def time_warp(features: np.ndarray, sigma: float = 0.15) -> np.ndarray:
    """Apply Gaussian time warping to features."""
    from scipy.interpolate import interp1d
    T = len(features)
    if T < 2:
        return features
    t_orig = np.arange(T)
    t_warp = t_orig + np.random.normal(0, sigma * T, T)
    t_warp = np.clip(t_warp, 0, T - 1)
    f = interp1d(t_orig, features, axis=0, kind='linear', fill_value='extrapolate', assume_sorted=False)
    return f(t_warp).astype(np.float32)


def horizontal_flip(features: np.ndarray) -> np.ndarray:
    """Flip left/right hands and mirror x-coordinates."""
    flipped = features.copy()
    # Swap left and right hands
    lh = features[:, 0:63].copy()
    rh = features[:, 63:126].copy()
    features[:, 0:63] = rh
    features[:, 63:126] = lh
    
    # Swap hand flags
    flags = features[:, 159:163].copy()
    features[:, 159] = flags[:, 1]
    features[:, 160] = flags[:, 0]
    
    # Swap arms (shoulders and elbows)
    arms = features[:, 126:138].reshape(-1, 4, 3)
    arms[:, [0, 1]] = arms[:, [1, 0]]
    arms[:, [2, 3]] = arms[:, [3, 2]]
    features[:, 126:138] = arms.reshape(-1, 12)
    
    # Mirror x-coordinates
    features[:, 0:63:3] *= -1
    features[:, 63:126:3] *= -1
    features[:, 126:138:3] *= -1
    features[:, 138:159:3] *= -1
    
    return features


def random_shift(features: np.ndarray, max_shift: float = 0.1) -> np.ndarray:
    """Apply random horizontal shift."""
    shifted = features.copy()
    shift = np.random.uniform(-max_shift, max_shift)
    shifted[:, 0:63:3] += shift
    shifted[:, 63:126:3] += shift
    shifted[:, 126:138:3] += shift
    shifted[:, 138:159:3] += shift
    return shifted


def vertical_shift(features: np.ndarray, max_shift: float = 0.1) -> np.ndarray:
    """Apply random vertical shift."""
    shifted = features.copy()
    shift = np.random.uniform(-max_shift, max_shift)
    shifted[:, 1:63:3] += shift
    shifted[:, 64:126:3] += shift
    shifted[:, 127:138:3] += shift
    shifted[:, 139:159:3] += shift
    return shifted


def gaussian_noise(features: np.ndarray, std: float = 0.01) -> np.ndarray:
    """Add Gaussian noise to features."""
    noise = np.random.normal(0, std, features.shape).astype(np.float32)
    return features + noise


def random_dropout(features: np.ndarray, p: float = 0.1) -> np.ndarray:
    """Random dropout of features."""
    mask = np.random.random(features.shape) > p
    return features * mask


def mixup_features(x1: np.ndarray, x2: np.ndarray, y1: np.ndarray, y2: np.ndarray, alpha: float = 0.2):
    """Apply MixUp augmentation to features and labels."""
    if alpha <= 0:
        return x1, y1
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    mixed_x = lam * x1 + (1 - lam) * x2
    mixed_y = lam * y1 + (1 - lam) * y2
    return mixed_x, mixed_y


def augment_features(features: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Apply random augmentations to features."""
    if np.random.random() > 0.5:
        return features
    
    feats = features.copy()
    
    if np.random.random() < 0.3:
        feats = time_warp(feats, sigma=0.15)
    if np.random.random() < 0.5:
        feats = horizontal_flip(feats)
    if np.random.random() < 0.3:
        feats = random_shift(feats, max_shift=0.1)
    if np.random.random() < 0.2:
        feats = gaussian_noise(feats, std=0.01)
    if np.random.random() < 0.1:
        feats = random_dropout(feats, p=0.1)
    
    return feats


def time_warp(features: np.ndarray, sigma: float = 0.15) -> np.ndarray:
    """Apply Gaussian time warping to features."""
    from scipy.interpolate import interp1d
    T = len(features)
    if T < 2:
        return features
    t_orig = np.arange(T)
    t_warp = t_orig + np.random.normal(0, sigma * T, T)
    t_warp = np.clip(t_warp, 0, T - 1)
    f = interp1d(t_orig, features, axis=0, kind='linear', fill_value='extrapolate', assume_sorted=False)
    return f(t_warp).astype(np.float32)


def horizontal_flip(features: np.ndarray) -> np.ndarray:
    """Flip left/right hands and mirror x-coordinates."""
    flipped = features.copy()
    # Swap left and right hands
    lh = features[:, 0:63].copy()
    rh = features[:, 63:126].copy()
    features[:, 0:63] = rh
    features[:, 63:126] = lh
    
    # Swap hand flags
    flags = features[:, 159:163].copy()
    features[:, 159] = flags[:, 1]
    features[:, 160] = flags[:, 0]
    
    # Swap arms (shoulders and elbows)
    arms = features[:, 126:138].reshape(-1, 4, 3)
    arms[:, [0, 1]] = arms[:, [1, 0]]
    arms[:, [2, 3]] = arms[:, [3, 2]]
    features[:, 126:138] = arms.reshape(-1, 12)
    
    # Mirror x-coordinates
    features[:, 0:63:3] *= -1
    features[:, 63:126:3] *= -1
    features[:, 126:138:3] *= -1
    features[:, 138:159:3] *= -1
    
    return features


def random_shift(features: np.ndarray, max_shift: float = 0.1) -> np.ndarray:
    """Apply random horizontal shift."""
    shifted = features.copy()
    shift = np.random.uniform(-max_shift, max_shift)
    shifted[:, 0:63:3] += shift
    shifted[:, 63:126:3] += shift
    shifted[:, 126:138:3] += shift
    shifted[:, 138:159:3] += shift
    return shifted


def vertical_shift(features: np.ndarray, max_shift: float = 0.1) -> np.ndarray:
    """Apply random vertical shift."""
    shifted = features.copy()
    shift = np.random.uniform(-max_shift, max_shift)
    shifted[:, 1:63:3] += shift
    shifted[:, 64:126:3] += shift
    shifted[:, 127:138:3] += shift
    shifted[:, 139:159:3] += shift
    return shifted


def gaussian_noise(features: np.ndarray, std: float = 0.01) -> np.ndarray:
    """Add Gaussian noise to features."""
    noise = np.random.normal(0, std, features.shape).astype(np.float32)
    return features + noise


def random_dropout(features: np.ndarray, p: float = 0.1) -> np.ndarray:
    """Random dropout of features."""
    mask = np.random.random(features.shape) > p
    return features * mask


def mixup_features(x1: np.ndarray, x2: np.ndarray, y1: np.ndarray, y2: np.ndarray, alpha: float = 0.2):
    """Apply MixUp augmentation to features and labels."""
    if alpha <= 0:
        return x1, y1
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    mixed_x = lam * x1 + (1 - lam) * x2
    mixed_y = lam * y1 + (1 - lam) * y2
    return mixed_x, mixed_y


def augment_features(features: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Apply random augmentations to features."""
    if np.random.random() > 0.5:
        return features
    
    feats = features.copy()
    
    if np.random.random() < 0.3:
        feats = time_warp(feats, sigma=0.15)
    if np.random.random() < 0.5:
        feats = horizontal_flip(feats)
    if np.random.random() < 0.3:
        feats = random_shift(feats, max_shift=0.1)
    if np.random.random() < 0.2:
        feats = gaussian_noise(feats, std=0.01)
    if np.random.random() < 0.1:
        feats = random_dropout(feats, p=0.1)
    
    return feats


def time_warp(features: np.ndarray, sigma: float = 0.15) -> np.ndarray:
    """Apply Gaussian time warping to features."""
    from scipy.interpolate import interp1d
    T = len(features)
    if T < 2:
        return features
    t_orig = np.arange(T)
    t_warp = t_orig + np.random.normal(0, sigma * T, T)
    t_warp = np.clip(t_warp, 0, T - 1)
    f = interp1d(t_orig, features, axis=0, kind='linear', fill_value='extrapolate', assume_sorted=False)
    return f(t_warp).astype(np.float32)


def horizontal_flip(features: np.ndarray) -> np.ndarray:
    """Flip left/right hands and mirror x-coordinates."""
    flipped = features.copy()
    # Swap left and right hands
    lh = features[:, 0:63].copy()
    rh = features[:, 63:126].copy()
    features[:, 0:63] = rh
    features[:, 63:126] = lh
    
    # Swap hand flags
    flags = features[:, 159:163].copy()
    features[:, 159] = flags[:, 1]
    features[:, 160] = flags[:, 0]
    
    # Swap arms (shoulders and elbows)
    arms = features[:, 126:138].reshape(-1, 4, 3)
    arms[:, [0, 1]] = arms[:, [1, 0]]
    arms[:, [2, 3]] = arms[:, [3, 2]]
    features[:, 126:138] = arms.reshape(-1, 12)
    
    # Mirror x-coordinates
    features[:, 0:63:3] *= -1
    features[:, 63:126:3] *= -1
    features[:, 126:138:3] *= -1
    features[:, 138:159:3] *= -1
    
    return features


def random_shift(features: np.ndarray, max_shift: float = 0.1) -> np.ndarray:
    """Apply random horizontal shift."""
    shifted = features.copy()
    shift = np.random.uniform(-max_shift, max_shift)
    shifted[:, 0:63:3] += shift
    shifted[:, 63:126:3] += shift
    shifted[:, 126:138:3] += shift
    shifted[:, 138:159:3] += shift
    return shifted


def vertical_shift(features: np.ndarray, max_shift: float = 0.1) -> np.ndarray:
    """Apply random vertical shift."""
    shifted = features.copy()
    shift = np.random.uniform(-max_shift, max_shift)
    shifted[:, 1:63:3] += shift
    shifted[:, 64:126:3] += shift
    shifted[:, 127:138:3] += shift
    shifted[:, 139:159:3] += shift
    return shifted


def gaussian_noise(features: np.ndarray, std: float = 0.01) -> np.ndarray:
    """Add Gaussian noise to features."""
    noise = np.random.normal(0, std, features.shape).astype(np.float32)
    return features + noise


def random_dropout(features: np.ndarray, p: float = 0.1) -> np.ndarray:
    """Random dropout of features."""
    mask = np.random.random(features.shape) > p
    return features * mask


def mixup_features(x1: np.ndarray, x2: np.ndarray, y1: np.ndarray, y2: np.ndarray, alpha: float = 0.2):
    """Apply MixUp augmentation to features and labels."""
    if alpha <= 0:
        return x1, y1
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    mixed_x = lam * x1 + (1 - lam) * x2
    mixed_y = lam * y1 + (1 - lam) * y2
    return mixed_x, mixed_y


def augment_features(features: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Apply random augmentations to features."""
    if np.random.random() > 0.5:
        return features
    
    feats = features.copy()
    
    if np.random.random() < 0.3:
        feats = time_warp(feats, sigma=0.15)
    if np.random.random() < 0.5:
        feats = horizontal_flip(feats)
    if np.random.random() < 0.3:
        feats = random_shift(feats, max_shift=0.1)
    if np.random.random() < 0.2:
        feats = gaussian_noise(feats, std=0.01)
    if np.random.random() < 0.1:
        feats = random_dropout(feats, p=0.1)
    
    return feats


def time_warp(features: np.ndarray, sigma: float = 0.15) -> np.ndarray:
    """Apply Gaussian time warping to features."""
    from scipy.interpolate import interp1d
    T = len(features)
    if T < 2:
        return features
    t_orig = np.arange(T)
    t_warp = t_orig + np.random.normal(0, sigma * T, T)
    t_warp = np.clip(t_warp, 0, T - 1)
    f = interp1d(t_orig, features, axis=0, kind='linear', fill_value='extrapolate', assume_sorted=False)
    return f(t_warp).astype(np.float32)


def horizontal_flip(features: np.ndarray) -> np.ndarray:
    """Flip left/right hands and mirror x-coordinates."""
    flipped = features.copy()
    # Swap left and right hands
    lh = features[:, 0:63].copy()
    rh = features[:, 63:126].copy()
    features[:, 0:63] = rh
    features[:, 63:126] = lh
    
    # Swap hand flags
    flags = features[:, 159:163].copy()
    features[:, 159] = flags[:, 1]
    features[:, 160] = flags[:, 0]
    
    # Swap arms (shoulders and elbows)
    arms = features[:, 126:138].reshape(-1, 4, 3)
    arms[:, [0, 1]] = arms[:, [1, 0]]
    arms[:, [2, 3]] = arms[:, [3, 2]]
    features[:, 126:138] = arms.reshape(-1, 12)
    
    # Mirror x-coordinates
    features[:, 0:63:3] *= -1
    features[:, 63:126:3] *= -1
    features[:, 126:138:3] *= -1
    features[:, 138:159:3] *= -1
    
    return flipped


def random_shift(features: np.ndarray, max_shift: float = 0.1) -> np.ndarray:
    """Apply random horizontal shift."""
    shifted = features.copy()
    shift = np.random.uniform(-max_shift, max_shift)
    shifted[:, 0:63:3] += shift
    shifted[:, 63:126:3] += shift
    shifted[:, 126:138:3] += shift
    shifted[:, 138:159:3] += shift
    return shifted


def vertical_shift(features: np.ndarray, max_shift: float = 0.1) -> np.ndarray:
    """Apply random vertical shift."""
    shifted = features.copy()
    shift = np.random.uniform(-max_shift, max_shift)
    shifted[:, 1:63:3] += shift
    shifted[:, 64:126:3] += shift
    shifted[:, 127:138:3] += shift
    shifted[:, 139:159:3] += shift
    return shifted


def gaussian_noise(features: np.ndarray, std: float = 0.01) -> np.ndarray:
    """Add Gaussian noise to features."""
    noise = np.random.normal(0, std, features.shape).astype(np.float32)
    return features + noise


def random_dropout(features: np.ndarray, p: float = 0.1) -> np.ndarray:
    """Random dropout of features."""
    mask = np.random.random(features.shape) > p
    return features * mask


def mixup_features(x1: np.ndarray, x2: np.ndarray, y1: np.ndarray, y2: np.ndarray, alpha: float = 0.2):
    """Apply MixUp augmentation to features and labels."""
    if alpha <= 0:
        return x1, y1
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    mixed_x = lam * x1 + (1 - lam) * x2
    mixed_y = lam * y1 + (1 - lam) * y2
    return mixed_x, mixed_y


def augment_features(features: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Apply random augmentations to features."""
    if np.random.random() > 0.5:
        return features
    
    feats = features.copy()
    
    if np.random.random() < 0.3:
        feats = time_warp(feats, sigma=0.15)
    if np.random.random() < 0.5:
        feats = horizontal_flip(feats)
    if np.random.random() < 0.3:
        feats = random_shift(feats, max_shift=0.1)
    if np.random.random() < 0.2:
        feats = gaussian_noise(feats, std=0.01)
    if np.random.random() < 0.1:
        feats = random_dropout(feats, p=0.1)
    
    return feats