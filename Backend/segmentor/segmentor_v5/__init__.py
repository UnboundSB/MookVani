"""
YAP Segmentor v5 — modular production fine-tune pipeline.

This package replaces the monolithic ``segmentor_fine_tune_v5.py`` (which grew to
1347 lines and contained triplicated classes). Responsibilities are split across
single-purpose modules so no file approaches the 1300-line ceiling:

    config      - paths, hardware, training hyper-parameters (single source of truth)
    extractor   - MediaPipeFeatureExtractor (163-dim pose/hand/face features)
    cache       - FeatureCache (npz feature cache keyed by video path hash)
    augment     - time_warp, horizontal_flip, random_shift, gaussian_noise,
                  random_dropout, augment_features, apply_mixup
    boundaries  - nms_boundaries, smooth_boundaries, enforce_min_segment
    losses      - focal / BCE / temporal-smooth / boundary-NMS losses + compute_losses
    ema         - EMA teacher for Mean-Teacher semi-supervised training
    datasets    - LabeledDataset, UnlabeledDataset (train/val split, boundary targets)
    train       - train_one_epoch, validate
    main        - entry point (model build, pretrained load, train/val loop)

Usage (from D:\MookVani\Backend):
    python segmentor\segmentor_fine_tune_v5.py      # backward-compat shim
    python -m segmentor.segmentor_v5                # if segmentor/ is a package
"""

from .config import CFG, Config
from .extractor import MediaPipeFeatureExtractor
from .cache import FeatureCache
from .augment import (
    apply_mixup, time_warp, horizontal_flip, random_shift,
    gaussian_noise, random_dropout, augment_features,
)
from .boundaries import nms_boundaries, smooth_boundaries, enforce_min_segment
from .losses import (
    focal_bce_loss, bce_with_logits_loss, temporal_smooth_loss,
    boundary_nms_penalty, compute_losses,
)
from .ema import EMA
from .datasets import LabeledDataset, UnlabeledDataset
from .train import train_one_epoch, validate
from .main import main

__all__ = [
    "CFG", "Config",
    "MediaPipeFeatureExtractor",
    "FeatureCache",
    "apply_mixup", "time_warp", "horizontal_flip", "random_shift",
    "gaussian_noise", "random_dropout", "augment_features",
    "nms_boundaries", "smooth_boundaries", "enforce_min_segment",
    "focal_bce_loss", "bce_with_logits_loss", "temporal_smooth_loss",
    "boundary_nms_penalty", "compute_losses",
    "EMA", "LabeledDataset", "UnlabeledDataset",
    "train_one_epoch", "validate", "main",
]
