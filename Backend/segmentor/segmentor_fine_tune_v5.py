"""
Backward-compatible shim for ``segmentor_fine_tune_v5.py``.

The monolithic 1347-line file was split into the modular package
``segmentor/segmentor_v5/`` (config, extractor, cache, augment, boundaries,
losses, ema, datasets, train, main). This shim re-exports everything so existing
imports / entry points keep working:

    python segmentor\segmentor_fine_tune_v5.py   # -> segmentor_v5.main.main()
"""

import sys
from pathlib import Path

# Ensure the segmentor dir (where yap_conformer.py lives) is importable.
_SEGMENTOR_DIR = Path(__file__).resolve().parent
if str(_SEGMENTOR_DIR) not in sys.path:
    sys.path.insert(0, str(_SEGMENTOR_DIR))

from segmentor_v5 import main   # noqa: E402
from segmentor_v5 import (   # noqa: E402
    CFG, Config,
    MediaPipeFeatureExtractor,
    FeatureCache,
    apply_mixup, time_warp, horizontal_flip, random_shift,
    gaussian_noise, random_dropout, augment_features,
    nms_boundaries, smooth_boundaries, enforce_min_segment,
    focal_bce_loss, bce_with_logits_loss, temporal_smooth_loss,
    boundary_nms_penalty, compute_losses,
    EMA, LabeledDataset, UnlabeledDataset,
    train_one_epoch, validate,
)

if __name__ == "__main__":
    main.main()
