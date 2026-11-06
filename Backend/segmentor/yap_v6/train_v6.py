"""
YAP Segmentor v6 — Production Fine-tune
Clean, working training script
"""

import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yap_v6.config import cfg
from yap_v6.extractor import MediaPipeFeatureExtractor
from yap_v6.datasets import create_dataloaders
from yap_v6.model import create_model, EMA
from yap_v6.losses import compute_losses
from yap_v6.training import train, validate


def main():
    print("=" * 60)
    print("YAP Segmentor v6 — Production Fine-tune")
    print("=" * 60)

    # Datasets
    from yap_v6.datasets import create_dataloaders
    labeled_loader, val_loader, unlabeled_loader = create_dataloaders(
        labeled_batch_size=8,
        unlabeled_batch_size=8,
        val_split=0.2
    )

    # Model
    from yap_v6.model import create_model
    model = create_model()

    # EMA teacher
    from yap_v6.model import EMA
    ema = EMA(model.yap, decay=0.999)

    # Optimizer
    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.amp import GradScaler

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=1e-3,
    )
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-6, verbose=True)
    scaler = torch.amp.GradScaler('cuda')

    from yap_v6.training import train, validate

    train(model, labeled_loader, unlabeled_loader, val_loader, optimizer, scheduler, scaler, 50)

    print("\n[DONE] Training complete")


if __name__ == "__main__":
    main()