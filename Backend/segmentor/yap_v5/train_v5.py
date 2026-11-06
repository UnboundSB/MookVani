"""
YAP Segmentor v5 — Production Fine-tuning Entry Point
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
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yap_v5.config import cfg
from yap_v5.extractor import MediaPipeFeatureExtractor
from yap_v5.datasets import LabeledDataset, UnlabeledDataset, create_dataloaders
from yap_v5.model import create_model, EMA, count_parameters
from yap_v5.training import train, validate
from yap_v5.losses import compute_losses

# ────────────────────────── Main ──────────────────────────

def main():
    print("=" * 60)
    print("YAP Segmentor v5 — Production Fine-tune")
    print("=" * 60)
    
    # Datasets
    labeled_loader, val_loader, unlabeled_loader = create_dataloaders(
        labeled_batch_size=8,
        unlabeled_batch_size=8,
        val_split=0.2
    )
    
    # Model
    model = create_model()
    
    # EMA teacher
    from yap_v5.model import EMA
    ema = EMA(model.yap, decay=0.999)
    
    # Optimizer
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=1e-3,
    )
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-6, verbose=True)
    from torch.cuda.amp import GradScaler
    scaler = GradScaler()
    
    # Train
    from yap_v5.training import train, validate
    
    train(
        model=model,
        labeled_loader=labeled_loader,
        unlabeled_loader=unlabeled_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        ema=ema,
        epochs=50,
        checkpoint_dir=cfg.CHECKPOINT_DIR,
        cfg=cfg
    )
    
    print("\n[DONE] Training complete")


if __name__ == "__main__":
    main()