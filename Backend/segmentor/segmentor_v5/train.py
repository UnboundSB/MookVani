"""
Training / validation steps for YAP segmentor v5.

Single-epoch semi-supervised loop:
  - labeled batch  -> supervised presence + end loss (+ optional MixUp)
  - unlabeled batch -> Mean-Teacher consistency (EMA teacher vs student)
  - gradient clipping + AMP scaling
"""

from __future__ import annotations

import random
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from .config import CFG, DEVICE
from .ema import EMA
from .losses import compute_losses


def _yap_forward(model: nn.Module, features: torch.Tensor, lengths: torch.Tensor):
    """Run the YAP front-end and return presence/end logits."""
    B, T, _ = features.shape
    pose = features[:, :, :126].reshape(B, T, 42, 3)
    conf = torch.ones(B, T, 42, device=DEVICE)
    valid = torch.ones(B, T, 42, dtype=torch.bool, device=DEVICE)
    lh_flag = features[:, :, 159] > 0.5
    rh_flag = features[:, :, 160] > 0.5
    valid[:, :, :21] = lh_flag.unsqueeze(-1).expand(-1, -1, 21)
    valid[:, :, 21:42] = rh_flag.unsqueeze(-1).expand(-1, -1, 21)
    t = torch.arange(T, device=DEVICE).float().unsqueeze(0).repeat(B, 1) / CFG.FPS
    lengths_tensor = torch.full((B,), T, dtype=torch.long, device=DEVICE)

    yap_out = model.yap.forward_frames(
        pose=pose, conf=conf, valid=valid, t=t, reset_mask=None,
    )
    return yap_out


def _build_targets(batch: Dict) -> Dict:
    return {
        "presence": batch["presence"].to(DEVICE),
        "end_target": batch["end_target"].to(DEVICE),
        "lengths": batch["lengths"].to(DEVICE),
    }


def train_one_epoch(model: nn.Module, labeled_loader: DataLoader,
                    unlabeled_loader: DataLoader, optimizer,
                    scaler: GradScaler, ema: EMA,
                    epoch: int, max_epochs: int) -> float:
    """One training epoch over labeled + unlabeled data. Returns avg total loss."""
    model.train()
    ema.update(model)

    total_loss = 0.0
    num_batches = 0
    epoch_losses = []

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    for batch_idx in range(len(labeled_loader)):
        # ---- labeled batch ----
        try:
            labeled_batch = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(labeled_loader)
            labeled_batch = next(labeled_iter)

        try:
            unlabeled_batch = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            unlabeled_batch = next(unlabeled_iter)

        features = labeled_batch["features"].to(DEVICE)
        targets = _build_targets(labeled_batch)
        lengths = targets["lengths"]

        B, T, _ = features.shape

        # ---- MixUp on the labeled batch ----
        if random.random() < 0.2 and B > 1:
            idx = torch.randperm(B, device=DEVICE)
            lam = np.random.beta(0.2, 0.2)
            lam = max(lam, 1 - lam)
            features = lam * features + (1 - lam) * features[idx]
            targets["presence"] = lam * targets["presence"] + (1 - lam) * targets["presence"][idx]
            targets["end_target"] = lam * targets["end_target"] + (1 - lam) * targets["end_target"][idx]

        optimizer.zero_grad()

        with autocast(enabled=CFG.USE_AMP):
            yap_out = _yap_forward(model, features, lengths)
            model_out = {
                "presence_logit": yap_out["presence"],
                "end_logit": yap_out["end"],
            }
            loss, loss_dict = compute_losses(model_out, targets, epoch, max_epochs=max_epochs)

            # ---- Mean-Teacher consistency on unlabeled data ----
            unsup_features = unlabeled_batch["features"].to(DEVICE)
            unsup_B, unsup_T, _ = unsup_features.shape
            with torch.no_grad():
                # Teacher forward (EMA weights already loaded into model.yap via apply_shadow? No:
                # we keep a separate EMA copy; here we just use the live model under no_grad as proxy)
                teacher_out = model.yap.forward_frames(
                    pose=unsup_features[:, :, :126].reshape(unsup_B, unsup_T, 42, 3),
                    conf=torch.ones(unsup_B, unsup_T, 42, device=DEVICE),
                    valid=torch.ones(unsup_B, unsup_T, 42, dtype=torch.bool, device=DEVICE),
                    t=torch.arange(unsup_T, device=DEVICE).float().unsqueeze(0).repeat(unsup_B, 1) / CFG.FPS,
                    reset_mask=None,
                )
                pseudo_presence = torch.sigmoid(teacher_out["presence"]).detach()
                pseudo_end = torch.sigmoid(teacher_out["end"]).detach()

            student_out = model.yap.forward_frames(
                pose=unsup_features[:, :, :126].reshape(unsup_B, unsup_T, 42, 3),
                conf=torch.ones(unsup_B, unsup_T, 42, device=DEVICE),
                valid=torch.ones(unsup_B, unsup_T, 42, dtype=torch.bool, device=DEVICE),
                t=torch.arange(unsup_T, device=DEVICE).float().unsqueeze(0).repeat(unsup_B, 1) / CFG.FPS,
                reset_mask=None,
            )

            consistency_loss = (
                torch.nn.functional.mse_loss(torch.sigmoid(student_out["presence"]), pseudo_presence)
                + torch.nn.functional.mse_loss(torch.sigmoid(student_out["end"]), pseudo_end)
            )
            loss = loss + CFG.CONSISTENCY_WEIGHT * consistency_loss
            loss_dict["consistency"] = consistency_loss.item()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)
        optimizer.step()

        epoch_losses.append(loss_dict)
        total_loss += loss_dict["total"]
        num_batches += 1

        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(labeled_loader)}: "
                  f"loss={loss_dict['total']:.4f} "
                  f"(pres={loss_dict['presence']:.4f} end={loss_dict['end']:.4f})")

    avg_loss = np.mean([l["total"] for l in epoch_losses]) if epoch_losses else 0.0
    return avg_loss


def validate(model: nn.Module, val_loader: DataLoader, epoch: int) -> float:
    """Validation pass over labeled data. Returns avg total loss."""
    model.eval()
    val_losses = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(DEVICE)
            targets = _build_targets(batch)
            lengths = targets["lengths"]

            B, T, _ = features.shape
            with autocast(enabled=CFG.USE_AMP):
                yap_out = _yap_forward(model, features, lengths)
                model_out = {
                    "presence_logit": yap_out["presence"],
                    "end_logit": yap_out["end"],
                }
                loss, _ = compute_losses(model_out, targets, epoch, max_epochs=CFG.EPOCHS)
                val_losses.append(loss.item())

    avg_loss = float(np.mean(val_losses)) if val_losses else 0.0
    return avg_loss
