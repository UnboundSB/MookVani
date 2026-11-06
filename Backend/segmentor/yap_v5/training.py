"""
Training loop for YAP Segmentor v5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import random
from pathlib import Path
from typing import Dict, Tuple, Optional, Iterator

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from .config import cfg
from .losses import compute_losses
from .model import EMA, count_parameters
from .datasets import LabeledDataset, UnlabeledDataset, create_dataloaders
from .model import create_model


def _ensure_device(model: nn.Module, device: torch.device):
    """Recursively ensure all tensors in model are on the correct device."""
    for param in model.parameters():
        if param.device != device:
            param.data = param.data.to(device)
    for buffer in model.buffers():
        if buffer.device != device:
            buffer.data = buffer.data.to(device)
    for module in model.modules():
        if hasattr(module, '_non_persistent_buffers_set'):
            for buffer_name in module._non_persistent_buffers_set:
                if hasattr(module, buffer_name):
                    buf = getattr(module, buffer_name)
                    if isinstance(buf, torch.Tensor) and buf.device != device:
                        setattr(module, buffer_name, buf.to(device))


def train_one_epoch(
    model: nn.Module,
    labeled_loader: Iterator,
    unlabeled_loader: Iterator,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    ema: EMA,
    epoch: int,
    max_epochs: int,
    cfg
) -> float:
    """Train for one epoch."""
    model.train()
    # Ensure all model tensors are on the correct device
    _ensure_device(model, cfg.DEVICE)
    
    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)
    
    epoch_losses = []
    num_batches = len(labeled_loader)
    
    for batch_idx in range(len(labeled_loader)):
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
        
        # Labeled batch
        features = labeled_batch['features'].to(cfg.DEVICE)
        presence_gt = labeled_batch['presence'].to(cfg.DEVICE)
        end_target_gt = labeled_batch['end_target'].to(cfg.DEVICE)
        lengths = labeled_batch['lengths'].to(cfg.DEVICE)
        
        B, T, _ = features.shape
        
        # MixUp augmentation
        if random.random() < 0.2 and B > 1:
            idx = torch.randperm(B)
            lam = np.random.beta(0.2, 0.2)
            lam = max(lam, 1 - lam)
            features = lam * features + (1 - lam) * features[idx]
            presence_gt = lam * presence_gt + (1 - lam) * presence_gt[idx]
            end_target_gt = lam * end_target_gt + (1 - lam) * end_target_gt[idx]
        
        with torch.amp.autocast(device_type='cuda', enabled=True):
            # Forward through YAP
            pose = features[:, :, :126].reshape(B, T, 42, 3)
            conf = torch.ones(B, T, 42, device=cfg.DEVICE)
            valid = torch.ones(B, T, 42, dtype=torch.bool, device=cfg.DEVICE)
            
            lh_flag = features[:, :, 159] > 0.5
            rh_flag = features[:, :, 160] > 0.5
            valid[:, :, :21] = lh_flag.unsqueeze(-1).expand(-1, -1, 21)
            valid[:, :, 21:42] = rh_flag.unsqueeze(-1).expand(-1, -1, 21)
            
            t = torch.arange(T, device=cfg.DEVICE).float().unsqueeze(0).repeat(B, 1) / cfg.FPS
            lengths_tensor = torch.full((B,), T, dtype=torch.long, device=cfg.DEVICE)
            
            yap_out = model.yap.forward_frames(
                pose=pose,
                conf=conf,
                valid=valid,
                t=t,
                reset_mask=None,
            )
            
            presence_logit = yap_out['presence']
            end_logit = yap_out['end']
            
            presence_gt = labeled_batch['presence'].to(cfg.DEVICE)
            end_target_gt = labeled_batch['end_target'].to(cfg.DEVICE)
            
            model_out = {'presence_logit': presence_logit, 'end_logit': end_logit}
            targets_dict = {
                'presence': presence_gt,
                'end_target': end_target_gt,
                'lengths': lengths,
            }
            
            loss, loss_dict = compute_losses(model_out, targets_dict, epoch, max_epochs=50)
            
            # Semi-supervised consistency loss
            if unlabeled_loader is not None:
                try:
                    unsup_batch = next(unlabeled_iter)
                except StopIteration:
                    unsup_iter = iter(unlabeled_loader)
                    unsup_batch = next(unsup_iter)
                
                unsup_features = unsup_batch['features'].to(cfg.DEVICE)
                unsup_B, unsup_T, _ = unsup_features.shape
                
                with torch.no_grad():
                    # Teacher predictions (EMA)
                    unsup_pose = unsup_features[:, :, :126].reshape(unsup_B, unsup_T, 42, 3)
                    unsup_conf = torch.ones(unsup_B, unsup_T, 42, device=cfg.DEVICE)
                    unsup_valid = torch.ones(unsup_B, unsup_T, 42, dtype=torch.bool, device=cfg.DEVICE)
                    unsup_t = torch.arange(unsup_T, device=cfg.DEVICE).float().unsqueeze(0).repeat(unsup_B, 1) / cfg.FPS
                    
                    with torch.no_grad():
                        unsup_out = model.yap.forward_frames(
                            pose=unsup_pose,
                            conf=unsup_conf,
                            valid=unsup_valid,
                            t=unsup_t,
                            reset_mask=None,
                        )
                    
                    pseudo_presence = torch.sigmoid(unsup_out['presence']).detach()
                    pseudo_end = torch.sigmoid(unsup_out['end']).detach()
                
                # Student forward on unlabeled
                student_out = model.yap.forward_frames(
                    pose=unsup_features[:, :, :126].reshape(unsup_B, unsup_T, 42, 3),
                    conf=torch.ones(unsup_B, unsup_T, 42, device=cfg.DEVICE),
                    valid=torch.ones(unsup_B, unsup_T, 42, dtype=torch.bool, device=cfg.DEVICE),
                    t=torch.arange(unsup_T, device=cfg.DEVICE).float().unsqueeze(0).repeat(unsup_B, 1) / cfg.FPS,
                    reset_mask=None,
                )
                
                consistency_loss = F.mse_loss(
                    torch.sigmoid(student_out['presence']), pseudo_presence
                ) + F.mse_loss(
                    torch.sigmoid(student_out['end']), pseudo_end
                )
                
                loss = loss + 0.1 * consistency_loss
                loss_dict['consistency'] = consistency_loss.item()
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            optimizer.step()
            
            epoch_losses.append(loss_dict)
        
        avg_loss = np.mean([l['total'] for l in epoch_losses])
        return avg_loss


def validate(model: nn.Module, val_loader: DataLoader, epoch: int, max_epochs: int) -> float:
    """Validate model on validation set."""
    model.eval()
    val_losses = []
    
    with torch.no_grad():
        for batch in val_loader:
            features = batch['features'].to(cfg.DEVICE)
            presence_gt = batch['presence'].to(cfg.DEVICE)
            end_target = batch['end_target'].to(cfg.DEVICE)
            lengths = batch['lengths'].to(cfg.DEVICE)
            
            B, T, _ = features.shape
            
            # Convert to YAP input format
            pose = features[:, :, :126].reshape(B, T, 42, 3)
            conf = torch.ones(B, T, 42, device=cfg.DEVICE)
            valid = torch.ones(B, T, 42, dtype=torch.bool, device=cfg.DEVICE)
            
            lh_flag = features[:, :, 159] > 0.5
            rh_flag = features[:, :, 160] > 0.5
            valid[:, :, :21] = lh_flag.unsqueeze(-1).expand(-1, -1, 21)
            valid[:, :, 21:42] = rh_flag.unsqueeze(-1).expand(-1, -1, 21)
            
            t = torch.arange(T, device=cfg.DEVICE).float().unsqueeze(0).repeat(B, 1) / cfg.FPS
            lengths_tensor = torch.full((B,), T, dtype=torch.long, device=cfg.DEVICE)
            
            yap_out = model.yap.forward_frames(
                pose=pose,
                conf=conf,
                valid=valid,
                t=t,
                reset_mask=None,
            )
            
            presence_logit = yap_out['presence']
            end_logit = yap_out['end']
            
            presence_gt = batch['presence'].to(cfg.DEVICE)
            end_target = batch['end_target'].to(cfg.DEVICE)
            lengths = batch['lengths'].to(cfg.DEVICE)
            
            model_out = {'presence_logit': presence_logit, 'end_logit': end_logit}
            targets_dict = {
                'presence': presence_gt,
                'end_target': end_target,
                'lengths': lengths,
            }
            
            loss, loss_dict = compute_losses(model_out, targets_dict, epoch, max_epochs=50)
            val_losses.append(loss_dict)
    
    avg_loss = np.mean([l['total'] for l in val_losses])
    return avg_loss


def train(
    model: nn.Module,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    ema: EMA,
    epochs: int,
    checkpoint_dir: Path,
    cfg
):
    """Main training loop."""
    print(f"[INFO] Starting training for {cfg.EPOCHS} epochs")
    print(f"[INFO] Device: {cfg.DEVICE}")
    print(f"[INFO] Labeled batches/epoch: {len(labeled_loader)}")
    print(f"[INFO] Unlabeled batches/epoch: {len(unlabeled_loader)}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, cfg.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{cfg.EPOCHS}")
        
        # Train
        train_loss = train_one_epoch(
            model, labeled_loader, unlabeled_loader,
            optimizer, scaler, ema, epoch, cfg.EPOCHS, cfg
        )
        
        # Validate
        val_loss = validate(model, val_loader, epoch, cfg.EPOCHS)
        
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Checkpoint
        if epoch % 3 == 0:
            ckpt_path = cfg.CHECKPOINT_DIR / f"epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'ema_state_dict': ema.shadow,
                'val_loss': val_loss,
            }, ckpt_path)
            print(f"[INFO] Saved checkpoint: {ckpt_path}")
        
        # Best model checkpoint
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'ema_state_dict': ema.shadow,
                'val_loss': val_loss,
            }, cfg.CHECKPOINT_DIR / "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= 8:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if epoch % 3 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'ema_state_dict': ema.shadow,
                'val_loss': val_loss,
            }, cfg.CHECKPOINT_DIR / f"epoch_{epoch}.pth")
    
    print(f"\n[DONE] Best val loss: {best_val_loss:.4f}")


def main():
    from .config import cfg
    from .datasets import create_dataloaders
    from .model import create_model, EMA, count_parameters
    
    print("=" * 60)
    print("YAP Segmentor v5 — Production Fine-tune")
    print("=" * 60)
    
    # Datasets
    labeled_loader, val_loader, unlabeled_loader = create_dataloaders()
    
    # Model
    model = create_model()
    
    # EMA teacher
    ema = EMA(model.yap, decay=0.999)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=1e-3,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-6, verbose=True)
    scaler = torch.cuda.amp.GradScaler()
    
    # Train
    train(model, labeled_loader, unlabeled_loader, val_loader, optimizer, scaler, ema, 50)
    
    # Cleanup
    # (cleanup handled in dataset __del__ or manually)


if __name__ == "__main__":
    main()