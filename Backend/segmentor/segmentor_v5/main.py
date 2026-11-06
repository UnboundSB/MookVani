"""
Entry point for YAP Segmentor v5 fine-tuning.

Builds the YAPpingConformer model, loads the Stage-2 pretrained segmentor
weights, freezes the Conformer / word head, and runs the semi-supervised
Mean-Teacher train/val loop with early stopping and checkpointing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from pathlib import Path

from .config import CFG, DEVICE
from .datasets import LabeledDataset, UnlabeledDataset
from .ema import EMA
from .train import train_one_epoch, validate

from yap_conformer import YAPConfig, YAPpingConformer, ConformerConfig, hand_part


def _build_model() -> nn.Module:
    cfg = YAPConfig(parts=[hand_part("left_hand"), hand_part("right_hand")])
    conformer_cfg = ConformerConfig(num_words=228)
    model = YAPpingConformer(cfg, conformer_cfg).to(DEVICE)

    # Load Stage-2 pretrained segmentor weights if available.
    if CFG.PRETRAINED_CKPT.exists():
        state = torch.load(CFG.PRETRAINED_CKPT, map_location=DEVICE)
        model.load_state_dict(state, strict=False)
        print(f"[INFO] Loaded pretrained segmentor weights from {CFG.PRETRAINED_CKPT}")

    # Freeze Conformer blocks + word head; train only YAP backbone.
    for param in model.blocks.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = False
    for param in model.yap.parameters():
        param.requires_grad = True

    print(f"[INFO] Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return model


def main():
    torch.manual_seed(CFG.SEED)
    print("=" * 60)
    print("YAP Segmentor v5 — Production Fine-tune")
    print("=" * 60)
    print(f"[INFO] Device: {DEVICE}")

    # ---- datasets / loaders ----
    train_dataset = LabeledDataset(max_len=CFG.MAX_LEN, augment=True,
                                   split="train", val_split=CFG.VAL_SPLIT, seed=CFG.SEED)
    val_dataset = LabeledDataset(max_len=CFG.MAX_LEN, augment=False,
                                 split="val", val_split=CFG.VAL_SPLIT, seed=CFG.SEED)
    unlabeled_dataset = UnlabeledDataset(max_len=CFG.MAX_LEN, max_videos=50)

    labeled_loader = DataLoader(train_dataset, batch_size=CFG.LABELED_BATCH_SIZE,
                                shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,
                            num_workers=0, pin_memory=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=CFG.UNLABELED_BATCH_SIZE,
                                  shuffle=True, num_workers=0, pin_memory=True)

    # ---- model / optim ----
    model = _build_model()
    ema = EMA(model.yap, decay=CFG.EMA_DECAY)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY,
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.PLATEAU_PATIENCE,
        factor=CFG.PLATEAU_FACTOR, min_lr=CFG.MIN_LR, verbose=True,
    )
    scaler = GradScaler()

    # ---- train / val loop ----
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, CFG.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{CFG.EPOCHS}")

        train_loss = train_one_epoch(
            model, labeled_loader, unlabeled_loader, optimizer, scaler, ema,
            epoch, CFG.EPOCHS,
        )
        val_loss = validate(model, val_loader, epoch)

        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        scheduler.step(val_loss)

        # ---- checkpoint best ----
        if val_loss < best_val_loss - CFG.MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "ema_state_dict": ema.shadow,
                "val_loss": val_loss,
            }, CFG.CHECKPOINT_DIR / "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= CFG.EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

        # ---- periodic checkpoint ----
        if epoch % CFG.SAVE_EVERY == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "ema_state_dict": ema.shadow,
                "val_loss": val_loss,
            }, CFG.CHECKPOINT_DIR / f"epoch_{epoch}.pth")

    print(f"\n[DONE] Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
