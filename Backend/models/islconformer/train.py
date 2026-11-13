#!/usr/bin/env python
"""
Train Minimal ISL Conformer on F:\datasets\mookvani_dataset

Usage:
    python -m models.islconformer.train --epochs 50 --batch_size 32 --lr 3e-4
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

# ─── Paths ───
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BACKEND_DIR)

from models.islconformer.model import MinimalISLConformer
from models.islconformer.dataset import (
    build_word_dataloaders, collate_word_batch, WordTensorDataset
)

DATA_ROOT = r"F:\datasets\mookvani_dataset\ISL_CSLRT_Corpus\Frames_Word_Level"
CLASSES_JSON = os.path.join(BACKEND_DIR, "models", "word_vocabulary", "classes.json")
SAVE_DIR = os.path.join(BACKEND_DIR, "models", "islconformer")


def load_classes(path):
    with open(path) as f:
        classes = json.load(f)
    if isinstance(classes, dict):
        classes = classes.get("classes", classes)
    return classes


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load vocabulary
    classes = load_classes(args.classes)
    num_classes = len(classes)
    print(f"[INFO] {num_classes} classes from {args.classes}")

    # Data
    train_loader, val_loader = build_word_dataloaders(
        args.data_root, classes,
        batch_size=args.batch_size,
        max_len=args.max_len,
        num_workers=args.num_workers,
        val_split=args.val_split,
        use_tensors=args.use_tensors
    )

    # Model
    model = MinimalISLConformer(
        input_dim=163,
        num_classes=num_classes,
        d_model=args.d_model,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(device)

    print(f"[MODEL] {sum(p.numel() for p in model.parameters()):,} params")

    # Load pretrained if specified
    if args.init and os.path.exists(args.init):
        ckpt = torch.load(args.init, map_location=device, weights_only=False)
        state = ckpt.get('model', ckpt)
        model.load_state_dict(state, strict=False)
        print(f"[INIT] Loaded weights from {args.init}")

    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    os.makedirs(args.out_dir, exist_ok=True)
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n{'='*60}")
    print(f"Training Minimal ISL Conformer")
    print(f"  Classes: {num_classes} | Device: {device}")
    print(f"  Batch: {args.batch_size} | Epochs: {args.epochs} | LR: {args.lr}")
    print(f"  d_model: {args.d_model} | heads: {args.num_heads}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{args.epochs} [train]", leave=False)
        for clips, labels, lengths in pbar:
            clips, labels = clips.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(clips, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100*correct/total:.1f}%")

        train_acc = correct / total
        avg_train_loss = train_loss / len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for clips, labels, lengths in val_loader:
                clips, labels = clips.to(device), labels.to(device)
                logits = model(clips, lengths)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step()

        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"Train Loss: {avg_train_loss:.3f} Acc: {train_acc:.3f} | "
              f"Val Loss: {avg_val_loss:.3f} Acc: {val_acc:.3f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'classes': classes,
                'val_acc': val_acc,
            }, os.path.join(args.out_dir, "best_model.pth"))
            print(f"  ★ New best: {val_acc:.3f}")

        # Save latest
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'history': history,
        }, os.path.join(args.out_dir, "latest.pth"))

        # Save history
        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val acc: {best_acc:.3f}")
    print(f"Model saved to {args.out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Train Minimal ISL Conformer")
    p.add_argument("--data_root", default=r"F:\datasets\mookvani_dataset\ISL_CSLRT_Corpus\Frames_Word_Level")
    p.add_argument("--classes", default=os.path.join(BACKEND_DIR, "models", "word_vocabulary", "classes.json"))
    p.add_argument("--out_dir", default=SAVE_DIR)
    p.add_argument("--init", default=None, help="Pretrained checkpoint")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_len", type=int, default=120)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--use_tensors", action="store_true", help="Use pre-extracted .pt tensors")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    import numpy as np
    train(parse_args())