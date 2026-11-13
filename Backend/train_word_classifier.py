"""
Minimal Word Classifier Training — Initializes from Conformer-CTC backbone.

Uses the pretrained ISLHybridConformerCTC backbone (loads 185/185 tensors)
and adds a fresh classification head for the 228-class vocabulary.

Run: python train_word_classifier.py --epochs 20 --batch_size 32
"""
import os, sys, json, argparse, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from models.isl_conformer import WordCategorizerModel
from utils.checkpoint_utils import extract_state_dict, load_flexible
from utils.dataloader_word import (
    WORD_LEVEL_DIR_DEFAULT, MIN_SAMPLES_PER_CLASS, MAX_SAMPLES_PER_CLASS,
    discover_word_classes, split_word_classes, collate_word_batch
)


class WordDataset(Dataset):
    def __init__(self, items, max_len=400):
        self.items = items
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        clip = torch.load(path, map_location="cpu", weights_only=True).float()
        if clip.ndim == 3:
            clip = clip[0]
        T = min(clip.shape[0], self.max_len)
        clip = clip[:T]
        return clip, label, T


def collate(batch):
    clips, labels, lengths = zip(*batch)
    max_T = max(lengths)
    B = len(clips)
    C = clips[0].shape[1]
    x = torch.zeros(B, max_T, C)
    for i, c in enumerate(clips):
        x[i, :c.shape[0]] = c
    return x, torch.tensor(lengths), torch.tensor(labels)


def build_pretrained_backbone(ckpt_path, num_classes, device):
    """Load Conformer-CTC backbone and convert to WordCategorizerModel."""
    from training.train_conformer_ctc import ISLHybridConformerCTC

    # Load Conformer-CTC model
    ctc_model = ISLHybridConformerCTC(num_classes).to(device)
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = extract_state_dict(raw)
    load_flexible(ctc_model, sd, label="CTC_BACKBONE")

    # Create WordCategorizerModel with same architecture
    word_model = WordCategorizerModel(num_classes).to(device)

    # Transfer compatible weights: input_stage, pos_enc, gru_layers, transformer, tcnn
    ctc_sd = ctc_model.state_dict()
    word_sd = word_model.state_dict()

    transferred = 0
    for k, v in ctc_sd.items():
        if k in word_sd and v.shape == word_sd[k].shape:
            word_sd[k].copy_(v)
            transferred += 1

    print(f"[INIT] Transferred {transferred} backbone tensors from Conformer-CTC")
    return word_model


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load vocabulary
    with open(args.classes) as f:
        classes = json.load(f)
    if isinstance(classes, dict):
        classes = classes["classes"]
    num_classes = len(classes)
    print(f"[INFO] {num_classes} classes")

    # Discover data
    infos = discover_word_classes(args.data_dir, args.min_samples, args.max_samples)
    print(f"[INFO] {len(infos)} classes, {sum(len(i.files) for i in infos)} clips")

    # Split
    train_items, val_items = split_word_classes(
        infos, val_fraction=0.2, seed=args.seed
    )
    print(f"[DATA] Train: {len(train_items)} | Val: {len(val_items)}")

    train_ds = WordDataset(train_items, args.max_len)
    val_ds = WordDataset(val_items, args.max_len)

    train_ld = DataLoader(train_ds, args.batch_size, shuffle=True,
                          collate_fn=collate, num_workers=args.num_workers)
    val_ld = DataLoader(val_ds, args.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=args.num_workers)

    # Model: initialize from Conformer-CTC backbone
    model = build_pretrained_backbone(args.ctc_ckpt, num_classes, device)

    # Freeze backbone initially, train only classifier
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-5
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    os.makedirs(args.out_dir, exist_ok=True)
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n{'='*60}")
    print(f"WORD CLASSIFIER TRAINING | {num_classes} classes")
    print(f"Device: {device} | Batch: {args.batch_size} | Epochs: {args.epochs}")
    print(f"Backbone: Conformer-CTC (frozen) | Head: fresh")
    print(f"{'='*60}\n")

    for ep in range(1, args.epochs + 1):
        # Unfreeze backbone after epoch 5
        if ep == 6:
            for p in model.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
            print("[UNFREEZE] Full model training")

        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_ld, desc=f"Ep {ep:02d}/{args.epochs} [train]", leave=False)
        for x, lengths, labels in pbar:
            x, lengths, labels = x.to(device), lengths.to(device), labels.to(device)
            optimizer.zero_grad()

            log_probs, out_lengths = model(x, lengths)
            # Average over time for classification
            logits = log_probs.mean(1)  # (B, C+1)
            logits = logits[:, :-1]  # Remove blank

            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            preds = logits.argmax(1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100*train_correct/train_total:.1f}%")

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, lengths, labels in val_ld:
                x, lengths, labels = x.to(device), lengths.to(device), labels.to(device)
                log_probs, _ = model(x, lengths)
                logits = log_probs.mean(1)[:, :-1]
                loss = criterion(logits, labels)
                val_loss += loss.item()
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        avg_train_loss = train_loss / len(train_ld)
        avg_val_loss = val_loss / len(val_ld)

        scheduler.step()

        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        print(f"Ep {ep:02d}: train_loss={avg_train_loss:.3f} train_acc={train_acc:.3f} | "
              f"val_loss={avg_val_loss:.3f} val_acc={val_acc:.3f} | "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_word_classifier.pth"))
            print(f"  ★ NEW BEST: {val_acc:.3f}")

        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nDone. Best val acc: {best_acc:.3f}")
    print(f"Model saved to: {args.out_dir}/best_word_classifier.pth")


def parse_args():
    p = argparse.ArgumentParser(description="Train word classifier from Conformer-CTC backbone")
    p.add_argument("--data_dir", default=WORD_LEVEL_DIR_DEFAULT)
    p.add_argument("--classes", default=str(BACKEND_DIR / "models" / "word_vocabulary" / "classes.json"))
    p.add_argument("--ctc_ckpt", default=str(BACKEND_DIR / "models" / "isl_conformer_ctc" / "stage3_ema_best.pth"))
    p.add_argument("--out_dir", default=str(BACKEND_DIR / "models" / "word_classifier_from_ctc"))
    p.add_argument("--min_samples", type=int, default=MIN_SAMPLES_PER_CLASS)
    p.add_argument("--max_samples", type=int, default=MAX_SAMPLES_PER_CLASS)
    p.add_argument("--max_len", type=int, default=400)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())