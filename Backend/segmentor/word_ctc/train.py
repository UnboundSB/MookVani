"""
Training loop for word-level CTC.

Saves best checkpoint to Backend/models/word_ctc/best_model.pth
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from segmentor.word_ctc.dataset import WordCTCDataset, collate_fn
from segmentor.word_ctc.model import ConformerCTC
from segmentor.word_ctc.loss import AntiCollapseCTCLoss


def _make_ctc_targets(labels: torch.Tensor, lengths: torch.Tensor, device):
    """Build CTC targets: repeat word label across frames separated by blanks.

    Returns (targets_flat_1d, target_lengths_1d) for nn.CTCLoss.
    """
    target_flat = []
    target_lens = []
    for i in range(len(labels)):
        t = int(lengths[i].item())
        n_emit = max(1, int(0.25 * t))  # word on ~25% of valid frames
        seq = []
        for k in range(n_emit):
            seq.append(int(labels[i].item()))
            if k < n_emit - 1:
                seq.append(0)  # blank between repeats
        target_flat.extend(seq)
        target_lens.append(len(seq))
    targets_ctc = torch.tensor(target_flat, dtype=torch.long, device=device)
    target_lengths = torch.tensor(target_lens, dtype=torch.long, device=device)
    return targets_ctc, target_lengths


def train_word_ctc(epochs: int = 50, batch_size: int = 8, lr: float = 3e-4,
                   max_per_class: int = 20, patience: int = 8,
                   device: str | None = None, out_dir: str | None = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if out_dir is None:
        out_dir = BACKEND_DIR / "models" / "word_ctc"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = WordCTCDataset(split="train", max_per_class=max_per_class)
    val_ds = WordCTCDataset(split="val", max_per_class=max_per_class)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_fn, pin_memory=True)

    model = ConformerCTC(num_words=train_ds.num_classes).to(device)
    criterion = AntiCollapseCTCLoss(blank=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=4, factor=0.5, min_lr=1e-6, verbose=True
    )

    best_val = float("inf")
    patience_counter = 0
    history = {"train": [], "val": []}

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        train_losses = []
        for x, lengths, labels in train_loader:
            x = x.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            targets_ctc, target_lengths = _make_ctc_targets(labels, lengths, device)

            log_probs = model(x, lengths)  # (B, T, C)
            loss, info = criterion(log_probs, targets_ctc, lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(info["ctc"])

        avg_train = float(sum(train_losses) / max(len(train_losses), 1))

        # ---- val ----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, lengths, labels in val_loader:
                x = x.to(device)
                lengths = lengths.to(device)
                labels = labels.to(device)
                targets_ctc, target_lengths = _make_ctc_targets(labels, lengths, device)
                log_probs = model(x, lengths)
                loss, info = criterion(log_probs, targets_ctc, lengths, target_lengths)
                val_losses.append(info["ctc"])
        avg_val = float(sum(val_losses) / max(len(val_losses), 1))

        scheduler.step(avg_val)
        history["train"].append(avg_train)
        history["val"].append(avg_val)

        print(f"Epoch {epoch:02d}: train_ctc={avg_train:.4f} val_ctc={avg_val:.4f} "
              f"(blank={info['mean_blank']:.3f} nonblank={info['mean_nonblank']:.3f})")

        # ---- checkpoint ----
        if avg_val < best_val:
            best_val = avg_val
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val,
                "classes": train_ds.idx_to_class,
            }, out_dir / "best_model.pth")
            print(f"  -> new best saved ({avg_val:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"[DONE] Best val CTC: {best_val:.4f}")
    return best_val, history
