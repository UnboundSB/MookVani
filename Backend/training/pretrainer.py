"""Pretraining script for Indian Sign Language (ISL) MediaPipe Pose sequences.

Leak-Free Data Pipeline:
  1. Train/Validation Split occurs FIRST on raw unaugmented poses.
  2. Validation Set: Evaluated ONLY on clean, unaugmented pose sequences normalized to median length.
  3. Augmentation: Applied dynamically on-the-fly ONLY during training iterations AFTER split,
     preventing any data leakage between train and validation distributions.

Architecture:
  Input (B, T=6, 163) -> Pose Projection & Positional Embedding
                      -> Transformer Encoder (temporal multi-head self-attention)
                      -> Temporal CNN (TCNN) with Residual Highway
                      -> Dual Global Pooling (Avg + Max)
                      -> Classifier Head -> Logits (B, 114)
"""

import os
import sys
import glob
import json
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

# Windows console encoding configuration
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from utils.dataloader_word import (  # noqa: E402
    MAX_SAMPLES_PER_CLASS,
    MIN_SAMPLES_PER_CLASS,
    WORD_LEVEL_DIR_DEFAULT,
    discover_word_classes,
    split_word_classes,
)
from utils.pose_augmentation import (  # noqa: E402
    AugmentConfig,
    augment_pose_sequence as canonical_augment,
    as_sequence,
    fill_missing_temporal,
)
from utils.training_utils import (  # noqa: E402
    build_warm_restart_scheduler,
    escape_from_plateau,
)

# The pretraining corpus.  It used to be `tensors_isl_csltr_word_level`, which
# holds ONE clip per class: the "validation" split was therefore the training
# data, and the reported 98% validation accuracy was pure memorisation.  The
# pretrainer now runs on the word corpus with a real file-level split.
DATA_DIR = WORD_LEVEL_DIR_DEFAULT
SAVE_DIR = os.path.join(BACKEND_DIR, "models", "pretrained_model")
PLOTS_DIR = os.path.join(SAVE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

MEDIAN_LENGTH = 62   # median frames of the word corpus (matches the Phase-2 curriculum)
MAX_SEQ_LEN = 256    # the learned positional embedding has to cover the longest clip
RESTART_EVERY = 10   # cosine warm restart every N epochs
ESCAPE_PATIENCE = 8  # epochs without a val improvement -> escape the plateau
ESCAPE_SCALE = 0.02

# The canonical augmentation set (Backend/utils/pose_augmentation.py).
PRETRAIN_AUGMENT = AugmentConfig(
    flip_prob=0.5,
    tilt_prob=0.6,
    tilt_deg=12.0,
    shift_prob=0.5,
    shift_dx=0.04,
    shift_dy=0.02,
    scale_prob=0.5,
    scale_lo=0.92,
    scale_hi=1.08,
    speed_prob=0.5,
    speed_lo=0.85,
    speed_hi=1.15,
    time_shift_prob=0.2,
    time_shift_max=3,
    mask_prob=0.1,
    mask_max=3,
    noise_std=0.006,
)


# ==============================================================================
# 1. POSE AUGMENTATION (APPLIED STRICTLY ON TRAIN SPLIT)
# ==============================================================================
# The augmentation itself is the canonical one (Backend/utils/pose_augmentation.py):
# the mirror swaps every left/right pair around the signer's own axis (the mouth
# corners included), the tilt rotates around the neck and the shift is bounded.
# The old local copy rotated around the ORIGIN (the corpus is anchor relative,
# so the body sits at x ~ +3.5 shoulder widths) and never swapped the mouth.
def augment_pose_sequence(pose_seq, config=PRETRAIN_AUGMENT):
    """Geometric + temporal augmentation of one clip."""
    return canonical_augment(pose_seq, config)


def slash_to_target_length(pose_seq, target_len=6, is_train=False):
    """Slashes / normalizes a pose sequence to target_len (median length)."""
    T, D = pose_seq.shape
    if T == target_len:
        return pose_seq

    if T > target_len:
        if is_train and T > target_len + 2:
            start = random.randint(0, T - target_len)
            return pose_seq[start:start + target_len]
        else:
            indices = np.linspace(0, T - 1, target_len).round().astype(int)
            return pose_seq[indices]
    else:
        t_seq = pose_seq.T.unsqueeze(0)  # (1, 163, T)
        interpolated = F.interpolate(t_seq, size=target_len, mode='linear', align_corners=True)
        return interpolated.squeeze(0).T  # (target_len, 163)


# ==============================================================================
# 2. LEAK-FREE DATASET CLASSES (SPLIT FIRST, AUGMENT AFTER SPLIT)
# ==============================================================================

class PretrainPoseDataset(Dataset):
    """Pretraining dataset over the word clips of a REAL train/val split.

    The previous version loaded ONE clip per class and used it for both the
    train and the validation split, so the reported validation accuracy was the
    accuracy on data the model had already memorised (98% on 114 clips).  The
    split is now done on the FILE LIST through the shared selection rule of
    ``utils/dataloader_word.py``, so no clip is ever seen twice.

    Every clip is resampled to a fixed ``median_len`` frames, which keeps the
    batch rectangular without a padding mask (the model pools over time).
    """

    def __init__(self, items, is_train=True, median_len=MEDIAN_LENGTH,
                 samples_per_class=4, fill_missing="none"):
        self.items = list(items)
        self.is_train = is_train
        self.median_len = median_len
        self.samples_per_class = max(1, samples_per_class)
        self.fill_missing = fill_missing
        self.augment_config = PRETRAIN_AUGMENT if is_train else None

    def __len__(self):
        return len(self.items) * self.samples_per_class

    def __getitem__(self, idx):
        path, label = self.items[idx % len(self.items)]
        try:
            x = torch.load(path, weights_only=True)
        except Exception as exc:  # pragma: no cover
            print(f"  [WARN] could not load {path}: {exc}")
            x = torch.zeros(self.median_len, 163)
        x = x.float() if torch.is_tensor(x) else torch.from_numpy(x).float()
        x = as_sequence(x)

        if self.fill_missing and self.fill_missing != "none":
            x = fill_missing_temporal(x, mode=self.fill_missing)
        if self.augment_config is not None:
            x = augment_pose_sequence(x, self.augment_config)

        # Fixed length: the model pools over time, so a uniform length avoids
        # the padding mask entirely.
        x = slash_to_target_length(x, target_len=self.median_len, is_train=self.is_train)
        return x, label


# ==============================================================================
# 3. MODEL ARCHITECTURE: TRANSFORMER + TCNN
# ==============================================================================

class TemporalCNNBlock(nn.Module):
    """1D Temporal Convolutional Block with Residual Highway."""
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        out = self.drop1(self.act1(self.bn1(self.conv1(x))))
        out = self.drop2(self.act2(self.bn2(self.conv2(out))))
        return out + res


class PoseTransformerTCNN(nn.Module):
    """Data enters Transformer -> TCNN predicts class of the pose."""
    def __init__(self, input_dim=163, d_model=128, nhead=4, num_transformer_layers=2,
                 tcnn_channels=(128, 256), num_classes=114, max_seq_len=16, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers, enable_nested_tensor=False)
        self.transformer_norm = nn.LayerNorm(d_model)

        tcnn_layers = []
        curr_ch = d_model
        for out_ch in tcnn_channels:
            tcnn_layers.append(TemporalCNNBlock(curr_ch, out_ch, kernel_size=3, dropout=dropout))
            curr_ch = out_ch
        self.tcnn = nn.Sequential(*tcnn_layers)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        pooled_dim = curr_ch * 2

        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout * 2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        B, T, D = x.shape

        h = self.input_proj(x)
        h = h + self.pos_embedding[:, :T, :]

        h = self.transformer(h)
        h = self.transformer_norm(h)  # (B, T, d_model)

        h_tcnn = h.transpose(1, 2)  # (B, d_model, T)
        h_tcnn = self.tcnn(h_tcnn)  # (B, out_ch, T)

        avg_p = self.avg_pool(h_tcnn).squeeze(-1)
        max_p = self.max_pool(h_tcnn).squeeze(-1)
        pooled = torch.cat([avg_p, max_p], dim=-1)

        logits = self.classifier(pooled)  # (B, num_classes)
        return logits


# ==============================================================================
# 4. TRAINING & EVALUATION PIPELINE
# ==============================================================================

def train_pretrainer(epochs=25, batch_size=32, lr=1.5e-3, samples_per_class=4,
                     data_dir=DATA_DIR, val_fraction=0.2, fill_missing="none"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}", flush=True)

    # The vocabulary and the split come from the SHARED selection rule, so the
    # pretrainer, the word phases and the sentence phases all speak the same
    # class list, and the train/val split is done on the FILE LIST (no clip is
    # ever in both partitions).
    infos = discover_word_classes(data_dir, min_samples_per_class=MIN_SAMPLES,
                                  max_samples_per_class=MAX_SAMPLES)
    classes = [i.name for i in infos]
    num_classes = len(classes)
    print(f"[INFO] Found {num_classes} classes in {data_dir}", flush=True)

    train_items, val_items = split_word_classes(infos, val_fraction=val_fraction, seed=42)
    print(f"[LEAK-PREVENTION] File-level split: {len(train_items)} train clips / "
          f"{len(val_items)} val clips, no overlap.", flush=True)

    train_dataset = PretrainPoseDataset(train_items, is_train=True, median_len=MEDIAN_LENGTH,
                                         samples_per_class=samples_per_class,
                                         fill_missing=fill_missing)
    val_dataset = PretrainPoseDataset(val_items, is_train=False, median_len=MEDIAN_LENGTH,
                                       samples_per_class=1, fill_missing=fill_missing)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"[INFO] Dataset sizes: {len(train_dataset)} train samples, {len(val_dataset)} val samples.", flush=True)

    model = PoseTransformerTCNN(
        input_dim=163, d_model=128, nhead=4, num_transformer_layers=2,
        tcnn_channels=(128, 256), num_classes=num_classes, max_seq_len=MAX_SEQ_LEN
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Transformer + TCNN initialized ({total_params:,} parameters).", flush=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for g in optimizer.param_groups:
        g["base_lr"] = g["lr"]
    # Cosine WITH WARM RESTARTS so a plateau can be escaped instead of the LR
    # annealing to zero for the rest of the run.
    scheduler, restart_every = build_warm_restart_scheduler(
        optimizer, total_epochs=epochs, restart_every=RESTART_EVERY, eta_min=1e-5)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "val_precision": [], "val_recall": [], "val_f1": []
    }

    best_val_acc = 0.0
    epochs_without_improvement = 0
    escapes = 0
    best_model_path = os.path.join(SAVE_DIR, "best_pretrained_model.pth")
    final_model_path = os.path.join(SAVE_DIR, "final_pretrained_model.pth")

    print(f"\n[START] Beginning Pretraining Loop ({epochs} epochs, warm restart every "
          f"{restart_every})...", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for poses, labels in train_loader:
            poses, labels = poses.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(poses)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += loss.item() * poses.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += poses.size(0)

        scheduler.step()

        train_loss = total_loss / total
        train_acc = correct / total

        # Clean validation pass
        model.eval()
        val_loss_sum = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for poses, labels in val_loader:
                poses, labels = poses.to(device), labels.to(device)
                logits = model(poses)
                loss = criterion(logits, labels)

                val_loss_sum += loss.item() * poses.size(0)
                preds = logits.argmax(dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_loss = val_loss_sum / len(val_dataset)
        val_acc = accuracy_score(val_targets, val_preds)
        val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(val_targets, val_preds, average='weighted', zero_division=0)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_precision"].append(val_prec)
        history["val_recall"].append(val_rec)
        history["val_f1"].append(val_f1)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}% F1: {val_f1:.4f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # --- plateau escape -------------------------------------------------
        if ESCAPE_PATIENCE > 0 and epochs_without_improvement >= ESCAPE_PATIENCE:
            escapes += 1
            print(f"  [ESCAPE] No val improvement for {epochs_without_improvement} epochs "
                  f"(val acc {val_acc*100:.2f}%) - kicking the model out of the plateau.", flush=True)
            print(f"  [ESCAPE] {escape_from_plateau(model, optimizer, scale=ESCAPE_SCALE)}", flush=True)
            epochs_without_improvement = 0

    torch.save(model.state_dict(), final_model_path)
    print(f"\n[SUCCESS] Pretraining finished! Best Val Acc: {best_val_acc*100:.2f}% "
          f"| escapes: {escapes}", flush=True)

    with open(os.path.join(SAVE_DIR, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)

    with open(os.path.join(SAVE_DIR, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Classification Report
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()
    final_preds, final_targets = [], []
    with torch.no_grad():
        for poses, labels in val_loader:
            poses, labels = poses.to(device), labels.to(device)
            preds = model(poses).argmax(dim=-1)
            final_preds.extend(preds.cpu().numpy())
            final_targets.extend(labels.cpu().numpy())

    report = classification_report(final_targets, final_preds, target_names=classes, zero_division=0)
    with open(os.path.join(SAVE_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    # Generate Evaluation Visualizations
    generate_evaluation_plots(history, final_targets, final_preds, classes)


def generate_evaluation_plots(history, targets, preds, classes):
    """Generates Accuracy vs Loss, Confusion Matrix, and Precision/Recall/F1 curves."""
    print("[INFO] Generating training plots and confusion matrix...", flush=True)

    # 1. Accuracy vs Loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#e53e3e", lw=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss (Clean Unaugmented)", color="#dd6b20", lw=2, linestyle="--")
    ax1.set_title("Training & Validation Loss", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=11)

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Acc (%)", color="#3182ce", lw=2)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], label="Val Acc (%) (Clean Unaugmented)", color="#38a169", lw=2, linestyle="--")
    ax2.set_title("Training & Validation Accuracy", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    loss_acc_path = os.path.join(PLOTS_DIR, "acc_vs_loss.png")
    plt.savefig(loss_acc_path, dpi=300)
    plt.close()
    print(f"  [SAVED] {loss_acc_path}", flush=True)

    # 2. Precision, Recall, F1 Metrics
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, history["val_f1"], label="Weighted F1-Score", color="#805ad5", lw=2.5)
    ax.plot(epochs, history["val_precision"], label="Precision", color="#319795", lw=2, linestyle=":")
    ax.plot(epochs, history["val_recall"], label="Recall", color="#d69e2e", lw=2, linestyle="--")
    ax.set_title("Validation Metrics Progression (Leak-Free Out-of-Sample)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=11)

    plt.tight_layout()
    metrics_path = os.path.join(PLOTS_DIR, "metrics.png")
    plt.savefig(metrics_path, dpi=300)
    plt.close()
    print(f"  [SAVED] {metrics_path}", flush=True)

    # 3. Confusion Matrix
    cm = confusion_matrix(targets, preds)
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(xticks=np.arange(len(classes)),
           yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes,
           title=f"Pretrained Model Confusion Matrix ({len(classes)} Classes)",
           ylabel="True Label",
           xlabel="Predicted Label")

    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor", fontsize=6)
    plt.setp(ax.get_yticklabels(), fontsize=6)

    plt.tight_layout()
    cm_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"  [SAVED] {cm_path}", flush=True)


if __name__ == "__main__":
    train_pretrainer(epochs=25, batch_size=32, lr=1.5e-3, samples_per_class=4)
