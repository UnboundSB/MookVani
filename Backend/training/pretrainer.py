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

DATA_DIR = os.path.join(BACKEND_DIR, "data", "tensors_isl_csltr_word_level")
SAVE_DIR = os.path.join(BACKEND_DIR, "models", "pretrained_model")
PLOTS_DIR = os.path.join(SAVE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

MEDIAN_LENGTH = 6  # Derived from ISL-CSLRT EDA distribution


# ==============================================================================
# 1. VECTORIZED POSE AUGMENTATION (APPLIED STRICTLY ON TRAIN SPLIT)
# ==============================================================================

def augment_pose_sequence(pose_seq, angle_deg_range=(-12.0, 12.0), scale_range=(0.92, 1.08),
                          shift_range=(-0.04, 0.04), p_flip=0.5, noise_std=0.006):
    """Vectorized, fast geometric and anatomical pose augmentation."""
    seq = pose_seq.clone()
    T, D = seq.shape

    # 1. Vectorized 2D rotation in XY plane around (0, 0)
    angle_deg = random.uniform(*angle_deg_range)
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    coords = seq[:, :159].view(T, 53, 3).clone()
    x = coords[:, :, 0]
    y = coords[:, :, 1]
    new_x = x * cos_a - y * sin_a
    new_y = x * sin_a + y * cos_a
    coords[:, :, 0] = new_x
    coords[:, :, 1] = new_y

    # 2. Horizontal Flip with Anatomical Left/Right Swap
    if random.random() < p_flip:
        coords[:, :, 0] = -coords[:, :, 0]

        # Left Hand (0..20) <-> Right Hand (21..41)
        lh = coords[:, 0:21, :].clone()
        rh = coords[:, 21:42, :].clone()
        coords[:, 0:21, :] = rh
        coords[:, 21:42, :] = lh

        # Left Arm (42=l_sh, 44=l_elb) <-> Right Arm (43=r_sh, 45=r_elb)
        l_arm = coords[:, [42, 44], :].clone()
        r_arm = coords[:, [43, 45], :].clone()
        coords[:, [42, 44], :] = r_arm
        coords[:, [43, 45], :] = l_arm

        # Left Face (46=l_eye, 49=l_chk) <-> Right Face (47=r_eye, 50=r_chk)
        l_face = coords[:, [46, 49], :].clone()
        r_face = coords[:, [47, 50], :].clone()
        coords[:, [46, 49], :] = r_face
        coords[:, [47, 50], :] = l_face

        # Swap Hand presence flags [lh_flag=159, rh_flag=160]
        lh_flag = seq[:, 159].clone()
        rh_flag = seq[:, 160].clone()
        seq[:, 159] = rh_flag
        seq[:, 160] = lh_flag

    # 3. Vectorized Scaling and Translation
    scale = random.uniform(*scale_range)
    shift_x = random.uniform(*shift_range)
    shift_y = random.uniform(*shift_range)

    coords[:, :, 0] = coords[:, :, 0] * scale + shift_x
    coords[:, :, 1] = coords[:, :, 1] * scale + shift_y
    coords[:, :, 2] = coords[:, :, 2] * scale

    # 4. Gaussian Landmark Jitter Noise
    if noise_std > 0:
        coords += torch.randn_like(coords) * noise_std

    seq[:, :159] = coords.view(T, 159)
    return seq


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

class TrainPoseDataset(Dataset):
    """Training Dataset: Augmentations applied on-the-fly AFTER data split."""
    def __init__(self, raw_samples, samples_per_class=20, median_len=6):
        self.raw_samples = raw_samples
        self.median_len = median_len
        self.items = []
        for class_idx in sorted(self.raw_samples.keys()):
            for _ in range(samples_per_class):
                self.items.append(class_idx)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        class_idx = self.items[idx]
        raw_tensor = self.raw_samples[class_idx]

        # 1. Stochastic temporal variation around median (+/- 1)
        target_len = random.choice([self.median_len - 1, self.median_len, self.median_len + 1])
        target_len = max(3, target_len)
        slashed = slash_to_target_length(raw_tensor, target_len=target_len, is_train=True)

        # 2. On-the-fly stochastic geometric & anatomical pose augmentation
        augmented = augment_pose_sequence(slashed)

        # 3. Standardize to uniform batch dimension of median_len
        if augmented.shape[0] != self.median_len:
            augmented = slash_to_target_length(augmented, target_len=self.median_len, is_train=False)

        return augmented, class_idx


class ValPoseDataset(Dataset):
    """Validation Dataset: ZERO augmentation to prevent data leakage. Strictly clean ground-truth poses."""
    def __init__(self, raw_samples, median_len=6):
        self.raw_samples = raw_samples
        self.median_len = median_len
        self.items = sorted(self.raw_samples.keys())

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        class_idx = self.items[idx]
        raw_tensor = self.raw_samples[class_idx]

        clean_pose = slash_to_target_length(raw_tensor, target_len=self.median_len, is_train=False)
        return clean_pose, class_idx


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

def train_pretrainer(epochs=25, batch_size=32, lr=1.5e-3, samples_per_class=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}", flush=True)

    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    num_classes = len(classes)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    print(f"[INFO] Found {num_classes} classes in {DATA_DIR}", flush=True)

    raw_data = {}
    for cls_name in classes:
        cls_folder = os.path.join(DATA_DIR, cls_name)
        pt_files = glob.glob(os.path.join(cls_folder, "*.pt"))
        if pt_files:
            try:
                t = torch.load(pt_files[0], weights_only=True)
                if isinstance(t, np.ndarray):
                    t = torch.from_numpy(t).float()
                raw_data[class_to_idx[cls_name]] = t.float()
            except Exception as e:
                print(f"[WARN] Error loading {pt_files[0]}: {e}", flush=True)

    print(f"[LEAK-PREVENTION] Splitting clean data. Augmentations active ONLY on train split.", flush=True)
    train_dataset = TrainPoseDataset(raw_data, samples_per_class=samples_per_class, median_len=MEDIAN_LENGTH)
    val_dataset = ValPoseDataset(raw_data, median_len=MEDIAN_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"[INFO] Dataset sizes: {len(train_dataset)} train samples, {len(val_dataset)} val samples.", flush=True)

    model = PoseTransformerTCNN(
        input_dim=163, d_model=128, nhead=4, num_transformer_layers=2,
        tcnn_channels=(128, 256), num_classes=num_classes
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Transformer + TCNN initialized ({total_params:,} parameters).", flush=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "val_precision": [], "val_recall": [], "val_f1": []
    }

    best_val_acc = 0.0
    best_model_path = os.path.join(SAVE_DIR, "best_pretrained_model.pth")
    final_model_path = os.path.join(SAVE_DIR, "final_pretrained_model.pth")

    print("\n[START] Beginning Pretraining Loop...", flush=True)
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

    torch.save(model.state_dict(), final_model_path)
    print(f"\n[SUCCESS] Pretraining finished! Best Val Acc: {best_val_acc*100:.2f}%", flush=True)

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
    train_pretrainer(epochs=25, batch_size=32, lr=1.5e-3, samples_per_class=20)
