import os
import sys
import math
import json
import shutil
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# Ensure backend root is on sys.path
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from models.isl_sentence_model import ISLSentenceReconformer
from utils.dataloader_word import get_word_dataloaders
from utils.pose_augmentation import AugmentConfig

class FocalLoss(nn.Module):
    """
    Focal Loss with label smoothing on pooled log-probabilities.
    Downweights easy examples and forces the model to focus on hard gesture classes.
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', label_smoothing=0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, log_probs, targets):
        # log_probs: (B, C)
        n_classes = log_probs.size(-1)
        with torch.no_grad():
            smooth_target = torch.full_like(log_probs, self.label_smoothing / (n_classes - 1))
            smooth_target.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        
        ce_loss = -(smooth_target * log_probs).sum(dim=-1)
        pt = torch.exp(log_probs.gather(1, targets.unsqueeze(1)).squeeze(1))
        focal_loss = self.alpha * ((1.0 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def generate_plots(history, save_dir, artifact_dir=None):
    """Generates 3-panel metrics curves (Loss, Accuracy, and F1/Precision/Recall)."""
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=200)
    
    # 1. Loss Plot
    axes[0].plot(epochs_range, history['train_loss'], label='Train Loss', color='#1E88E5', lw=2)
    axes[0].plot(epochs_range, history['val_loss'], label='Val Loss', color='#E53935', lw=2)
    axes[0].set_title("Loss over Epochs (Focal Loss)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].legend(loc='upper right', frameon=True)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # 2. Accuracy Plot
    axes[1].plot(epochs_range, history['train_acc'], label='Train Acc', color='#1E88E5', lw=2)
    axes[1].plot(epochs_range, history['val_acc'], label='Val Acc', color='#43A047', lw=2)
    axes[1].set_title("Accuracy over Epochs", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy", fontsize=11)
    axes[1].legend(loc='lower right', frameon=True)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    # 3. F1, Precision, Recall Plot
    axes[2].plot(epochs_range, history['val_f1'], label='Val Macro F1', color='#8E24AA', lw=2)
    axes[2].plot(epochs_range, history['val_prec'], label='Val Precision', color='#00897B', linestyle='--', lw=1.8)
    axes[2].plot(epochs_range, history['val_rec'], label='Val Recall', color='#FB8C00', linestyle='-.', lw=1.8)
    axes[2].set_title("Validation F1, Precision & Recall", fontsize=13, fontweight='bold')
    axes[2].set_xlabel("Epoch", fontsize=11)
    axes[2].set_ylabel("Score", fontsize=11)
    axes[2].legend(loc='lower right', frameon=True)
    axes[2].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "reconformer_metrics.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Metrics plot saved to: {plot_path}")

    if artifact_dir and os.path.exists(artifact_dir):
        try:
            shutil.copy2(plot_path, os.path.join(artifact_dir, "reconformer_metrics.png"))
        except Exception as e:
            print(f"Warning: could not copy metrics plot to artifact dir: {e}")

def generate_confusion_matrix_plot(y_true, y_pred, classes, save_dir, artifact_dir=None):
    """
    Generates high-resolution normalized confusion matrix with FULL row and column labels
    for all classes, ensuring a distinct diagonal is clearly visible and every class name is readable.
    Saved in the model's dedicated directory.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    # Row-normalized recall matrix (each row sums to 1.0)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    plt.figure(figsize=(48, 46), dpi=150)
    sns.heatmap(
        cm_norm,
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        annot=False,
        cbar_kws={'label': 'Row-Normalized Recall (True Class Accuracy)', 'shrink': 0.8}
    )
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.title(f"Reconformer Word Classification: Confusion Matrix ({len(classes)} Balanced Classes + [NOTHING])", fontsize=22, fontweight='bold', pad=20)
    plt.xlabel("Predicted Word Class", fontsize=18, labelpad=15)
    plt.ylabel("True Word Class", fontsize=18, labelpad=15)
    plt.tight_layout()

    cm_path = os.path.join(save_dir, "reconformer_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix plot saved to: {cm_path}")

    # Copy to Workspace Root (for instant review) and Artifact directory
    destinations = [
        os.path.join(BACKEND_DIR, "..", "confusion_matrix.png"),
        os.path.join(BACKEND_DIR, "confusion_matrix.png"),
    ]
    if artifact_dir and os.path.exists(artifact_dir):
        destinations.append(os.path.join(artifact_dir, "reconformer_confusion_matrix.png"))

    for dst in destinations:
        try:
            shutil.copy2(cm_path, dst)
            print(f"Copied confusion matrix to: {dst}")
        except Exception as e:
            print(f"Warning copying confusion matrix to {dst}: {e}")

def train_reconformer_word(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 75)
    print("MOOKVANI ISL: RECONFORMER WORD RECOGNITION TRAINING")
    print("=" * 75)
    print(f"Device:               {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Data Directory:       {args.data_dir}")
    print(f"Pretrained Weights:   {args.pretrained_weights}")
    print(f"Epochs:               {args.epochs}")
    print(f"Batch Size:           {args.batch_size}")
    print(f"Learning Rate:        {args.lr}")
    print(f"D_Model:              {args.d_model}")
    print(f"Samples per Class:    <= {args.max_samples_per_class} (min {args.min_samples_per_class})")
    print("=" * 75 + "\n")

    os.makedirs(args.save_dir, exist_ok=True)
    model_save_path = os.path.join(args.save_dir, "reconformer_word_model.pth")
    meta_save_path = os.path.join(args.save_dir, "classes_balanced.json")

    # 1. Initialize the word DataLoader: the TOP-MOST FREQUENT classes of the
    #    word corpus, at most 15 clips per class, and a class that owns fewer
    #    than 14 clips is eliminated from the loader (its files stay on disk,
    #    the pretrainer and the other models still use them).
    train_loader, val_loader, class_to_idx, idx_to_class, classes = get_word_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        min_samples_per_class=args.min_samples_per_class,
        max_samples_per_class=args.max_samples_per_class,
        val_fraction=args.val_fraction,
        max_len=args.max_len,
        include_nothing_class=True,
        augment_config=AugmentConfig(),
        fill_missing=args.fill_missing,
        seed=42,
        save_meta_path=meta_save_path
    )

    all_class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    num_classes = len(classes)  # gesture classes count; the head outputs num_classes + 1 for [NOTHING]

    # 2. Build Model: RNN -> Transformer -> TCNN
    print(f"\nBuilding FRESH ISLSentenceReconformer (Gesture Classes: {num_classes}, Total Classes with [NOTHING]: {num_classes + 1})...")
    model = ISLSentenceReconformer(
        input_dim=163,
        num_classes=num_classes,
        d_model=args.d_model,
        lstm_layers=1,
        trans_layers=4,
        num_heads=args.num_heads,
        dropout=args.dropout,
        use_rnn=True, # Data enters RNN -> Transformer -> TCNN
        freeze_base=False
    )

    # 3. Model Initialization (Fresh from scratch as requested)
    if args.fresh or not args.pretrained_weights or not os.path.exists(args.pretrained_weights):
        print("Initializing FRESH ISLSentenceReconformer from scratch with clean random weights! (No pretrained weights)")
    else:
        print(f"Loading pretrained weights from: {args.pretrained_weights}")
        model.load_base_weights(args.pretrained_weights)

    model = model.to(device)

    # 4. Criterion, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_f1': [], 'val_prec': [], 'val_rec': []
    }

    best_val_f1 = 0.0
    best_val_acc = 0.0
    unfreeze_epoch = 8 if args.freeze_base else -1

    for epoch in range(args.epochs):
        # Dynamic Unfreezing for End-to-End Fine-Tuning
        if epoch == unfreeze_epoch:
            print("\n>>> UNFREEZING ALL BACKBONE LAYERS FOR FULL END-TO-END FINE-TUNING! <<<")
            for param in model.parameters():
                param.requires_grad = True

        # --- Training Loop ---
        model.train()
        train_loss = 0.0
        all_preds_train = []
        all_targets_train = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{args.epochs:02d} [Train]")
        for x, lengths, labels in pbar:
            x, lengths, labels = x.to(device), lengths.to(device), labels.to(device)
            optimizer.zero_grad()

            # Robust masked temporal logit pooling
            logits_pooled = model.forward_word_logits(x, lengths)
            loss = criterion(logits_pooled, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()
            preds = logits_pooled.argmax(dim=-1)
            all_preds_train.extend(preds.cpu().numpy())
            all_targets_train.extend(labels.cpu().numpy())

        train_loss /= len(train_loader)
        train_acc = accuracy_score(all_targets_train, all_preds_train)

        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        all_preds_val = []
        all_targets_val = []

        with torch.no_grad():
            for x, lengths, labels in val_loader:
                x, lengths, labels = x.to(device), lengths.to(device), labels.to(device)
                logits_pooled = model.forward_word_logits(x, lengths)
                loss = criterion(logits_pooled, labels)

                val_loss += loss.item()
                preds = logits_pooled.argmax(dim=-1)
                all_preds_val.extend(preds.cpu().numpy())
                all_targets_val.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = accuracy_score(all_targets_val, all_preds_val)
        val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(
            all_targets_val, all_preds_val, average='macro', zero_division=0
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_prec'].append(val_prec)
        history['val_rec'].append(val_rec)

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            best_flag = " [*BEST SAVED*]"
        else:
            best_flag = ""

        print(
            f"Epoch {epoch+1:02d}: Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%, F1: {val_f1:.4f} | "
            f"LR: {current_lr:.6f}{best_flag}"
        )

    print("\n" + "=" * 75)
    print("TRAINING FINISHED!")
    print(f"Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print(f"Best Validation Macro F1: {best_val_f1:.4f}")
    print(f"Model saved to:           {model_save_path}")
    print("=" * 75 + "\n")

    # 5. Final Evaluation on Best Model Checkpoint
    print("Running final evaluation on best saved checkpoint...")
    model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
    model.eval()

    final_preds, final_targets = [], []
    with torch.no_grad():
        for x, lengths, labels in val_loader:
            x, lengths = x.to(device), lengths.to(device)
            logits_pooled = model.forward_word_logits(x, lengths)
            preds = logits_pooled.argmax(dim=-1)
            final_preds.extend(preds.cpu().numpy())
            final_targets.extend(labels.numpy())

    # Generate Metrics Plots
    artifact_dir = r"C:\Users\UnboundSB\.gemini\antigravity-ide\brain\e79d288d-e609-482f-b54e-82f0b240e570"
    generate_plots(history, args.save_dir, artifact_dir=artifact_dir)

    # Generate Confusion Matrix
    generate_confusion_matrix_plot(final_targets, final_preds, all_class_names, args.save_dir, artifact_dir=artifact_dir)

    # Generate Classification Report
    report_text = classification_report(
        final_targets, final_preds,
        labels=range(len(all_class_names)),
        target_names=all_class_names,
        zero_division=0
    )
    report_path = os.path.join(args.save_dir, "reconformer_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"Classification report saved to: {report_path}")

    # Copy report to artifact dir
    if os.path.exists(artifact_dir):
        shutil.copy2(report_path, os.path.join(artifact_dir, "reconformer_classification_report.txt"))

    # Save training history JSON
    history_path = os.path.join(args.save_dir, "history_word.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Reconformer Word Recognition Model on Balanced INCLUDE50 Classes")
    parser.add_argument("--data_dir", type=str, default=r"d:\MookVani\Backend\data\tensors_include50_word_level")
    parser.add_argument("--save_dir", type=str, default=r"d:\MookVani\Backend\models\word_reconformer_229")
    parser.add_argument("--pretrained_weights", type=str, default=r"d:\MookVani\Backend\models\word_reconformer_264\reconformer_word_model_final.pth", help="Path to pretrained weights")
    parser.add_argument("--fresh", action="store_true", help="Train a fresh model from scratch without loading pretrained weights")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument("--d_model", type=int, default=128, help="Model hidden dimension (default: 128)")
    parser.add_argument("--num_heads", type=int, default=4, help="Transformer attention heads (default: 4)")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability (default: 0.2)")
    parser.add_argument("--min_samples_per_class", type=int, default=14, help="A class with fewer clips than this is eliminated from the loader (default: 14)")
    parser.add_argument("--max_samples_per_class", type=int, default=20, help="Hard budget of clips loaded per class (default: 20)")
    parser.add_argument("--val_fraction", type=float, default=0.2, help="Fraction of the clips kept for validation (default: 0.2)")
    parser.add_argument("--max_len", type=int, default=300, help="Maximum number of frames per clip (default: 300)")
    parser.add_argument("--fill_missing", type=str, default="none", choices=["none", "hold", "interpolate"], help="Fill the drop-out holes of a clip with the last valid value")
    parser.add_argument("--freeze_base", action="store_true", help="Freeze pretrained base for initial warmup epochs")
    args = parser.parse_args()

    train_reconformer_word(args)
