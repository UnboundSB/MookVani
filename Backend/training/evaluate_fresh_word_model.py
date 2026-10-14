"""
Evaluation and Visualization Script for Trained Reconformer Word Model
Generates:
1. Training curves (Loss, Accuracy, Macro F1)
2. 229x229 Normalized Confusion Matrix
3. Detailed Classification Report
"""
import os
import sys
import json
import shutil
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score

# Ensure Backend in path
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from models.isl_sentence_model import ISLSentenceReconformer
from scripts.dataset import get_balanced_word_dataloaders

def evaluate_and_plot():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running evaluation on {device}...")

    save_dir = os.path.join(BACKEND_DIR, "models", "word_level")
    classes_json = os.path.join(save_dir, "classes_balanced.json")
    with open(classes_json, "r") as f:
        meta = json.load(f)
    classes = meta["classes"]
    num_classes = len(classes)
    print(f"Total vocabulary classes: {num_classes}")

    # History from 15 epoch training
    history = {
        'train_loss': [5.3956, 5.3675, 5.2947, 5.1726, 5.0512, 4.9483, 4.8508, 4.7955, 4.7154, 4.6268, 4.5668, 4.5064, 4.4840, 4.4559, 4.4591],
        'train_acc':  [0.0079, 0.0107, 0.0115, 0.0147, 0.0179, 0.0191, 0.0206, 0.0310, 0.0310, 0.0413, 0.0488, 0.0484, 0.0524, 0.0524, 0.0480],
        'val_loss':   [5.3443, 5.3532, 5.2192, 5.1331, 4.9851, 5.0299, 5.1804, 4.8079, 4.7486, 4.7098, 4.6923, 4.6401, 4.6025, 4.5944, 4.5897],
        'val_acc':    [0.0102, 0.0073, 0.0189, 0.0102, 0.0146, 0.0291, 0.0146, 0.0262, 0.0349, 0.0335, 0.0393, 0.0291, 0.0320, 0.0349, 0.0393],
        'val_f1':     [0.0046, 0.0046, 0.0049, 0.0052, 0.0038, 0.0088, 0.0054, 0.0080, 0.0130, 0.0143, 0.0162, 0.0099, 0.0165, 0.0108, 0.0161],
    }

    # Save history json
    history_path = os.path.join(save_dir, "history_fresh_word.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # 1. Plot Metrics Curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs_range = range(1, 16)
    axes[0].plot(epochs_range, history['train_loss'], label='Train Loss', color='#1E88E5', lw=2)
    axes[0].plot(epochs_range, history['val_loss'], label='Val Loss', color='#E53935', lw=2)
    axes[0].set_title("Loss over Epochs (Focal Loss)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].legend(loc='upper right', frameon=True)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].plot(epochs_range, [x * 100 for x in history['train_acc']], label='Train Acc (%)', color='#1E88E5', lw=2)
    axes[1].plot(epochs_range, [x * 100 for x in history['val_acc']], label='Val Acc (%)', color='#43A047', lw=2)
    axes[1].axhline(y=100.0/num_classes, color='gray', linestyle=':', label=f'Random Baseline ({100/num_classes:.2f}%)')
    axes[1].set_title("Accuracy over Epochs", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1].legend(loc='lower right', frameon=True)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    axes[2].plot(epochs_range, history['val_f1'], label='Val Macro F1', color='#8E24AA', lw=2)
    axes[2].set_title("Validation Macro F1 Score", fontsize=13, fontweight='bold')
    axes[2].set_xlabel("Epoch", fontsize=11)
    axes[2].set_ylabel("Macro F1", fontsize=11)
    axes[2].legend(loc='lower right', frameon=True)
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    metrics_path = os.path.join(save_dir, "reconformer_metrics.png")
    plt.savefig(metrics_path, dpi=200)
    plt.close()
    print(f"Metrics plot saved to: {metrics_path}")

    # Load validation dataloader
    data_dir = os.path.join(BACKEND_DIR, "data", "tensors_include50_word_level")
    _, val_loader, class_to_idx, idx_to_class, qualifying_classes = get_balanced_word_dataloaders(
        data_dir=data_dir,
        batch_size=32,
        samples_per_class=14,
        val_samples_per_class=3,
        include_nothing_class=True,
        seed=42
    )
    all_class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    gesture_classes_count = len(qualifying_classes)

    # Load Model: RNN -> Transformer -> TCNN
    model = ISLSentenceReconformer(
        input_dim=163,
        num_classes=gesture_classes_count,
        d_model=128,
        lstm_layers=1,
        trans_layers=4,
        num_heads=4,
        dropout=0.3,
        use_rnn=True,
        freeze_base=False
    ).to(device)

    model_weight_path = os.path.join(save_dir, "reconformer_word_model.pth")
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    print(f"Successfully loaded model weights from {model_weight_path}")

    final_preds, final_targets = [], []
    with torch.no_grad():
        for x, lengths, labels in val_loader:
            x, lengths = x.to(device), lengths.to(device)
            probs_pooled, _ = model.predict_word_probs(x, lengths)
            preds = probs_pooled.argmax(dim=-1)
            final_preds.extend(preds.cpu().numpy())
            final_targets.extend(labels.numpy())

    final_preds = np.array(final_preds)
    final_targets = np.array(final_targets)

    # 2. Confusion Matrix Plot
    cm = confusion_matrix(final_targets, final_preds, labels=range(num_classes))
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    plt.figure(figsize=(24, 22), dpi=200)
    sns.heatmap(
        cm_norm,
        cmap='Blues',
        xticklabels=False,
        yticklabels=False,
        cbar_kws={'label': 'Normalized Recall per Class'}
    )
    plt.title(f"Fresh Reconformer Word Model: Confusion Matrix ({num_classes} Classes)", fontsize=16, fontweight='bold', pad=14)
    plt.xlabel("Predicted Class Index", fontsize=12)
    plt.ylabel("True Class Index", fontsize=12)
    plt.tight_layout()

    cm_path = os.path.join(save_dir, "reconformer_confusion_matrix_fresh.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Confusion matrix saved to: {cm_path}")

    # 3. Classification Report
    report_text = classification_report(
        final_targets, final_preds,
        labels=range(num_classes),
        target_names=classes,
        zero_division=0
    )
    report_path = os.path.join(save_dir, "reconformer_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Classification report saved to: {report_path}")

    # 4. Copy to EDA plots and Artifact Directory
    artifact_dir = r"C:\Users\UnboundSB\.gemini\antigravity-ide\brain\e79d288d-e609-482f-b54e-82f0b240e570"
    eda_dir = os.path.join(BACKEND_DIR, "eda", "plots")
    os.makedirs(eda_dir, exist_ok=True)

    # Save copies to EDA
    shutil.copy2(metrics_path, os.path.join(eda_dir, "reconformer_metrics.png"))
    shutil.copy2(cm_path, os.path.join(eda_dir, "reconformer_confusion_matrix.png"))
    shutil.copy2(cm_path, os.path.join(save_dir, "reconformer_confusion_matrix_229.png"))

    if os.path.exists(artifact_dir):
        shutil.copy2(metrics_path, os.path.join(artifact_dir, "reconformer_metrics.png"))
        shutil.copy2(cm_path, os.path.join(artifact_dir, "reconformer_confusion_matrix.png"))
        shutil.copy2(report_path, os.path.join(artifact_dir, "reconformer_classification_report.txt"))

    print("ALL EVALUATIONS, PLOTS, AND ARTIFACTS GENERATED SUCCESSFULLY!")

if __name__ == "__main__":
    evaluate_and_plot()
