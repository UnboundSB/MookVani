"""
Generates high-resolution 50x50 inch confusion matrix with FULL row and column labels
for all classes, ensuring every word name is clearly readable.
"""
import os
import sys
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

BACKEND_DIR = r"d:\MookVani\Backend"
sys.path.insert(0, BACKEND_DIR)

from models.isl_sentence_model import ISLSentenceReconformer
from scripts.dataset import get_balanced_word_dataloaders

def generate_fully_labeled_matrix():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Generating fully labeled confusion matrix on {device}...")

    # Load 229 balanced classes metadata
    meta_path = os.path.join(BACKEND_DIR, "models", "word_level", "classes_balanced.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    classes = meta["classes"]
    num_classes = len(classes)
    gesture_classes_count = num_classes - 1 # excluding [NOTHING]
    print(f"Total vocabulary classes to label: {num_classes}")

    # Load Model
    model = ISLSentenceReconformer(
        input_dim=163,
        num_classes=gesture_classes_count,
        d_model=128,
        lstm_layers=1,
        trans_layers=4,
        num_heads=4,
        dropout=0.0,
        use_rnn=True,
        freeze_base=False
    ).to(device)

    model_weight_path = os.path.join(BACKEND_DIR, "models", "word_reconformer_229", "reconformer_word_model.pth")
    if not os.path.exists(model_weight_path):
        model_weight_path = os.path.join(BACKEND_DIR, "models", "word_level", "reconformer_word_model.pth")
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # Load dataset
    data_dir = os.path.join(BACKEND_DIR, "data", "tensors_include50_word_level")
    train_loader, val_loader, _, _, _ = get_balanced_word_dataloaders(
        data_dir=data_dir,
        batch_size=32,
        samples_per_class=14,
        val_samples_per_class=0,
        include_nothing_class=True,
        seed=42
    )

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, lengths, labels in val_loader:
            x, lengths = x.to(device), lengths.to(device)
            logits_pooled = model.forward_word_logits(x, lengths)
            preds = logits_pooled.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    cm = confusion_matrix(all_targets, all_preds, labels=range(num_classes))
    # Row-normalized recall (Sensitivity per class) so true positives on the diagonal stand out prominently
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    # Plot 48x46 inches with EVERY SINGLE CLASS LABEL PRINTED ON BOTH ROWS AND COLUMNS
    print("Plotting 48x46 inch confusion matrix with all row and column labels...")
    fig, ax = plt.subplots(figsize=(48, 46), dpi=150)
    
    sns.heatmap(
        cm_norm,
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        annot=False,
        cbar_kws={'label': 'Row-Normalized Recall (Class Sensitivity)', 'shrink': 0.8},
        ax=ax
    )
    
    # Explicitly enforce row (y-axis) and column (x-axis) ticks and labels
    ax.set_xticks(np.arange(num_classes) + 0.5)
    ax.set_xticklabels(classes, rotation=90, fontsize=5.5, ha='center')
    ax.set_yticks(np.arange(num_classes) + 0.5)
    ax.set_yticklabels(classes, rotation=0, fontsize=5.5, va='center')
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    
    # Overlay diagonal reference line (y = x) so the diagonal is crystal clear across all 229 classes
    ax.plot([0, num_classes], [0, num_classes], color='orange', linestyle='--', linewidth=1.5, alpha=0.75, label='Ideal Diagonal (y=x)')
    ax.legend(loc='upper left', fontsize=14, framealpha=0.9)
    
    plt.title(f"Reconformer Word Classification: Confusion Matrix ({num_classes} Classes with Full Row & Column Labels)", fontsize=22, fontweight='bold', pad=25)
    plt.xlabel("Predicted Word Class (Columns)", fontsize=18, labelpad=15)
    plt.ylabel("True Word Class (Rows)", fontsize=18, labelpad=15)
    
    # Ensure generous margins so row labels on the left are completely unclipped
    fig.subplots_adjust(left=0.10, bottom=0.10, right=0.98, top=0.96)

    out_paths = [
        os.path.join(BACKEND_DIR, "..", "confusion_matrix.png"),
        os.path.join(BACKEND_DIR, "confusion_matrix.png"),
        os.path.join(BACKEND_DIR, "models", "word_reconformer_229", "reconformer_confusion_matrix.png"),
        r"C:\Users\UnboundSB\.gemini\antigravity-ide\brain\e79d288d-e609-482f-b54e-82f0b240e570\reconformer_confusion_matrix.png"
    ]

    for p in out_paths:
        try:
            plt.savefig(p, dpi=150)
            print(f"Saved labeled confusion matrix to {p}")
        except Exception as e:
            print(f"Error saving to {p}: {e}")

    plt.close()
    print("Done generating labeled confusion matrix!")

if __name__ == "__main__":
    generate_fully_labeled_matrix()
