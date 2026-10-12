import os
import sys
import json
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, random_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.isl_sentence_model import ISLSentenceReconformer
from training.train_word_135 import Balanced135WordDataset, collate_fn_word, pool_logits

def evaluate_word_135():
    DATA_DIR = r"d:\MookVani\Backend\data\tensors_include50_word_level"
    MODEL_DIR = r"d:\MookVani\Backend\models\word_level"
    PLOTS_DIR = os.path.join(MODEL_DIR, "plots_135")
    ARTIFACTS_DIR = r"C:\Users\UnboundSB\.gemini\antigravity-ide\brain\e5352803-5ec5-49d4-b73a-3ddf13c0a033"
    
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, "reconformer_word_135_best.pth")
    CLASSES_PATH = os.path.join(MODEL_DIR, "classes_135.json")
    
    with open(CLASSES_PATH, "r") as f:
        meta = json.load(f)
    classes = meta["classes"]
    gesture_classes = [c for c in classes if c != "NONE"]
    num_classes = len(classes)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating 135-Word Model on {device} ({num_classes} classes)")
    
    # Model architecture
    model = ISLSentenceReconformer(num_classes=len(gesture_classes), d_model=128, lstm_layers=1, dropout=0.0, use_rnn=True).to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    
    # Dataset and exact 80/20 validation split (seed 42)
    full_dataset = Balanced135WordDataset(DATA_DIR, classes, max_samples_per_class=15, is_train=False)
    torch.manual_seed(42)
    train_size = int(0.80 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_set = random_split(full_dataset, [train_size, val_size])
    
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, collate_fn=collate_fn_word, num_workers=0)
    
    all_preds, all_targets = [], []
    
    print(f"Running inference across {val_size} validation samples...")
    with torch.no_grad():
        for x, lengths, labels in val_loader:
            x, lengths, labels = x.to(device), lengths.to(device), labels.to(device)
            logits_seq, _ = model(x, lengths)
            logits_pooled = pool_logits(logits_seq, lengths)
            preds = logits_pooled.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    acc = (all_preds == all_targets).mean() * 100.0
    print(f"Validation Exact Match Accuracy: {acc:.2f}% across 136 classes")
    
    # Generate Confusion Matrix
    print("Computing confusion matrix...")
    cm = confusion_matrix(all_targets, all_preds, labels=range(num_classes))
    
    # High-resolution confusion matrix plot
    plt.figure(figsize=(28, 26))
    sns.heatmap(cm, cmap='Blues', xticklabels=classes, yticklabels=classes, annot=False, cbar=True)
    plt.title(f"MookVani 135-Word Reconformer Confusion Matrix (Val Acc: {acc:.2f}%)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Predicted Class", fontsize=12, labelpad=10)
    plt.ylabel("True Class", fontsize=12, labelpad=10)
    plt.xticks(fontsize=6, rotation=90)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    
    cm_path = os.path.join(PLOTS_DIR, "word_135_confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_path}")
    
    # Copy all plots to artifacts directory for display
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    shutil.copy(cm_path, os.path.join(ARTIFACTS_DIR, "word_135_confusion_matrix.png"))
    
    word_metrics_src = os.path.join(PLOTS_DIR, "word_135_metrics.png")
    if os.path.exists(word_metrics_src):
        shutil.copy(word_metrics_src, os.path.join(ARTIFACTS_DIR, "word_135_metrics.png"))
        
    sentence_plots_dir = r"d:\MookVani\Backend\models\islsentence_synthetic\plots"
    for p_name in ["ctc_metrics.png", "loss_and_wer.png"]:
        src = os.path.join(sentence_plots_dir, p_name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(ARTIFACTS_DIR, p_name))
            
    print("Successfully copied all plots to Artifacts directory!")

if __name__ == "__main__":
    evaluate_word_135()
