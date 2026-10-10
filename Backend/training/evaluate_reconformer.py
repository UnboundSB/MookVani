import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.isl_sentence_model import ISLSentenceReconformer
from training.train_reconformer_word import ISLAugmentedWordDataset, collate_fn

def evaluate_model():
    DATA_DIR = r"d:\MookVani\Backend\data\tensors_include50_word_level"
    MODEL_PATH = r"d:\MookVani\Backend\models\word_level\reconformer_word_model_final.pth"
    
    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    classes.append("NONE")
    num_classes = len(classes)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on {device}")
    
    # Initialize the model with the exact same architecture as training
    model = ISLSentenceReconformer(num_classes=num_classes, d_model=128, lstm_layers=1, dropout=0.0, use_rnn=True, freeze_base=False).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    
    # Load dataset (turn off augmentations)
    full_dataset = ISLAugmentedWordDataset(DATA_DIR, classes, is_train=False)
    
    # CRITICAL: Fix Data Leakage by strictly seeding the random split
    torch.manual_seed(42)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    all_preds = []
    all_targets = []
    
    print(f"Running inference on all {len(val_dataset)} videos/samples...")
    with torch.no_grad():
        for x, lengths, labels in dataloader:
            x, lengths, labels = x.to(device), lengths.to(device), labels.to(device)
            logits_seq, _ = model(x, lengths)
            
            # Mean pool temporal logits
            logits_pooled = logits_seq.mean(dim=1)
            
            preds = logits_pooled.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    print("Generating Confusion Matrix (this might take a second for 263 classes)...")
    cm = confusion_matrix(all_targets, all_preds)
    
    # Plotting a massive 263x263 heatmap
    plt.figure(figsize=(50, 50))
    sns.heatmap(cm, cmap='Blues', xticklabels=classes, yticklabels=classes, annot=False)
    plt.title("Confusion Matrix for INCLUDE50 (263 Classes)")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    
    plot_path = r"d:\MookVani\Backend\models\word_level\reconformer_confusion_matrix_final.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Saved massive confusion matrix to {plot_path}")
    
    # Save classification report
    report = classification_report(all_targets, all_preds, labels=range(len(classes)), target_names=classes, zero_division=0)
    report_path = r"d:\MookVani\Backend\models\word_level\reconformer_classification_report_final.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved highly detailed classification report to {report_path}")

if __name__ == "__main__":
    evaluate_model()
