import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from models.isl_conformer import ISL_Conformer
from data.datasets import build_word_dataloaders

def plot_conf_matrix():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_dir = "data/tensors_word_level_163_val"
    train_dir = "data/tensors_word_level_163_train" # Needed for vocabulary build in current setup
    model_path = "models/best_word_model.pth"
    word_class_json = "models/word_class_to_idx.json"
    plot_dir = "models/plots"

    if not os.path.exists(word_class_json):
        print(f"Error: {word_class_json} missing.")
        return

    with open(word_class_json, "r") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    print("Loading dataloaders...")
    _, val_loader, _ = build_word_dataloaders(train_dir, val_dir, batch_size=32)

    print("Loading model...")
    model = ISL_Conformer(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    all_preds = []
    all_labels = []

    print("Evaluating validation set...")
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model.forward_single_frame_logits(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    print("Generating confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(24, 20))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=[idx_to_class.get(i, str(i)) for i in range(num_classes)],
                yticklabels=[idx_to_class.get(i, str(i)) for i in range(num_classes)])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Word-Level Validation Confusion Matrix')
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    
    os.makedirs(plot_dir, exist_ok=True)
    out_path = os.path.join(plot_dir, "word_confusion_matrix.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix to {out_path}")

if __name__ == "__main__":
    plot_conf_matrix()
