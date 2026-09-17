import os
import glob
import random
from collections import defaultdict
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

from models.conformer import ISL_Conformer
from data.dataset import SaneSkeletalAugmentation, ISLWordLevelDataset

def word_level_collate(batch):
    inputs = torch.stack([x for x, _ in batch])
    labels = torch.stack([y for _, y in batch])
    lengths = torch.ones(len(batch), dtype=torch.long)
    targets = labels.unsqueeze(1) + 1
    tgt_lens = torch.ones(len(batch), dtype=torch.long)
    return inputs, targets, lengths, tgt_lens

def get_class_accuracies(model, val_loader, device, num_classes):
    model.eval()
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for batch_inputs, batch_targets, in_lens, _ in val_loader:
            batch_inputs = batch_inputs.to(device)
            in_lens = in_lens.to(device)
            log_probs, _ = model(batch_inputs, in_lens) # (B, T, C)
            
            batch_preds = log_probs.argmax(dim=-1) # shape (B, T)
            for b in range(batch_inputs.size(0)):
                decoded = []
                prev = -1
                for t in range(batch_preds.size(1)):
                    p = batch_preds[b, t].item()
                    if p != 0 and p != prev:
                        decoded.append(p - 1)
                    prev = p
                
                target = batch_targets[b].item() - 1
                pred = decoded[0] if len(decoded) > 0 else -1
                
                all_targets.append(target)
                all_preds.append(pred)
                
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
    accuracies = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    return accuracies, cm

def build_dataset(train_pairs, train_by_class, hard_classes, class_to_idx, syn_dir, stage, target_count=30):
    balanced_train_pairs = []

    print(f"Balancing dataset for Stage {stage}...")
    for cls, files in train_by_class.items():
        if not files: continue
        
        if stage == 2:
            current_target = 100 if cls in hard_classes else 20
        else:
            current_target = target_count
            
        # 1. Add original files (truncate to current_target if there are too many)
        original_to_add = files[:current_target]
        for f in original_to_add:
            balanced_train_pairs.append((f, cls))
            
        # 2. If we still need more samples, duplicate existing file paths.
        # The ISLWordLevelDataset will apply on-the-fly augmentation differently every time they are loaded!
        current = len(original_to_add)
        while current < current_target:
            base_path = random.choice(files)
            balanced_train_pairs.append((base_path, cls))
            current += 1

    # On-the-fly augmentation applied during training (conservatively)
    augmenter = SaneSkeletalAugmentation(apply_prob=0.7, max_rot_degrees=10.0, max_shift=0.05, noise_std=0.01)
    return ISLWordLevelDataset(balanced_train_pairs, class_to_idx, transform=augmenter)

def run_training_cycle(model, train_loader, val_loader, optimizer, scheduler, criterion, device, epochs, cycle_name, metrics, best_val_loss, save_path, idx_to_class):
    print(f"\n{'='*40}\nStarting {cycle_name}\n{'='*40}")
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, valid_batches = 0.0, 0
        correct_train, total_train = 0, 0

        for batch_inputs, batch_targets, in_lens, tgt_lens in tqdm(train_loader, desc=f"{cycle_name} Epoch {epoch}/{epochs}"):
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            in_lens = in_lens.to(device)

            optimizer.zero_grad()
            log_probs, pooled_lens = model(batch_inputs, in_lens)
            log_probs = log_probs.permute(1, 0, 2)
            loss = criterion(log_probs, batch_targets, pooled_lens.clamp(min=1), tgt_lens.to(device))

            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                train_loss += loss.item()
                valid_batches += 1
                
                # Accuracy tracking for train
                batch_preds = log_probs.argmax(dim=-1)
                for b in range(batch_inputs.size(0)):
                    decoded = []
                    prev = -1
                    for t in range(batch_preds.size(0)):
                        p = batch_preds[t, b].item()
                        if p != 0 and p != prev:
                            decoded.append(p - 1)
                        prev = p
                    target = batch_targets[b].item() - 1
                    if len(decoded) > 0 and decoded[0] == target: correct_train += 1
                    total_train += 1

        avg_train_loss = train_loss / max(1, valid_batches)
        train_accuracy = (correct_train / total_train) * 100 if total_train > 0 else 0

        model.eval()
        val_loss, val_valid_batches = 0.0, 0
        correct_val, total_val = 0, 0
        sample_printed = False
        
        with torch.no_grad():
            for batch_inputs, batch_targets, in_lens, tgt_lens in val_loader:
                batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
                in_lens = in_lens.to(device)
                log_probs, pooled_lens = model(batch_inputs, in_lens)
                log_probs = log_probs.permute(1, 0, 2)
                loss = criterion(log_probs, batch_targets, pooled_lens.clamp(min=1), tgt_lens.to(device))
                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss += loss.item()
                    val_valid_batches += 1
                
                batch_preds = log_probs.argmax(dim=-1)
                for b in range(batch_inputs.size(0)):
                    decoded = []
                    prev = -1
                    for t in range(batch_preds.size(0)):
                        p = batch_preds[t, b].item()
                        if p != 0 and p != prev:
                            decoded.append(p - 1)
                        prev = p
                    target = batch_targets[b].item() - 1
                    if len(decoded) > 0 and decoded[0] == target: correct_val += 1
                    total_val += 1
                    
                if not sample_printed and batch_inputs.size(0) > 0:
                    target = batch_targets[0].item() - 1
                    target_word = idx_to_class.get(target, "Unknown")
                    decoded_sample = []
                    prev = -1
                    for t in range(batch_preds.size(0)):
                        p = batch_preds[t, 0].item()
                        if p != 0 and p != prev:
                            decoded_sample.append(p - 1)
                        prev = p
                    decoded_words = [idx_to_class.get(idx, "Unknown") for idx in decoded_sample]
                    print(f"  [Sample] Target: {target_word} | Predicted: {decoded_words}")
                    sample_printed = True

        avg_val_loss = val_loss / max(1, val_valid_batches)
        val_accuracy = (correct_val / total_val) * 100 if total_val > 0 else 0
        
        metrics['train_loss'].append(avg_train_loss)
        metrics['val_loss'].append(avg_val_loss)
        metrics['train_acc'].append(train_accuracy)
        metrics['val_acc'].append(val_accuracy)

        scheduler.step(avg_val_loss)
        
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Val Acc: {val_accuracy:.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved for {cycle_name}.")
            
    return best_val_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    WORD_DATA_DIR = os.path.join(BASE_DIR, "data", "tensors_word_level_163")
    SYN_DIR = os.path.join(BASE_DIR, "data", "tensors_word_level_163_synth")
    
    all_files = glob.glob(f"{WORD_DATA_DIR}/**/*.pt", recursive=True)
    if not all_files:
        print(f"No files found in {WORD_DATA_DIR} - run extract_features.py first.")
        return

    classes = sorted(set(os.path.basename(os.path.dirname(p)) for p in all_files))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    num_classes = len(classes)

    class_to_files = defaultdict(list)
    for p in all_files:
        class_to_files[os.path.basename(os.path.dirname(p))].append(p)

    train_pairs, val_pairs = [], []
    for cls, files in class_to_files.items():
        random.shuffle(files)
        split_point = max(1, int(0.8 * len(files)))
        for f in files[:split_point]: train_pairs.append((f, cls))
        for f in files[split_point:]: val_pairs.append((f, cls))
        
    print(f"Real-data split: {len(train_pairs)} train / {len(val_pairs)} val")

    train_by_class = defaultdict(list)
    for f, cls in train_pairs:
        train_by_class[cls].append(f)
        
    val_dataset = ISLWordLevelDataset(val_pairs, class_to_idx, transform=None)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=word_level_collate, num_workers=2, pin_memory=True)

    model = ISL_Conformer(input_dim=163, num_classes=num_classes, dropout=0.6).to(device)
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    metrics = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    # ---------------------------------------------------------
    # STAGE 1: Base Training (20 Epochs)
    # ---------------------------------------------------------
    base_model_path = "best_isl_conformer_word.pth"
    if not os.path.exists(base_model_path):
        print("\n=== STAGE 1: BASE TRAINING ===")
        train_dataset_s1 = build_dataset(train_pairs, train_by_class, [], class_to_idx, SYN_DIR, stage=1, target_count=30)
        train_loader_s1 = DataLoader(train_dataset_s1, batch_size=32, shuffle=True, collate_fn=word_level_collate, num_workers=2, pin_memory=True)
        
        optimizer_s1 = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
        scheduler_s1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_s1, mode='min', factor=0.5, patience=3)
        
        run_training_cycle(model, train_loader_s1, val_loader, optimizer_s1, scheduler_s1, criterion, device, 
                           epochs=20, cycle_name="Stage 1 - Base", metrics=metrics, best_val_loss=float('inf'), save_path=base_model_path, idx_to_class=idx_to_class)
    
    # Load best base model to begin Stage 2
    if os.path.exists(base_model_path):
        model.load_state_dict(torch.load(base_model_path, map_location=device))
        print(f"\nLoaded Stage 1 weights from {base_model_path}")
    
    # ---------------------------------------------------------
    # STAGE 2: Iterative Hard Negative Mining (4 cycles x 5 epochs)
    # ---------------------------------------------------------
    print("\n=== STAGE 2: DYNAMIC HARD NEGATIVE MINING ===")
    best_loss_s2 = float('inf')
    optimizer_s2 = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler_s2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_s2, mode='min', factor=0.5, patience=3)
    
    ACCURACY_THRESHOLD = 0.80 # Classes above 80% graduate out of intensive mining
    
    for cycle in range(1, 5):
        print(f"\n--- Stage 2 | Cycle {cycle}/4 ---")
        accuracies, _ = get_class_accuracies(model, val_loader, device, num_classes)
        
        # Identify hard classes dynamically
        hard_classes = []
        for i, acc in enumerate(accuracies):
            if acc < ACCURACY_THRESHOLD:
                hard_classes.append(idx_to_class[i])
                
        if len(hard_classes) == 0:
            print("All classes achieved >= 80% accuracy! Early stopping Stage 2.")
            break
            
        print(f"Found {len(hard_classes)} classes under {int(ACCURACY_THRESHOLD*100)}% accuracy. Oversampling them...")
        
        train_dataset_cycle = build_dataset(train_pairs, train_by_class, hard_classes, class_to_idx, SYN_DIR, stage=2)
        train_loader_cycle = DataLoader(train_dataset_cycle, batch_size=32, shuffle=True, collate_fn=word_level_collate, num_workers=2, pin_memory=True)
        
        best_loss_s2 = run_training_cycle(model, train_loader_cycle, val_loader, optimizer_s2, scheduler_s2, criterion, device, 
                                          epochs=5, cycle_name=f"Stage 2 - Cycle {cycle}", metrics=metrics, best_val_loss=best_loss_s2, save_path="best_isl_conformer_word_stage2.pth", idx_to_class=idx_to_class)
    
    # ---------------------------------------------------------
    # STAGE 3: Full Dataset Harmonization (10 Epochs)
    # ---------------------------------------------------------
    print("\n=== STAGE 3: FINAL HARMONIZATION ===")
    if os.path.exists("best_isl_conformer_word_stage2.pth"):
        model.load_state_dict(torch.load("best_isl_conformer_word_stage2.pth", map_location=device))
        
    train_dataset_stage3 = build_dataset(train_pairs, train_by_class, [], class_to_idx, SYN_DIR, stage=3, target_count=30)
    train_loader_stage3 = DataLoader(train_dataset_stage3, batch_size=32, shuffle=True, collate_fn=word_level_collate, num_workers=2, pin_memory=True)
    
    optimizer_s3 = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4) # Lower LR for final harmonization
    scheduler_s3 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_s3, mode='min', factor=0.5, patience=3)
    
    run_training_cycle(model, train_loader_stage3, val_loader, optimizer_s3, scheduler_s3, criterion, device, 
                       epochs=10, cycle_name="Stage 3 - Final", metrics=metrics, best_val_loss=float('inf'), save_path="ultimate_isl_conformer_word.pth", idx_to_class=idx_to_class)
                       
    # ---------------------------------------------------------
    # FINAL METRICS AND PLOTTING
    # ---------------------------------------------------------
    if os.path.exists("ultimate_isl_conformer_word.pth"):
        model.load_state_dict(torch.load("ultimate_isl_conformer_word.pth", map_location=device))
        
    print("\nGenerating final visual metrics...")
    
    # 1. Learning Curves
    epochs_range = range(1, len(metrics['train_loss']) + 1)
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, metrics['train_loss'], label="Train Loss")
    plt.plot(epochs_range, metrics['val_loss'], label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("CTC Loss")
    plt.title("Full Pipeline Loss (Stages 1-3)")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, metrics['train_acc'], label="Train Acc (%)")
    plt.plot(epochs_range, metrics['val_acc'], label="Val Acc (%)")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Full Pipeline Accuracy (Stages 1-3)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("curriculum_learning_curves.png", dpi=300)
    print("Saved -> curriculum_learning_curves.png")
    
    # 2. Final Confusion Matrix Heatmap
    _, cm = get_class_accuracies(model, val_loader, device, num_classes)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title("Final Validation Confusion Matrix")
    plt.xlabel("Predicted Sign")
    plt.ylabel("True Sign")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig("final_confusion_matrix.png", dpi=300)
    print("Saved -> final_confusion_matrix.png")
    
    print("\nTraining completely finished! The ultimate model is saved as 'ultimate_isl_conformer_word.pth'.")

if __name__ == "__main__":
    main()
