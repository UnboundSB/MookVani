import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.isl_sentence_model import ISLSentenceReconformer
from training.train_reconformer_word import ISLAugmentedWordDataset, collate_fn, FocalLoss

def evaluate_weak_classes(model, val_loader, classes, device, threshold=0.60):
    model.eval()
    all_preds = []
    all_targets = []
    
    print("Evaluating current model to find weak classes...")
    with torch.no_grad():
        for x, lengths, labels in tqdm(val_loader, desc="Eval"):
            x, lengths, labels = x.to(device), lengths.to(device), labels.to(device)
            logits_seq, _ = model(x, lengths)
            preds = logits_seq.mean(dim=1).argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    report = classification_report(all_targets, all_preds, labels=range(len(classes)), target_names=classes, output_dict=True, zero_division=0)
    
    weak_classes = []
    for cls_name in classes:
        if cls_name == "NONE": continue
        if cls_name in report and report[cls_name]['f1-score'] < threshold:
            weak_classes.append(cls_name)
            
    return weak_classes

def train_weak_classes():
    DATA_DIR = r"d:\MookVani\Backend\data\tensors_include50_word_level"
    MODEL_PATH = r"d:\MookVani\Backend\models\word_level\reconformer_word_model.pth"
    
    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    classes.append("NONE")
    num_classes = len(classes)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ISLSentenceReconformer(num_classes=num_classes, d_model=128, lstm_layers=1, dropout=0.5, use_rnn=True, freeze_base=False).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    
    # Aggressive Focal Loss to heavily penalize misclassifications
    criterion = FocalLoss(alpha=2.0, gamma=3.0, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    full_dataset = ISLAugmentedWordDataset(DATA_DIR, classes, is_train=True)
    
    # CRITICAL: Fix Data Leakage
    torch.manual_seed(42)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Disable augmentation on validation set
    val_dataset.dataset.is_train = False
    
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    # PHASE 1: Active Learning Loop
    print("\n" + "="*60)
    print("PHASE 1: DYNAMIC ACTIVE LEARNING LOOP")
    print("="*60)
    
    # Initial Evaluation to determine N (number of weak classes)
    weak_classes = evaluate_weak_classes(model, val_loader, classes, device, threshold=0.60)
    N = len(weak_classes)
    
    if N == 0:
        print("No weak classes found! (All classes > 60% F1). Skipping Phase 1.")
        cycles = 0
    else:
        # Dynamically calculate cycles! N // 5 cycles.
        cycles = max(1, N // 5)
        epochs_per_cycle = 5
        print(f"Found {N} weak classes. Scaling compute to {cycles} cycles of {epochs_per_cycle} epochs each.")
        
    for cycle in range(cycles):
        print(f"\n--- CYCLE {cycle+1}/{cycles} ---")
        
        # If it's not the first cycle, re-evaluate to get the fresh weak classes
        if cycle > 0:
            weak_classes = evaluate_weak_classes(model, val_loader, classes, device, threshold=0.60)
            
        print(f"Targeting {len(weak_classes)} weak classes: {weak_classes[:10]}...")
        
        if len(weak_classes) == 0:
            print("All weak classes have crossed the 60% F1 threshold! Breaking early.")
            break
            
        # 2. Extract only weak classes from the 80% train split (No leakage!)
        filtered_filepaths, filtered_labels = [], []
        for idx in train_dataset.indices:
            fp = full_dataset.filepaths[idx]
            lbl = full_dataset.labels[idx]
            class_name = full_dataset.classes[lbl]
            if class_name in weak_classes:
                filtered_filepaths.append(fp)
                filtered_labels.append(lbl)
                
        # 3. Create active learning dataset
        weak_dataset = ISLAugmentedWordDataset(DATA_DIR, classes, is_train=True)
        weak_dataset.filepaths = filtered_filepaths
        weak_dataset.labels = filtered_labels
        
        weak_loader = DataLoader(weak_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)
        
        # 4. Train for 5 epochs
        for epoch in range(epochs_per_cycle):
            model.train()
            train_loss = 0
            pbar = tqdm(weak_loader, desc=f"Cycle {cycle+1} - Epoch {epoch+1}/{epochs_per_cycle}")
            for x, lengths, labels in pbar:
                x, lengths, labels = x.to(device), lengths.to(device), labels.to(device)
                optimizer.zero_grad()
                logits_seq, _ = model(x, lengths)
                loss = criterion(logits_seq.mean(dim=1), labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            print(f"Cycle {cycle+1} - Epoch {epoch+1}: Weak Class Loss = {train_loss/len(weak_loader):.4f}")
            
    # PHASE 2: Train on ALL classes for 10 epochs to rebalance biases
    print("\n" + "="*50)
    print("PHASE 2: FULL DATASET REBALANCING (10 EPOCHS)")
    print("="*50)
    
    # Use the same train_dataset for full stabilization (no data leakage!)
    stabilization_dataset = ISLAugmentedWordDataset(DATA_DIR, classes, is_train=True)
    stabilization_dataset.filepaths = [full_dataset.filepaths[i] for i in train_dataset.indices]
    stabilization_dataset.labels = [full_dataset.labels[i] for i in train_dataset.indices]
    
    full_loader = DataLoader(stabilization_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)
    
    # Lower learning rate for stabilization
    for param_group in optimizer.param_groups:
        param_group['lr'] = 5e-5
        
    # Revert focal loss to normal for full dataset
    criterion_full = FocalLoss(alpha=1.0, gamma=2.0, label_smoothing=0.1)
        
    for epoch in range(10):
        model.train()
        train_loss = 0
        pbar = tqdm(full_loader, desc=f"Rebalancing - Epoch {epoch+1}/10")
        for x, lengths, labels in pbar:
            x, lengths, labels = x.to(device), lengths.to(device), labels.to(device)
            optimizer.zero_grad()
            logits_seq, _ = model(x, lengths)
            loss = criterion_full(logits_seq.mean(dim=1), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Rebalancing - Epoch {epoch+1}: Stabilization Loss = {train_loss/len(full_loader):.4f}")
        
    # Save the final rescue model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nSaved active-learning fine-tuned weights back to {MODEL_PATH}")

if __name__ == "__main__":
    train_weak_classes()
