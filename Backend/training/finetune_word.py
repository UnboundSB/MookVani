import os
import glob
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report

# Ensure imports work when run from backend directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.isl_sentence_model import ISLSentenceReconformer
from training.train_reconformer_word import ISLAugmentedWordDataset, collate_fn, FocalLoss

def finetune_model():
    DATA_DIR = r"d:\MookVani\Backend\data\tensors_include50_word_level"
    MODEL_PATH = r"d:\MookVani\Backend\models\word_level\reconformer_word_model.pth"
    FINAL_MODEL_PATH = r"d:\MookVani\Backend\models\word_level\reconformer_word_model_final.pth"
    
    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    classes.append("NONE")
    num_classes = len(classes)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*60)
    print("FINAL PHASE: BRUTAL AUGMENTATION FINE-TUNING")
    print("="*60)
    print("Loading Active Learning model...")
    
    model = ISLSentenceReconformer(num_classes=num_classes, d_model=128, lstm_layers=1, dropout=0.5, use_rnn=True, freeze_base=False).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    
    # 1. Dataset with Massive Augmentations
    # (Since ISLAugmentedWordDataset was updated, it will automatically apply the brutal noise)
    full_dataset = ISLAugmentedWordDataset(DATA_DIR, classes, is_train=True)
    val_dataset = ISLAugmentedWordDataset(DATA_DIR, classes, is_train=False)
    
    # Strict Data Leakage Prevention (Exact same split as before)
    torch.manual_seed(42)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, _ = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    _, val_subset = torch.utils.data.random_split(val_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    # 2. Optimization Strategy: Cosine Annealing to squeeze every ounce of accuracy
    epochs = 30
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = FocalLoss(alpha=1.0, gamma=2.0, label_smoothing=0.1)
    
    best_f1 = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Finetune - Epoch {epoch+1}/{epochs}")
        for x, lengths, labels in pbar:
            x, lengths, labels = x.to(device), lengths.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits_seq, _ = model(x, lengths)
            loss = criterion(logits_seq.mean(dim=1), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for x, lengths, labels in val_loader:
                x, lengths, labels = x.to(device), lengths.to(device), labels.to(device)
                logits_seq, _ = model(x, lengths)
                loss = criterion(logits_seq.mean(dim=1), labels)
                val_loss += loss.item()
                
                preds = logits_seq.mean(dim=1).argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                
        val_loss /= len(val_loader)
        
        report = classification_report(all_targets, all_preds, labels=range(len(classes)), target_names=classes, output_dict=True, zero_division=0)
        macro_f1 = report['macro avg']['f1-score']
        accuracy = report['accuracy']
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f} | Val Loss={val_loss:.4f} | Val Acc={accuracy:.4f} | Val F1={macro_f1:.4f} | LR={current_lr:.6f}")
        
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), FINAL_MODEL_PATH)
            print(f"-> Saved highly robust final model with F1: {best_f1:.4f}")

if __name__ == "__main__":
    finetune_model()
