import os
import sys
import glob
import json
import time
import gc
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# UTF-8 stdout on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.isl_sentence_model import ISLSentenceReconformer
import torch.nn.functional as F

class Balanced135WordDataset(Dataset):
    def __init__(self, data_dir, classes, max_samples_per_class=15, is_train=True):
        self.data_dir = data_dir
        self.classes = classes
        self.is_train = is_train
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        # Pre-collect real resting boundary frames from actual videos for realistic silent class
        self.rest_segments = []
        for cls in classes:
            if cls == "NONE":
                continue
            cls_dir = os.path.join(data_dir, cls)
            if os.path.isdir(cls_dir):
                files = sorted(glob.glob(os.path.join(cls_dir, "*.pt")))[:3]
                for f in files:
                    try:
                        x = torch.load(f, weights_only=True)
                        if x.shape[0] >= 35:
                            self.rest_segments.append(x[:18])
                            self.rest_segments.append(x[-18:])
                    except Exception:
                        pass
        if len(self.rest_segments) == 0:
            self.rest_segments.append(torch.randn(20, 163) * 0.05)
            
        self.samples = []
        for cls in classes:
            if cls == "NONE":
                # Add 15 realistic resting pose samples
                for _ in range(max_samples_per_class):
                    self.samples.append(("NONE", None, self.class_to_idx["NONE"]))
                continue
                
            cls_dir = os.path.join(data_dir, cls)
            if os.path.isdir(cls_dir):
                files = sorted(glob.glob(os.path.join(cls_dir, "*.pt")))[:max_samples_per_class]
                for f in files:
                    self.samples.append((cls, f, self.class_to_idx[cls]))
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        cls, filepath, label = self.samples[idx]
        
        if cls == "NONE":
            # Sample and stitch real resting segments from actual signers
            idx1 = np.random.randint(0, len(self.rest_segments))
            idx2 = np.random.randint(0, len(self.rest_segments))
            x = torch.cat([self.rest_segments[idx1], self.rest_segments[idx2]], dim=0)
            
            if self.is_train:
                if np.random.rand() > 0.4:
                    scale = np.random.uniform(0.8, 1.2)
                    new_T = max(5, int(x.size(0) * scale))
                    x_trans = x.unsqueeze(0).transpose(1, 2)
                    x_interp = F.interpolate(x_trans, size=new_T, mode='linear', align_corners=False)
                    x = x_interp.transpose(1, 2).squeeze(0)
                if np.random.rand() > 0.3:
                    x = x + torch.randn_like(x) * 0.02
                if np.random.rand() > 0.5:
                    shift = torch.randn(1, 163) * 0.02
                    x = x + shift
            return x, x.shape[0], label
            
        x = torch.load(filepath, weights_only=True)
        
        if self.is_train:
            # 1. Temporal scaling (0.8x to 1.2x)
            if np.random.rand() > 0.4:
                scale = np.random.uniform(0.8, 1.2)
                new_T = max(5, int(x.size(0) * scale))
                x_trans = x.unsqueeze(0).transpose(1, 2)
                x_interp = F.interpolate(x_trans, size=new_T, mode='linear', align_corners=False)
                x = x_interp.transpose(1, 2).squeeze(0)
                
            # 2. Subtle spatial jitter & shift
            if np.random.rand() > 0.3:
                x = x + torch.randn_like(x) * 0.02
            if np.random.rand() > 0.5:
                shift = torch.randn(1, 163) * 0.02
                x = x + shift
                
        return x, x.shape[0], label

def collate_fn_word(batch):
    xs, lengths, labels = zip(*batch)
    max_len = max(lengths)
    batch_size = len(xs)
    
    x_padded = torch.zeros(batch_size, max_len, xs[0].shape[-1])
    for i, x in enumerate(xs):
        x_padded[i, :lengths[i], :] = x
        
    return x_padded, torch.tensor(lengths, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

def pool_logits(logits_seq, lengths):
    B, T, C = logits_seq.shape
    arange = torch.arange(T, device=logits_seq.device).unsqueeze(0).expand(B, -1)
    mask = (arange < lengths.unsqueeze(1)).unsqueeze(-1).float()
    pooled = (logits_seq * mask).sum(dim=1) / lengths.unsqueeze(1).clamp(min=1).float()
    return pooled

def train_word_model():
    DATA_DIR = r"d:\MookVani\Backend\data\tensors_include50_word_level"
    MODEL_DIR = r"d:\MookVani\Backend\models\word_level"
    PLOTS_DIR = os.path.join(MODEL_DIR, "plots_135")
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, "reconformer_word_135_best.pth")
    FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "reconformer_word_135_final.pth")
    CLASSES_PATH = os.path.join(MODEL_DIR, "classes_135.json")
    
    # Filter to classes having at least 15 samples (135 classes) without modifying or deleting anything on disk
    all_classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    gesture_classes = sorted([c for c in all_classes if len(glob.glob(os.path.join(DATA_DIR, c, "*.pt"))) >= 15])
    classes = gesture_classes + ["NONE"]
    num_classes = len(classes) # 136
    
    print("=" * 65)
    print("MOOKVANI BALANCED 135-CLASS WORD LEVEL PRETRAINING")
    print("=" * 65)
    print(f"Total Gesture Classes: {len(gesture_classes)} + 1 [NONE] = {num_classes} classes")
    print(f"Samples Per Class:     Exactly 15 samples (Balanced in memory)")
    print(f"Total Dataset Size:    {num_classes * 15} samples (No disk files deleted)")
    print("=" * 65 + "\n")
    
    with open(CLASSES_PATH, "w") as f:
        json.dump({"classes": classes, "num_classes": num_classes}, f, indent=2)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create dataset
    full_dataset = Balanced135WordDataset(DATA_DIR, classes, max_samples_per_class=15, is_train=True)
    
    # 80-20 Train/Val Split (Deterministic seed 42)
    torch.manual_seed(42)
    train_size = int(0.80 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn_word, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, collate_fn=collate_fn_word, num_workers=0)
    
    print(f"Train samples: {train_size} | Validation samples: {val_size}")
    
    # ISLSentenceReconformer with 135 classes -> CNN output channels = 136 (0..135)
    model = ISLSentenceReconformer(num_classes=len(gesture_classes), d_model=128, lstm_layers=1, dropout=0.3, use_rnn=True).to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-2)
    epochs = 30
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_f1 = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        all_train_preds, all_train_targets = [], []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{epochs} [Train]")
        for x, lengths, labels in pbar:
            x, lengths, labels = x.to(device), lengths.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits_seq, _ = model(x, lengths) # [B, T, 136]
            logits_pooled = pool_logits(logits_seq, lengths) # [B, 136]
            
            loss = criterion(logits_pooled, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = logits_pooled.argmax(dim=-1)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_targets.extend(labels.cpu().numpy())
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        train_loss /= len(train_loader)
        train_acc = accuracy_score(all_train_targets, all_train_preds)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_val_preds, all_val_targets = [], []
        
        with torch.no_grad():
            for x, lengths, labels in val_loader:
                x, lengths, labels = x.to(device), lengths.to(device), labels.to(device)
                logits_seq, _ = model(x, lengths)
                logits_pooled = pool_logits(logits_seq, lengths)
                
                loss = criterion(logits_pooled, labels)
                val_loss += loss.item()
                preds = logits_pooled.argmax(dim=-1)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_targets.extend(labels.cpu().numpy())
                
        val_loss /= len(val_loader)
        val_acc = accuracy_score(all_val_targets, all_val_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(all_val_targets, all_val_preds, average='macro', zero_division=0)
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(f1)
        
        print(f"Epoch {epoch+1:02d}/{epochs}: Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%, F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f" -> Saved new best model to {BEST_MODEL_PATH} (F1: {best_f1:.4f}, Acc: {val_acc*100:.2f}%)")
            
        gc.collect()
        torch.cuda.empty_cache()
        
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    elapsed = time.time() - start_time
    print("\n" + "=" * 65)
    print(f"WORD PRETRAINING COMPLETE in {elapsed/60:.2f} minutes!")
    print(f"Best Val F1: {best_f1:.4f} | Best Weights: {BEST_MODEL_PATH}")
    print("=" * 65)
    
    # Plot training metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ep = range(1, epochs + 1)
    ax1.plot(ep, history['train_loss'], label='Train Loss', color='blue')
    ax1.plot(ep, history['val_loss'], label='Val Loss', color='orange')
    ax1.set_title('Cross-Entropy Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(ep, [a*100 for a in history['train_acc']], label='Train Acc %', color='blue')
    ax2.plot(ep, [a*100 for a in history['val_acc']], label='Val Acc %', color='orange')
    ax2.plot(ep, [f*100 for f in history['val_f1']], label='Val F1 %', color='green')
    ax2.set_title('Accuracy & F1 Progression')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "word_135_metrics.png"), dpi=150)
    plt.close()
    print(f"Plots saved to {os.path.join(PLOTS_DIR, 'word_135_metrics.png')}")

if __name__ == "__main__":
    train_word_model()
