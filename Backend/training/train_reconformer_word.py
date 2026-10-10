import os
import glob
import math
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.isl_sentence_model import ISLSentenceReconformer
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.ce = nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class ISLAugmentedWordDataset(Dataset):
    def __init__(self, data_dir, classes, max_len=300, is_train=True):
        self.data_dir = data_dir
        self.classes = classes
        self.max_len = max_len
        self.is_train = is_train
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        self.filepaths = []
        self.labels = []
        
        for cls in classes:
            cls_dir = os.path.join(data_dir, cls)
            if not os.path.exists(cls_dir): continue
            
            for f in glob.glob(os.path.join(cls_dir, "*.pt")):
                self.filepaths.append(f)
                self.labels.append(self.class_to_idx[cls])
                
        # Add artificial NONE samples
        if "NONE" in classes:
            none_idx = self.class_to_idx["NONE"]
            # Add a proportional number of NONE samples (equal to average samples per class)
            avg_samples = max(1, len(self.filepaths) // (len(classes) - 1)) if len(classes) > 1 else 32
            for i in range(avg_samples):
                self.filepaths.append(f"SYNTHETIC_NONE_{i}")
                self.labels.append(none_idx)
                
    def __len__(self):
        return len(self.filepaths)
        
    def augment_temporal(self, x):
        T, F = x.shape
        if T < 10:
            return x
            
        # 1. Massive Time-scaling (0.5x to 1.5x)
        scale = np.random.uniform(0.5, 1.5)
        new_T = max(5, int(T * scale))
        
        indices = np.linspace(0, T - 1, new_T)
        indices_floor = np.floor(indices).astype(int)
        indices_ceil = np.ceil(indices).astype(int)
        weight = (indices - indices_floor)[:, np.newaxis]
        x_interp = x[indices_floor] * (1 - weight) + x[indices_ceil] * weight
        
        # 2. Random Frame Dropping (Simulating missed tracking frames)
        if np.random.rand() > 0.5:
            drop_mask = (torch.rand(new_T, 1) > 0.15).float() # Drop ~15% of frames
            x_interp = x_interp * drop_mask
            
        return x_interp

    def augment_spatial(self, x):
        T, F = x.shape
        x_aug = x.clone()
        
        # 1. Heavy Jitter (Noise)
        noise = torch.randn_like(x_aug) * 0.02 # quadrupled jitter
        x_aug += noise
        
        # 2. Heavy Shift (Translation)
        shift = (torch.rand(1) - 0.5) * 0.15
        x_aug += shift
        
        # 3. Aggressive Tilt (Scale)
        scale = torch.rand(1) * 0.3 + 0.85 # 0.85 to 1.15
        x_aug *= scale
        
        # 4. Partial Feature Masking (Simulating occlusions)
        if np.random.rand() > 0.5:
            feature_mask = (torch.rand(1, F) > 0.1).float() # Mask out 10% of features (like a hand missing)
            x_aug = x_aug * feature_mask
            
        return x_aug

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        label = self.labels[idx]
        
        if path.startswith("SYNTHETIC_NONE"):
            # Generate garbage position impossible for humans (extreme random noise)
            T_noise = np.random.randint(20, 100)
            x = (torch.rand(T_noise, 163) - 0.5) * 20.0 
        else:
            x = torch.load(path, weights_only=True) # (T, 163)
            if self.is_train:
                x = self.augment_temporal(x)
                x = self.augment_spatial(x)
            
        # Truncate if too long
        if x.shape[0] > self.max_len:
            x = x[:self.max_len]
            
        length = x.shape[0]
        return x, length, label

def collate_fn(batch):
    xs, lengths, labels = zip(*batch)
    
    max_len = max(lengths)
    batch_size = len(xs)
    
    x_padded = torch.zeros(batch_size, max_len, xs[0].shape[-1])
    for i, x in enumerate(xs):
        x_padded[i, :lengths[i], :] = x
        
    return x_padded, torch.tensor(lengths), torch.tensor(labels)

def train_reconformer():
    DATA_DIR = r"d:\MookVani\Backend\data\tensors_include50_word_level"
    PRETRAINED_WEIGHTS = r"d:\MookVani\Backend\models\best_synthetic_model.pth"
    SAVE_PATH = r"d:\MookVani\Backend\models\word_level\reconformer_word_model.pth"
    
    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    classes.append("NONE")
    num_classes = len(classes)
    print(f"Found {num_classes-1} word classes in INCLUDE50, plus 1 NONE class.")
    
    # Dataset splits (80-20)
    full_dataset = ISLAugmentedWordDataset(DATA_DIR, classes, is_train=True)
    
    # CRITICAL: Fix Data Leakage with deterministic split
    torch.manual_seed(42)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Disable augmentations for validation
    val_dataset.dataset.is_train = False
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    # Initialize the ISLSentenceReconformer with reduced capacity (d_model=128)
    D_MODEL = 128
    # Increased dropout to 0.5 to heavily regularize the Transformer's massive parameter count!
    model = ISLSentenceReconformer(num_classes=num_classes, d_model=D_MODEL, lstm_layers=1, dropout=0.5, use_rnn=True, freeze_base=True)
    
    # Load pretrained synthetic weights into base conformer ONLY if d_model matches original
    if os.path.exists(PRETRAINED_WEIGHTS) and D_MODEL == 128:
        print("Loading non-randomized pretrained synthetic weights...")
        model.load_base_weights(PRETRAINED_WEIGHTS)
    else:
        print(f"Training from scratch! (Bypassed pretrained weights because d_model is {D_MODEL}, not 256)")
        
    model = model.to(device)
    
    # Advanced Loss & Optimizer
    criterion = FocalLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    
    epochs = 60
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 
               'val_f1': [], 'val_prec': [], 'val_rec': []}
               
    best_f1 = 0.0
    
    for epoch in range(epochs):
        if epoch == 35:
            print(">>> EPOCH 35: Unfreezing Transformer and Input Dense for Full End-to-End Tuning! <<<")
            for param in model.parameters():
                param.requires_grad = True
                
        model.train()
        train_loss = 0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for x, lengths, labels in pbar:
            x, lengths, labels = x.to(device), lengths.to(device), labels.to(device)
            
            optimizer.zero_grad()
            # Model returns (B, T_pooled, num_classes_vocab) and lengths
            logits_seq, _ = model(x, lengths)
            
            # Since this is word-level classification, we mean-pool the temporal logits
            logits_pooled = logits_seq.mean(dim=1) # (B, num_classes)
            
            loss = criterion(logits_pooled, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = logits_pooled.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
        train_acc = accuracy_score(all_targets, all_preds)
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds_val = []
        all_targets_val = []
        
        with torch.no_grad():
            for x, lengths, labels in val_loader:
                x, lengths, labels = x.to(device), lengths.to(device), labels.to(device)
                logits_seq, _ = model(x, lengths)
                logits_pooled = logits_seq.mean(dim=1)
                
                loss = criterion(logits_pooled, labels)
                val_loss += loss.item()
                preds = logits_pooled.argmax(dim=-1)
                
                all_preds_val.extend(preds.cpu().numpy())
                all_targets_val.extend(labels.cpu().numpy())
                
        val_loss /= len(val_loader)
        val_acc = accuracy_score(all_targets_val, all_preds_val)
        
        # Step the scheduler
        scheduler.step()
        
        # Calculate F1, Precision, Recall
        prec, rec, f1, _ = precision_recall_fscore_support(all_targets_val, all_preds_val, average='macro', zero_division=0)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {f1:.4f} | LR: {current_lr:.6f}")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(f1)
        history['val_prec'].append(prec)
        history['val_rec'].append(rec)
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), SAVE_PATH)
            print("-> Saved best model!")
            
    # Generate Plots
    generate_plots(history)

def generate_plots(history):
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(18, 5))
    
    # 1. Loss Plot
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss', color='blue')
    plt.plot(epochs_range, history['val_loss'], label='Val Loss', color='red')
    plt.title("Loss over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    # 2. Accuracy Plot
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, history['train_acc'], label='Train Acc', color='blue')
    plt.plot(epochs_range, history['val_acc'], label='Val Acc', color='red')
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    
    # 3. F1, Precision, Recall (Combined Plot with Legends)
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, history['val_f1'], label='Validation F1 Score', color='purple', linestyle='-')
    plt.plot(epochs_range, history['val_prec'], label='Validation Precision', color='green', linestyle='--')
    plt.plot(epochs_range, history['val_rec'], label='Validation Recall', color='orange', linestyle='-.')
    plt.title("F1, Precision, and Recall")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.legend(loc='lower right')
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = r"d:\MookVani\Backend\models\word_level\reconformer_metrics.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Metrics plot saved to {plot_path}")

if __name__ == "__main__":
    train_reconformer()
