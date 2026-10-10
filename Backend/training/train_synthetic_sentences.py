import os
import sys
import glob
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.isl_sentence_model import ISLSentenceReconformer

def compute_wer(ref, hyp):
    d = np.zeros((len(ref) + 1, len(hyp) + 1))
    for i in range(len(ref) + 1): d[i][0] = i
    for j in range(len(hyp) + 1): d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i-1] == hyp[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + 1)
    return d[len(ref)][len(hyp)] / max(len(ref), 1)

def compute_f1_prec_rec(ref, hyp):
    set_ref = set(ref)
    set_hyp = set(hyp)
    if len(set_hyp) == 0 and len(set_ref) == 0:
        return 1.0, 1.0, 1.0
    if len(set_hyp) == 0:
        return 0.0, 0.0, 0.0
    if len(set_ref) == 0:
        return 0.0, 0.0, 0.0
        
    true_pos = len(set_ref.intersection(set_hyp))
    precision = true_pos / len(set_hyp)
    recall = true_pos / len(set_ref)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
        
    return f1, precision, recall

class SyntheticSentenceDataset(Dataset):
    def __init__(self, data_dir, classes, max_words=10, is_train=True):
        self.data_dir = data_dir
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        self.class_files = {}
        for cls in classes:
            if cls == "NONE": continue
            cls_dir = os.path.join(data_dir, cls)
            if os.path.isdir(cls_dir):
                self.class_files[cls] = glob.glob(os.path.join(cls_dir, "*.pt"))
                
        self.max_words = max_words
        self.is_train = is_train
        
    def __len__(self):
        return 2000 if self.is_train else 500
        
    def __getitem__(self, idx):
        num_words = np.random.randint(1, self.max_words + 1)
        valid_classes = [c for c in self.classes if c != "NONE" and len(self.class_files.get(c, [])) > 0]
        
        sentence_classes = np.random.choice(valid_classes, num_words).tolist()
        
        tensors = []
        labels = []
        
        for i, cls in enumerate(sentence_classes):
            if i > 0 or np.random.rand() > 0.5:
                T_noise = np.random.randint(10, 30)
                noise_tensor = (torch.randn(T_noise, 163) * 0.1) 
                tensors.append(noise_tensor)
                labels.append(self.class_to_idx["NONE"])
                
            filepath = np.random.choice(self.class_files[cls])
            x = torch.load(filepath, weights_only=True)
            
            if self.is_train:
                import torch.nn.functional as F
                if np.random.rand() > 0.5:
                    speed_factor = np.random.uniform(0.7, 1.3)
                    new_T = max(2, int(x.size(0) * speed_factor))
                    x = x.unsqueeze(0).transpose(1, 2)
                    x = F.interpolate(x, size=new_T, mode='linear', align_corners=False)
                    x = x.transpose(1, 2).squeeze(0)
                
                # Heavy Gaussian Jitter randomly applied
                if np.random.rand() > 0.3:
                    x += torch.randn_like(x) * 0.05
                # Scale
                if np.random.rand() > 0.5:
                    x *= (torch.rand(1) * 0.2 + 0.90)
                
            tensors.append(x)
            labels.append(self.class_to_idx[cls])
            
        if np.random.rand() > 0.5:
            T_noise = np.random.randint(10, 30)
            noise_tensor = (torch.randn(T_noise, 163) * 0.1)
            tensors.append(noise_tensor)
            labels.append(self.class_to_idx["NONE"])
            
        full_tensor = torch.cat(tensors, dim=0)
        
        # Consistent Sentence-Level Spatial Shift
        # Applies the EXACT same positional shift across all frames in the sentence
        # so the relative poses remain perfectly aligned (prevents teleporting mid-sentence)
        if self.is_train and np.random.rand() > 0.5:
            sentence_shift = torch.randn(1, 163) * 0.03
            full_tensor += sentence_shift
            
        return full_tensor, full_tensor.shape[0], torch.tensor(labels, dtype=torch.long)

def collate_fn_sentence(batch):
    xs, lengths, labels = zip(*batch)
    max_len = max(lengths)
    batch_size = len(xs)
    
    x_padded = torch.zeros(batch_size, max_len, xs[0].shape[-1])
    for i, x in enumerate(xs):
        x_padded[i, :lengths[i], :] = x
        
    lengths = torch.tensor(lengths)
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    targets = torch.cat(labels)
    
    return x_padded, lengths, targets, target_lengths

def ctc_decode(log_probs, lengths, blank=263):
    preds = log_probs.argmax(dim=-1).transpose(0, 1).cpu().numpy()
    lengths = lengths.cpu().numpy()
    
    decoded = []
    for i in range(preds.shape[0]):
        pred_seq = preds[i, :lengths[i]]
        collapsed = []
        for j in range(len(pred_seq)):
            if pred_seq[j] != blank:
                if j == 0 or pred_seq[j] != pred_seq[j-1]:
                    collapsed.append(pred_seq[j])
        decoded.append(collapsed)
    return decoded

def get_targets(targets, target_lengths):
    targets = targets.cpu().numpy()
    target_lengths = target_lengths.cpu().numpy()
    true_seqs = []
    idx = 0
    for l in target_lengths:
        true_seqs.append(targets[idx:idx+l].tolist())
        idx += l
    return true_seqs

def train_synthetic_sentences():
    DATA_DIR = r"d:\MookVani\Backend\data\tensors_include50_word_level"
    MODEL_PATH = r"d:\MookVani\Backend\models\word_level\reconformer_word_model_final.pth"
    SAVE_PATH = r"d:\MookVani\Backend\models\sentence_level\reconformer_synthetic_sentence.pth"
    PLOT_PATH = r"d:\MookVani\Backend\models\sentence_level\ctc_metrics.png"
    
    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    classes.append("NONE")
    num_classes = len(classes)
    CTC_BLANK_ID = num_classes 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training CTC on {device} with {num_classes} classes (Blank ID = {CTC_BLANK_ID})")
    
    # Instantiate with use_rnn=True to utilize the fully expanded BiGRU -> Transformer -> CNN stack!
    model = ISLSentenceReconformer(num_classes=num_classes, d_model=128, lstm_layers=1, dropout=0.5, use_rnn=True).to(device)
    
    if os.path.exists(MODEL_PATH):
        print("Loading pre-trained weights to jumpstart CTC training (strict=False to allow architecture shifts)...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True), strict=False)
    
    train_dataset = SyntheticSentenceDataset(DATA_DIR, classes, max_words=10, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn_sentence, num_workers=2)
    
    criterion = nn.CTCLoss(blank=CTC_BLANK_ID, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    num_epochs = 100
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_loss = float('inf')
    
    history = {'train_loss': [], 'wer': [], 'acc': [], 'f1': []}
    
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_wer, train_acc, train_f1 = 0, 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        all_preds = []
        all_targets = []
        
        for x, lengths, targets, target_lengths in pbar:
            x, lengths, targets, target_lengths = x.to(device), lengths.to(device), targets.to(device), target_lengths.to(device)
            optimizer.zero_grad()
            
            log_probs, pooled_lengths = model(x, lengths)
            log_probs = log_probs.transpose(0, 1) 
            
            ctc_loss = criterion(log_probs, targets, pooled_lengths, target_lengths)
            
            probs = torch.exp(log_probs)
            non_blank_probs = probs[:, :, :-1].sum(dim=-1)
            sparsity_penalty = non_blank_probs.mean()
            
            loss = ctc_loss + 10.0 * sparsity_penalty
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Decode for metrics on the fly (no separate validation set)
            with torch.no_grad():
                decoded = ctc_decode(log_probs, pooled_lengths, blank=CTC_BLANK_ID)
                true_seqs = get_targets(targets, target_lengths)
                all_preds.extend(decoded)
                all_targets.extend(true_seqs)
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'ctc': f"{ctc_loss.item():.4f}"})
            
        train_loss /= len(train_loader)
        
        # Calculate Metrics on training set
        for ref, hyp in zip(all_targets, all_preds):
            train_wer += compute_wer(ref, hyp)
            train_acc += 1.0 if ref == hyp else 0.0
            f1, _, _ = compute_f1_prec_rec(ref, hyp)
            train_f1 += f1
            
        N = max(1, len(all_targets))
        train_wer /= N
        train_acc /= N
        train_f1 /= N
        
        history['train_loss'].append(train_loss)
        history['wer'].append(train_wer)
        history['acc'].append(train_acc)
        history['f1'].append(train_f1)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Loss: {train_loss:.4f} | WER: {train_wer:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print("-> Saved best CTC model!")
            
    # Plotting Metrics
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    epochs = range(1, num_epochs + 1)
    
    axs[0, 0].plot(epochs, history['train_loss'], label='Train Loss')
    axs[0, 0].set_title('CTC Loss (Lower is Better)')
    axs[0, 0].legend()
    
    axs[0, 1].plot(epochs, history['wer'], label='Train WER', color='red')
    axs[0, 1].set_title('Word Error Rate (Lower is Better)')
    axs[0, 1].legend()
    
    axs[1, 0].plot(epochs, history['acc'], label='Train Exact Match Accuracy', color='green')
    axs[1, 0].set_title('Sentence Accuracy (Higher is Better)')
    axs[1, 0].legend()
    
    axs[1, 1].plot(epochs, history['f1'], label='Train F1 Score')
    axs[1, 1].set_title('F1 Metrics (Higher is Better)')
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    print(f"Saved comprehensive metrics plot to {PLOT_PATH}")

if __name__ == "__main__":
    train_synthetic_sentences()
