import os
import sys
import glob
import json
import argparse
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
    if len(set_hyp) == 0 or len(set_ref) == 0:
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
    def __init__(self, data_dir, classes, samples_per_epoch=1000, max_words=10, is_train=True):
        self.data_dir = data_dir
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.samples_per_epoch = samples_per_epoch
        self.max_words = max_words
        self.is_train = is_train
        
        self.class_files = {}
        for cls in classes:
            if cls == "NONE":
                continue
            cls_dir = os.path.join(data_dir, cls)
            if os.path.isdir(cls_dir):
                self.class_files[cls] = glob.glob(os.path.join(cls_dir, "*.pt"))
                
    def __len__(self):
        return self.samples_per_epoch
        
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

def parse_args():
    parser = argparse.ArgumentParser(description="Stage-oriented CTC Training on Synthetic ISL Sentences")
    parser.add_argument("--data_dir", type=str, default=r"d:\MookVani\Backend\data\tensors_include50_word_level", help="Path to pose tensor directory")
    parser.add_argument("--pretrained_word_model", type=str, default=r"d:\MookVani\Backend\models\word_level\reconformer_word_model_final.pth", help="Pretrained word model checkpoint")
    parser.add_argument("--output_dir", type=str, default=r"d:\MookVani\Backend\models\islsentence_synthetic", help="Output directory for model and checkpoints")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for this training stage")
    parser.add_argument("--samples_per_epoch", type=int, default=1000, help="Number of synthetic sentences generated per epoch")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (safe for thermal/memory)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of DataLoader workers")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint in output_dir")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Explicit path to a checkpoint to resume from")
    return parser.parse_args()

def save_plots(history, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)
    num_epochs = len(history['train_loss'])
    epochs = range(1, num_epochs + 1)
    
    # Comprehensive 4-panel metrics plot
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    axs[0, 0].plot(epochs, history['train_loss'], label='CTC Loss', color='#1f77b4', lw=2)
    axs[0, 0].set_title('CTC Loss (Lower is Better)')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)
    axs[0, 0].legend()
    
    axs[0, 1].plot(epochs, history['wer'], label='Word Error Rate (WER)', color='#d62728', lw=2)
    axs[0, 1].set_title('Word Error Rate (Lower is Better)')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)
    axs[0, 1].legend()
    
    axs[1, 0].plot(epochs, history['acc'], label='Sentence Accuracy', color='#2ca02c', lw=2)
    axs[1, 0].set_title('Exact Match Sentence Accuracy (Higher is Better)')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].grid(True, linestyle='--', alpha=0.6)
    axs[1, 0].legend()
    
    axs[1, 1].plot(epochs, history['f1'], label='F1 Score', color='#9467bd', lw=2)
    axs[1, 1].set_title('Sequence F1 Score (Higher is Better)')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].grid(True, linestyle='--', alpha=0.6)
    axs[1, 1].legend()
    
    plt.tight_layout()
    ctc_plot_path = os.path.join(plots_dir, 'ctc_metrics.png')
    plt.savefig(ctc_plot_path, dpi=150)
    plt.close()
    
    # Loss vs WER comparison plot
    fig2, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(epochs, history['train_loss'], color='#1f77b4', label='Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='#1f77b4')
    ax2 = ax1.twinx()
    ax2.plot(epochs, history['wer'], color='#d62728', label='WER')
    ax2.set_ylabel('WER', color='#d62728')
    plt.title('Loss and WER Convergence Progression')
    plt.tight_layout()
    loss_wer_path = os.path.join(plots_dir, 'loss_and_wer.png')
    plt.savefig(loss_wer_path, dpi=150)
    plt.close()
    print(f"Saved plots to {plots_dir}")

def train_synthetic_sentences():
    args = parse_args()
    
    DATA_DIR = args.data_dir
    OUTPUT_DIR = args.output_dir
    PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "reconformer_synthetic_sentence_checkpoint.pth")
    BEST_MODEL_FILE = os.path.join(OUTPUT_DIR, "reconformer_synthetic_sentence_best.pth")
    FINAL_MODEL_FILE = os.path.join(OUTPUT_DIR, "reconformer_synthetic_sentence_final.pth")
    CLASSES_FILE = os.path.join(OUTPUT_DIR, "classes.json")
    HISTORY_FILE = os.path.join(OUTPUT_DIR, "history.json")
    
    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    classes.append("NONE")
    num_classes = len(classes)
    CTC_BLANK_ID = num_classes 
    
    with open(CLASSES_FILE, "w") as f:
        json.dump({"classes": classes, "num_classes": num_classes, "ctc_blank_id": CTC_BLANK_ID}, f, indent=2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print(f"ISL SENTENCE RECONFORMER - STAGE TRAINING")
    print(f"Device: {device}")
    print(f"Classes: {num_classes} (CTC Blank ID: {CTC_BLANK_ID})")
    print(f"Target Directory: {OUTPUT_DIR}")
    print(f"Plots Directory:  {PLOTS_DIR}")
    print(f"Stage Epochs: {args.epochs} | Samples/Epoch: {args.samples_per_epoch} | Batch: {args.batch_size}")
    print("=" * 60)
    
    model = ISLSentenceReconformer(num_classes=num_classes, d_model=128, lstm_layers=1, dropout=0.5, use_rnn=True).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    
    start_epoch = 0
    total_epochs = args.epochs
    best_loss = float('inf')
    history = {'train_loss': [], 'wer': [], 'acc': [], 'f1': []}
    
    # Determine checkpoint loading
    resume_path = None
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        resume_path = args.checkpoint_path
    elif args.resume and os.path.exists(CHECKPOINT_FILE):
        resume_path = CHECKPOINT_FILE
        
    if resume_path:
        print(f"Resuming training from checkpoint: {resume_path}...")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
        start_epoch = ckpt.get('epoch', 0)
        total_epochs = start_epoch + args.epochs
        best_loss = ckpt.get('best_loss', float('inf'))
        history = ckpt.get('history', history)
        print(f"Resumed at epoch {start_epoch}. Running {args.epochs} stage epochs to epoch {total_epochs}.")
    elif os.path.exists(args.pretrained_word_model):
        print(f"Loading pretrained word weights from {args.pretrained_word_model} (strict=False)...")
        model.load_state_dict(torch.load(args.pretrained_word_model, map_location=device, weights_only=True), strict=False)
        
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    train_dataset = SyntheticSentenceDataset(DATA_DIR, classes, samples_per_epoch=args.samples_per_epoch, max_words=10, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_sentence, num_workers=args.num_workers)
    
    criterion = nn.CTCLoss(blank=CTC_BLANK_ID, zero_infinity=True)
    
    for epoch in range(start_epoch, total_epochs):
        model.train()
        train_loss, train_wer, train_acc, train_f1 = 0, 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Stage Epoch {epoch+1}/{total_epochs} [Train]")
        
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
            
            with torch.no_grad():
                decoded = ctc_decode(log_probs, pooled_lengths, blank=CTC_BLANK_ID)
                true_seqs = get_targets(targets, target_lengths)
                all_preds.extend(decoded)
                all_targets.extend(true_seqs)
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'ctc': f"{ctc_loss.item():.4f}"})
            
        train_loss /= len(train_loader)
        
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
        
        print(f"Epoch {epoch+1}/{total_epochs} Complete: Loss: {train_loss:.4f} | WER: {train_wer:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        
        # Save modular state checkpoint every single epoch (guarantees safe resumption)
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': best_loss,
            'history': history,
            'classes': classes,
            'num_classes': num_classes,
            'ctc_blank_id': CTC_BLANK_ID,
            'samples_per_epoch': args.samples_per_epoch
        }
        torch.save(checkpoint_data, CHECKPOINT_FILE)
        
        if train_loss < best_loss:
            best_loss = train_loss
            checkpoint_data['best_loss'] = best_loss
            torch.save(model.state_dict(), BEST_MODEL_FILE)
            print(f"-> Saved new best model to {BEST_MODEL_FILE} (Loss: {best_loss:.4f})")
            
        # Update history JSON and plots periodically
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
            
        if (epoch + 1) % 2 == 0 or (epoch + 1) == total_epochs:
            save_plots(history, PLOTS_DIR)
            
    # Save final stage model
    torch.save(model.state_dict(), FINAL_MODEL_FILE)
    save_plots(history, PLOTS_DIR)
    print("\n" + "=" * 60)
    print("STAGE TRAINING COMPLETE!")
    print(f"Final Weights: {FINAL_MODEL_FILE}")
    print(f"Best Weights:  {BEST_MODEL_FILE}")
    print(f"Checkpoint:    {CHECKPOINT_FILE}")
    print(f"Metrics Plots: {PLOTS_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    train_synthetic_sentences()
