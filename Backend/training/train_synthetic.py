import os
import json
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.isl_conformer import ISL_Conformer
from models.isl_sentence_model import ISL_Sentence_Model
from data.synthetic_dataset import build_synthetic_dataloaders
from utils.metrics import compute_sentence_metrics

def clear_garbage():
    gc.collect()
    torch.cuda.empty_cache()

def train_synthetic_model(
    word_train_dir,
    word_class_json="models/word_class_to_idx.json",
    model_dir="models",
    plot_dir="models/plots",
    epochs=20,
    batch_size=8,
    lr=1e-4,
    epoch_size=2000,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    if not os.path.exists(word_class_json):
        print(f"Error: {word_class_json} missing. Run train_word.py first.")
        return

    with open(word_class_json, "r") as f:
        class_to_idx = json.load(f)

    # 1. Build loaders
    print("Building Synthetic Sentence DataLoaders...")
    train_loader, val_loader, word_vocab = build_synthetic_dataloaders(
        word_train_dir, class_to_idx, batch_size=batch_size, epoch_size=epoch_size, max_words=10
    )

    num_classes = len(word_vocab)
    print(f"Training synthetic sentence model. Vocab size (incl blank): {num_classes + 1}")

    # 2. Model & Loss
    model = ISL_Sentence_Model(num_classes=num_classes, d_model=256, lstm_layers=1, dropout=0.5, freeze_base=False).to(device)
    
    # Load pre-trained word-level weights to jumpstart the encoder
    word_weights_path = os.path.join(model_dir, "best_word_model.pth")
    if os.path.exists(word_weights_path):
        model.load_base_weights(word_weights_path)
    else:
        print(f"Warning: No pre-trained word model found at {word_weights_path}.")

    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    train_losses, val_losses, val_wers = [], [], []
    best_val_wer = float('inf')

    # 3. Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_inputs, batch_targets, in_lens, tgt_lens in train_loader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            in_lens, tgt_lens = in_lens.to(device), tgt_lens.to(device)
            
            optimizer.zero_grad()
            log_probs, pooled_lens = model(batch_inputs, in_lens)
            
            log_probs_t = log_probs.transpose(0, 1)
            loss = ctc_loss_fn(log_probs_t, batch_targets, pooled_lens, tgt_lens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Clear garbage after train loop to prevent memory fragmentation
        clear_garbage()

        # 4. Validation
        val_wer, val_acc, examples = compute_sentence_metrics(model, val_loader, device)
        
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for batch_inputs, batch_targets, in_lens, tgt_lens in val_loader:
                batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
                in_lens, tgt_lens = in_lens.to(device), tgt_lens.to(device)
                
                log_probs, pooled_lens = model(batch_inputs, in_lens)
                log_probs_t = log_probs.transpose(0, 1)
                loss = ctc_loss_fn(log_probs_t, batch_targets, pooled_lens, tgt_lens)
                v_loss += loss.item()
                
        avg_val_loss = v_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_wers.append(val_wer)
        
        scheduler.step(avg_val_loss)
        
        # Clear garbage after val loop to prevent memory fragmentation
        clear_garbage()

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | WER: {val_wer:.4f} | Exact Match: {val_acc:.4f}")
        if (epoch + 1) % 1 == 0 and examples:
            print(f"  Example Ref: {examples[0][0]}")
            print(f"  Example Hyp: {examples[0][1]}")

        if val_wer < best_val_wer:
            best_val_wer = val_wer
            torch.save(model.state_dict(), os.path.join(model_dir, "best_synthetic_model.pth"))
            print(f"  -> Saved new best synthetic model (WER: {val_wer:.4f})")

    # 5. Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('CTC Loss (Synthetic)')

    plt.subplot(1, 2, 2)
    plt.plot(val_wers, label='Val WER', color='orange')
    plt.legend()
    plt.title('Word Error Rate (Synthetic)')

    plt.savefig(os.path.join(plot_dir, "synthetic_training_metrics.png"))
    plt.close()
    
    print("Synthetic pre-training complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--word_train_dir", type=str, default="data/tensors_word_level_163_train")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epoch_size", type=int, default=2000)
    args = parser.parse_args()
    
    train_synthetic_model(args.word_train_dir, epochs=args.epochs, batch_size=args.batch_size, epoch_size=args.epoch_size)
