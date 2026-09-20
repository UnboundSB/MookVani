import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from models.isl_conformer import ISL_Conformer
from data.datasets import build_sentence_dataloaders
from utils.gloss_utils import load_sentence_to_words
from utils.metrics import compute_sentence_metrics

def train_sentence_model(
    sentence_dir,
    gloss_csv_path,
    word_frames_dir,
    word_class_json="models/word_class_to_idx.json",
    model_dir="models",
    plot_dir="models/plots",
    epochs=100,
    batch_size=8,
    lr=1e-4,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    if not os.path.exists(word_class_json):
        print(f"Error: {word_class_json} missing. Run train_word.py first.")
        return

    with open(word_class_json, "r") as f:
        class_to_idx = json.load(f)

    # 1. Load mappings
    sentence_to_words, _, _ = load_sentence_to_words(gloss_csv_path, word_frames_dir)
    
    # 2. Build loaders
    train_loader, val_loader, word_vocab = build_sentence_dataloaders(
        sentence_dir, sentence_to_words, class_to_idx, batch_size=batch_size
    )

    if not train_loader:
        print("Not enough sentence data to train.")
        return

    num_classes = len(word_vocab) + 1  # +1 for CTC Blank (0)
    print(f"Training sentence model. Vocab size (incl blank): {num_classes}")

    # 3. Model & Loss
    model = ISL_Conformer(num_classes=num_classes).to(device)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_losses, val_losses, val_wers = [], [], []
    best_val_wer = float('inf')

    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_inputs, batch_targets, in_lens, tgt_lens in train_loader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            in_lens, tgt_lens = in_lens.to(device), tgt_lens.to(device)
            
            optimizer.zero_grad()
            log_probs, pooled_lens = model(batch_inputs, in_lens)
            
            # log_probs is (B, T, C). CTCLoss expects (T, B, C)
            log_probs_t = log_probs.transpose(0, 1)
            
            loss = ctc_loss_fn(log_probs_t, batch_targets, pooled_lens, tgt_lens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 5. Validation
        val_wer, val_acc, examples = compute_sentence_metrics(model, val_loader, device)
        
        # Compute val CTC loss as well
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
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | WER: {val_wer:.4f} | Exact Match: {val_acc:.4f}")
        if (epoch + 1) % 10 == 0 and examples:
            print(f"  Example Ref: {examples[0][0]}")
            print(f"  Example Hyp: {examples[0][1]}")

        if val_wer < best_val_wer:
            best_val_wer = val_wer
            torch.save(model.state_dict(), os.path.join(model_dir, "best_sentence_model.pth"))
            print(f"  -> Saved new best model (WER: {val_wer:.4f})")

    # 6. Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('CTC Loss')

    plt.subplot(1, 2, 2)
    plt.plot(val_wers, label='Val WER (lower is better)', color='orange')
    plt.legend()
    plt.title('Word Error Rate')

    plt.savefig(os.path.join(plot_dir, "sentence_training_metrics.png"))
    plt.close()
    
    print("Sentence-level training complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence_dir", type=str, required=True)
    parser.add_argument("--gloss_csv_path", type=str, required=True)
    parser.add_argument("--word_frames_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    
    train_sentence_model(args.sentence_dir, args.gloss_csv_path, args.word_frames_dir, epochs=args.epochs, batch_size=args.batch_size)
