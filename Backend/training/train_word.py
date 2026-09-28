import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from models.isl_conformer import ISL_Conformer
from data.datasets import build_word_dataloaders

def train_word_model(
    train_dir,
    val_dir=None,
    model_dir="models",
    plot_dir="models/plots",
    epochs=50,
    batch_size=32,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    train_loader, val_loader, class_to_idx = build_word_dataloaders(train_dir, val_dir, batch_size=batch_size)
    if train_loader is None:
        print("Dataset directories not found.")
        return

    num_classes = len(class_to_idx)
    print(f"Training on {num_classes} classes.")

    with open(os.path.join(model_dir, "word_class_to_idx.json"), "w") as f:
        json.dump(class_to_idx, f, indent=4)

    # Word-level model: ISL_Conformer outputting logits for single-frame classification
    model = ISL_Conformer(num_classes=num_classes).to(device)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses, val_accuracies = [], [], []
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets, in_lens in train_loader:
            inputs, targets, in_lens = inputs.to(device), targets.to(device), in_lens.to(device)
            optimizer.zero_grad()
            
            # Forward pass through Conformer
            log_probs, lengths_pooled = model(inputs, in_lens) # (B, T, num_classes+1)
            
            # Mean pool across time dimension
            # Create a mask so we only pool over valid time steps
            max_t = log_probs.size(1)
            mask = torch.arange(max_t, device=device).unsqueeze(0) < lengths_pooled.unsqueeze(1)
            
            # Zero out invalid timesteps
            log_probs = log_probs * mask.unsqueeze(-1)
            
            # Sum and divide by actual lengths to get mean
            outputs = log_probs.sum(dim=1) / lengths_pooled.unsqueeze(-1).clamp(min=1)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets, in_lens in val_loader:
                inputs, targets, in_lens = inputs.to(device), targets.to(device), in_lens.to(device)
                
                log_probs, lengths_pooled = model(inputs, in_lens)
                
                max_t = log_probs.size(1)
                mask = torch.arange(max_t, device=device).unsqueeze(0) < lengths_pooled.unsqueeze(1)
                log_probs = log_probs * mask.unsqueeze(-1)
                outputs = log_probs.sum(dim=1) / lengths_pooled.unsqueeze(-1).clamp(min=1)
                
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(model_dir, "best_word_model.pth"))
            print(f"  -> Saved new best model (Acc: {val_acc:.2f}%)")

    # Plot metrics
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss per Epoch')

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy', color='green')
    plt.legend()
    plt.title('Validation Accuracy')

    plt.savefig(os.path.join(plot_dir, "word_training_metrics.png"))
    plt.close()
    
    print("Training complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    train_word_model(args.train_dir, args.val_dir, epochs=args.epochs, batch_size=args.batch_size)
