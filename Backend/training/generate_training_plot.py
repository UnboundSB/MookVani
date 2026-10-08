import re
import matplotlib.pyplot as plt
import os

log_file = r"C:\Users\UnboundSB\.gemini\antigravity-ide\brain\e5352803-5ec5-49d4-b73a-3ddf13c0a033\.system_generated\tasks\task-506.log"

train_loss, train_acc = [], []
val_loss, val_acc = [], []
epochs = []

with open(log_file, 'r', encoding='utf-8') as f:
    for line in f:
        # Looking for lines like: Epoch 1: Train Loss: 0.2345, Train Acc: 0.9876 | Val Loss: 0.1234, Val Acc: 0.9999
        match = re.search(r'Epoch (\d+): Train Loss: ([\d.]+), Train Acc: ([\d.]+) \| Val Loss: ([\d.]+), Val Acc: ([\d.]+)', line)
        if match:
            epochs.append(int(match.group(1)))
            train_loss.append(float(match.group(2)))
            train_acc.append(float(match.group(3)))
            val_loss.append(float(match.group(4)))
            val_acc.append(float(match.group(5)))

if epochs:
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss', marker='o', color='skyblue')
    plt.plot(epochs, val_loss, label='Val Loss', marker='s', color='coral')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Train Accuracy', marker='o', color='skyblue')
    plt.plot(epochs, val_acc, label='Val Accuracy', marker='s', color='coral')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    output_path = r"C:\Users\UnboundSB\.gemini\antigravity-ide\brain\e5352803-5ec5-49d4-b73a-3ddf13c0a033\synthetic_training_plots.png"
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved successfully to {output_path}")
else:
    print("Could not find epoch metrics in the log file.")
