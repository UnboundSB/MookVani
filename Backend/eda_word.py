import os
import glob
import random
from collections import Counter
import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_skeleton(ax, coords, title):
    """
    Plots a 53-point skeleton from (53, 3) coordinates.
    Indices:
    0-20: Left Hand
    21-41: Right Hand
    42-45: Arms (Pose subset)
    46-52: Face subset
    """
    x = coords[:, 0].numpy()
    y = coords[:, 1].numpy()
    
    # Scatter points with distinct colors
    ax.scatter(x[0:21], y[0:21], c='red', s=10, label='Left Hand')
    ax.scatter(x[21:42], y[21:42], c='blue', s=10, label='Right Hand')
    ax.scatter(x[42:46], y[42:46], c='green', s=25, marker='X', label='Arms/Pose')
    ax.scatter(x[46:53], y[46:53], c='orange', s=15, marker='^', label='Face')
    
    # Invert Y axis because image coordinates (like MediaPipe) typically have Y increasing downwards
    ax.invert_yaxis()
    ax.set_title(title, fontsize=8)
    ax.axis('off')

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    WORD_DATA_DIR = os.path.join(BASE_DIR, "data", "tensors_word_level_163")
    
    all_files = glob.glob(f"{WORD_DATA_DIR}/**/*.pt", recursive=True)
    if not all_files:
        print(f"No files found in {WORD_DATA_DIR}")
        return
        
    print(f"Found {len(all_files)} total tensor files.")
    
    # 1. Class Distribution Analysis
    class_counts = Counter()
    class_to_files = {}
    
    for p in all_files:
        cls_name = os.path.basename(os.path.dirname(p))
        class_counts[cls_name] += 1
        if cls_name not in class_to_files:
            class_to_files[cls_name] = []
        class_to_files[cls_name].append(p)
        
    classes = sorted(class_counts.keys())
    num_classes = len(classes)
    
    print("\n" + "="*50)
    print(f"CLASS DISTRIBUTION ({num_classes} Classes)")
    print("="*50)
    min_count = float('inf')
    max_count = 0
    min_class, max_class = "", ""
    
    for cls in classes:
        count = class_counts[cls]
        print(f" - {cls}: {count} samples")
        if count < min_count:
            min_count = count
            min_class = cls
        if count > max_count:
            max_count = count
            max_class = cls
            
    print("-" * 50)
    print(f"Smallest Class: {min_class} ({min_count} samples)")
    print(f"Largest Class:  {max_class} ({max_count} samples)")
    print(f"Average Samples per class: {len(all_files) / num_classes:.2f}")
    
    # 2. Coordinate Statistics
    print("\n" + "="*50)
    print("COORDINATE STATISTICS (X, Y, Z)")
    print("="*50)
    
    # Randomly sample 100 files to compute global stats quickly
    sample_files = random.sample(all_files, min(100, len(all_files)))
    all_coords = []
    for f in sample_files:
        tensor = torch.load(f, weights_only=True) # shape (1, 163)
        coords = tensor[0, :159].view(53, 3)
        all_coords.append(coords)
        
    all_coords = torch.cat(all_coords, dim=0) # shape (100*53, 3)
    mean_coords = all_coords.mean(dim=0)
    std_coords = all_coords.std(dim=0)
    min_coords, _ = all_coords.min(dim=0)
    max_coords, _ = all_coords.max(dim=0)
    
    print(f"Global Mean (X, Y, Z): {mean_coords.tolist()}")
    print(f"Global Std  (X, Y, Z): {std_coords.tolist()}")
    print(f"Global Min  (X, Y, Z): {min_coords.tolist()}")
    print(f"Global Max  (X, Y, Z): {max_coords.tolist()}")
    
    # 3. Plotting Class Distribution Bar Chart
    plt.figure(figsize=(14, 6))
    plt.bar(classes, [class_counts[c] for c in classes], color='skyblue')
    plt.xticks(rotation=90, fontsize=8)
    plt.xlabel('Sign Class')
    plt.ylabel('Number of Samples')
    plt.title('Word-Level Dataset Class Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "eda_class_distribution.png"), dpi=300)
    print("\nSaved -> eda_class_distribution.png")
    
    # 4. Plotting Skeletons Grid (1 random sample per class)
    cols = 8
    rows = (num_classes + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()
    
    for i, cls in enumerate(classes):
        ax = axes[i]
        random_file = random.choice(class_to_files[cls])
        tensor = torch.load(random_file, weights_only=True)
        coords = tensor[0, :159].view(53, 3)
        plot_skeleton(ax, coords, cls)
        
    # Hide empty subplots
    for i in range(num_classes, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "eda_skeletons_grid.png"), dpi=300)
    print("Saved -> eda_skeletons_grid.png")
    print("="*50)
    print("EDA Complete! Check the terminal output above and the PNG files.")

if __name__ == "__main__":
    main()
