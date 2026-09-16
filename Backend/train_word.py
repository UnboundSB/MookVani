import os
import glob
import random
from collections import defaultdict
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.conformer import ISL_Conformer
from data.dataset import SaneSkeletalAugmentation, ISLWordLevelDataset

def word_level_collate(batch):
    # batch items are (tensor(1,163), label). Wrap each single-frame sample as a length-1 "sequence"
    inputs = torch.stack([x for x, _ in batch])          # (B, 1, 163)
    labels = torch.stack([y for _, y in batch])          # (B,)
    lengths = torch.ones(len(batch), dtype=torch.long)   # each sample is 1 frame
    targets = labels.unsqueeze(1) + 1  # +1 because CTC blank=0; word_vocab convention: labels are 0-indexed here
    tgt_lens = torch.ones(len(batch), dtype=torch.long)
    return inputs, targets, lengths, tgt_lens

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    WORD_DATA_DIR = "./data/tensors_word_level_163"
    SYN_DIR = "./data/tensors_word_level_163_synth"
    
    all_files = glob.glob(f"{WORD_DATA_DIR}/**/*.pt", recursive=True)
    if not all_files:
        print(f"No files found in {WORD_DATA_DIR} - run extract_features.py first.")
        return

    classes = sorted(set(os.path.basename(os.path.dirname(p)) for p in all_files))
    class_to_idx = {c: i for i, c in enumerate(classes)}

    class_to_files = defaultdict(list)
    for p in all_files:
        class_to_files[os.path.basename(os.path.dirname(p))].append(p)

    train_pairs, val_pairs = [], []
    for cls, files in class_to_files.items():
        random.shuffle(files)
        split_point = max(1, int(0.8 * len(files)))
        for f in files[:split_point]:
            train_pairs.append((f, cls))
        for f in files[split_point:]:
            val_pairs.append((f, cls))

    print(f"Real-data split: {len(train_pairs)} train / {len(val_pairs)} val")

    TARGET_COUNT = 30
    train_by_class = defaultdict(list)
    for f, cls in train_pairs:
        train_by_class[cls].append(f)

    balanced_train_pairs = list(train_pairs)
    os.makedirs(SYN_DIR, exist_ok=True)

    print("Balancing training set with augmentations...")
    for cls, files in train_by_class.items():
        current = len(files)
        i = 0
        while current < TARGET_COUNT and files:
            base_path = random.choice(files)
            base_tensor = torch.load(base_path)
            synth_aug = SaneSkeletalAugmentation(apply_prob=1.0)
            synth_tensor = synth_aug(base_tensor)
            synth_path = os.path.join(SYN_DIR, f"{cls}__synth_{i}.pt")
            torch.save(synth_tensor, synth_path)
            balanced_train_pairs.append((synth_path, cls))
            current += 1
            i += 1

    augmenter = SaneSkeletalAugmentation(max_rot_degrees=10.0, max_shift=0.05, noise_std=0.01)
    train_dataset = ISLWordLevelDataset(balanced_train_pairs, class_to_idx, transform=augmenter)
    val_dataset = ISLWordLevelDataset(val_pairs, class_to_idx, transform=None)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=word_level_collate, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=word_level_collate, num_workers=2, pin_memory=True)

    num_classes_word = len(classes)
    
    model = ISL_Conformer(input_dim=163, num_classes=num_classes_word).to(device)
    criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    EPOCHS = 20
    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
        model.train()
        train_loss, valid_batches = 0.0, 0

        for batch_inputs, batch_targets, in_lens, tgt_lens in tqdm(train_loader, desc="Training"):
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            in_lens = in_lens.to(device)

            optimizer.zero_grad()
            log_probs, pooled_lens = model(batch_inputs, in_lens)
            log_probs = log_probs.permute(1, 0, 2)
            loss = criterion(log_probs, batch_targets, pooled_lens.clamp(min=1), tgt_lens.to(device))

            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                train_loss += loss.item()
                valid_batches += 1

        avg_train_loss = train_loss / max(1, valid_batches)

        model.eval()
        val_loss, val_valid_batches = 0.0, 0
        with torch.no_grad():
            for batch_inputs, batch_targets, in_lens, tgt_lens in val_loader:
                batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
                in_lens = in_lens.to(device)
                log_probs, pooled_lens = model(batch_inputs, in_lens)
                log_probs = log_probs.permute(1, 0, 2)
                loss = criterion(log_probs, batch_targets, pooled_lens.clamp(min=1), tgt_lens.to(device))
                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss += loss.item()
                    val_valid_batches += 1

        avg_val_loss = val_loss / max(1, val_valid_batches)
        scheduler.step(avg_val_loss)
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_isl_conformer_word.pth")
            print("New best model saved.")

if __name__ == "__main__":
    train()
