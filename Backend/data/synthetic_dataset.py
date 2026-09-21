import os
import glob
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class SyntheticSentenceDataset(Dataset):
    """
    Dynamically generates synthetic sentences by randomly sampling and stitching together
    word-level frame tensors. 
    """
    def __init__(self, word_dir, word_vocab, epoch_size=10000, min_words=2, max_words=10):
        self.word_vocab = word_vocab # word -> idx (1-indexed for CTC)
        self.epoch_size = epoch_size
        self.min_words = min_words
        self.max_words = max_words
        
        self.word_to_paths = {}
        # Glob all .pt files in the word directory
        for p in glob.glob(f"{word_dir}/*/*.pt"):
            label = os.path.basename(os.path.dirname(p))
            if label in self.word_vocab:
                if label not in self.word_to_paths:
                    self.word_to_paths[label] = []
                self.word_to_paths[label].append(p)
                
        self.available_words = list(self.word_to_paths.keys())
        if not self.available_words:
            raise ValueError(f"No word tensors found in {word_dir}")

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        # 1. Decide sequence length (number of words)
        num_words = random.randint(self.min_words, self.max_words)
        
        # 2. Pick random words
        chosen_words = random.choices(self.available_words, k=num_words)
        
        tensors = []
        target_indices = []
        
        for w in chosen_words:
            # 3. For each word, randomly pick one of its video instances (handles repetition awareness)
            path = random.choice(self.word_to_paths[w])
            tensor = torch.load(path, weights_only=True) # shape (T, 163)
            # Remove the batch dim if it's (1, T, 163) or similar
            if tensor.dim() == 3 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
                
            # Simulate video duration by repeating the frame
            # The Conformer downsamples by 2, so to avoid CTC loss errors,
            # we must ensure total input frames > target length * 2.
            # 5 to 15 frames per word is extremely safe and matches real video timing.
            if tensor.shape[0] == 1:
                duration = random.randint(5, 15)
                tensor = tensor.repeat(duration, 1)
                
            tensors.append(tensor)
            target_indices.append(self.word_vocab[w])
            
        # 4. Stitch along the time dimension
        stitched_tensor = torch.cat(tensors, dim=0) # (Total_T, 163)
        target = torch.tensor(target_indices, dtype=torch.long)
        
        return stitched_tensor, target

def synthetic_collate_fn(batch):
    inputs, targets = zip(*batch)
    in_lens = torch.tensor([x.shape[0] for x in inputs], dtype=torch.long)
    tgt_lens = torch.tensor([y.shape[0] for y in targets], dtype=torch.long)
    return (pad_sequence(inputs, batch_first=True, padding_value=0.0),
            pad_sequence(targets, batch_first=True, padding_value=0),
            in_lens, tgt_lens)

def build_synthetic_dataloaders(word_train_dir, class_to_idx, batch_size=8, num_workers=0, epoch_size=10000, max_words=10):
    word_vocab = {w: i + 1 for w, i in class_to_idx.items()}  # +1: 0 reserved for CTC blank
    
    # Train dataset (10,000 synthetic sentences per epoch)
    train_dataset = SyntheticSentenceDataset(word_train_dir, word_vocab, epoch_size=epoch_size, max_words=max_words)
    
    # Val dataset (1,000 synthetic sentences)
    val_dataset = SyntheticSentenceDataset(word_train_dir, word_vocab, epoch_size=1000, max_words=max_words)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=synthetic_collate_fn, num_workers=num_workers, pin_memory=True)
                              
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=synthetic_collate_fn, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, word_vocab
