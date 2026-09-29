import os
import glob
import random
import torch
import csv
import re
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

class SyntheticSentenceDataset(Dataset):
    """
    Dynamically generates synthetic sentences by stitching together word-level tensors.
    Now pulls REAL grammatical sentence structures from the CSLTR CSV!
    """
    def __init__(self, word_dir, word_vocab, csv_path, epoch_size=10000, is_train=True):
        self.word_vocab = word_vocab # word -> idx (1-indexed for CTC)
        self.epoch_size = epoch_size
        self.is_train = is_train
        
        # 1. Map valid word classes
        self.valid_words_map = {}
        for key in self.word_vocab.keys():
            sub_words = key.split('_')
            for sw in sub_words:
                sw = re.sub(r'[^\w\s\']', '', sw).lower().strip()
                if sw:
                    self.valid_words_map[sw] = key
        
        # 2. Extract valid sentences from CSV
        self.valid_sentences = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) < 2: continue
                text = row[1]
                cleaned = re.sub(r'[^\w\s\']', '', text).lower().strip()
                if not cleaned: continue
                
                words = cleaned.split()
                # If ALL words in the sentence exist in our 114 vocabulary map
                if len(words) > 1 and all(w in self.valid_words_map for w in words):
                    mapped_classes = [self.valid_words_map[w] for w in words]
                    self.valid_sentences.append(mapped_classes)
                    
        print(f"Synthesizer found {len(self.valid_sentences)} grammatically correct sentences to generate from!")
        
        # 3. Load available word tensors
        self.word_to_paths = {}
        for p in glob.glob(f"{word_dir}/*/*.pt"):
            label = os.path.basename(os.path.dirname(p))
            if label in self.word_vocab:
                if label not in self.word_to_paths:
                    self.word_to_paths[label] = []
                self.word_to_paths[label].append(p)
                
        # Remove any sentences that require a word we don't have tensors for
        self.valid_sentences = [
            s for s in self.valid_sentences if all(w in self.word_to_paths for w in s)
        ]
        if not self.valid_sentences:
            raise ValueError("No valid sentences could be constructed with the available word tensors!")

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        # 1. Pick a random valid grammatical sentence template
        chosen_sentence = random.choice(self.valid_sentences)
        
        # 1.5. Randomly inject "NONE" at the start, end, or middle
        # to teach the model to ignore background resting states!
        if 'NONE' in self.word_to_paths and random.random() < 0.6: # 60% chance to add NONE
            insert_idx = random.choice([0, len(chosen_sentence)])
            # we make a copy so we don't modify the original list in valid_sentences
            chosen_sentence = list(chosen_sentence)
            chosen_sentence.insert(insert_idx, 'NONE')
            
        tensors = []
        target_indices = []
        
        for w in chosen_sentence:
            # 2. Randomly pick one of its video instances (handles repetition awareness)
            path = random.choice(self.word_to_paths[w])
            tensor = torch.load(path, weights_only=True) # shape (T, 163)
            if tensor.dim() == 3 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
                
            # If the tensor is just a single frame (like a static image), duplicate it
            if tensor.shape[0] == 1:
                duration = random.randint(5, 15)
                tensor = tensor.repeat(duration, 1)
                
            # --- DYNAMIC ON-THE-FLY AUGMENTATION ---
            # 50% chance to temporally scale (speed up or slow down)
            if random.random() < 0.5 and tensor.shape[0] > 5:
                scale_factor = random.uniform(0.7, 1.3) # +/- 30% speed
                new_T = max(5, int(tensor.shape[0] * scale_factor))
                tensor_t = tensor.T.unsqueeze(0)
                tensor_t = F.interpolate(tensor_t, size=new_T, mode='linear', align_corners=False)
                tensor = tensor_t.squeeze(0).T
            
            # 50% chance to add slight spatial noise (simulates webcam blur/distance variance)
            if random.random() < 0.5:
                # Add random noise to coordinate axes
                noise = torch.randn_like(tensor) * random.uniform(0.005, 0.02)
                tensor = tensor + noise
                
            tensors.append(tensor)
            target_indices.append(self.word_vocab[w])
            
        # 3. Stitch along the time dimension
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

def build_synthetic_dataloaders(word_train_dir, word_val_dir, class_to_idx, csv_path, batch_size=8, num_workers=0, epoch_size=10000):
    word_vocab = {w: i + 1 for w, i in class_to_idx.items()}  # +1: 0 reserved for CTC blank
    
    # Train dataset (10,000 synthetic sentences per epoch)
    train_dataset = SyntheticSentenceDataset(word_train_dir, word_vocab, csv_path, epoch_size=epoch_size, is_train=True)
    
    # Val dataset (1,000 synthetic sentences, pulls from HOLDOUT folder!)
    val_dataset = SyntheticSentenceDataset(word_val_dir, word_vocab, csv_path, epoch_size=1000, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=synthetic_collate_fn, num_workers=num_workers, pin_memory=True)
                              
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=synthetic_collate_fn, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, word_vocab
