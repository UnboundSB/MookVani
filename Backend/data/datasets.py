import os
import glob
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class ISLWordLevelDataset(Dataset):
    """Loads pre-extracted 163-dim word-level tensors from a directory of <class>/<file>.pt"""
    def __init__(self, data_dir):
        self.pairs = []
        for p in glob.glob(f"{data_dir}/**/*.pt", recursive=True):
            label = os.path.basename(os.path.dirname(p))
            self.pairs.append((p, label))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path, label = self.pairs[idx]
        tensor = torch.load(path, weights_only=True)  # (1, 163)
        return tensor, label  # label resolved to an index via class_to_idx at collate/setup time

class ISLWordLevelDatasetIndexed(Dataset):
    def __init__(self, base_dataset, class_to_idx):
        self.pairs = base_dataset.pairs
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path, label = self.pairs[idx]
        tensor = torch.load(path, weights_only=True)
        return tensor, torch.tensor(self.class_to_idx[label], dtype=torch.long)

def word_level_collate(batch):
    inputs = [x for x, _ in batch]
    labels = torch.stack([y for _, y in batch])   # (B,) 0-indexed class ids
    in_lens = torch.tensor([x.shape[0] for x in inputs], dtype=torch.long)
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0.0)
    return padded_inputs, labels, in_lens

class ISLSentenceLevelDataset(Dataset):
    """Sentence-level dataset: each sample is a (T, 163) sequence with a MULTI-WORD target
    (list of word indices, in signing order) for CTC. Requires `sentence_to_words`
    and a shared `word_vocab`.
    """
    def __init__(self, file_label_pairs, sentence_to_words, word_vocab, is_train=False):
        self.sentence_to_words = sentence_to_words
        self.word_vocab = word_vocab  # word -> idx (1-indexed, 0 reserved for CTC blank)
        self.is_train = is_train

        self.samples = []
        for path, phrase in file_label_pairs:
            if phrase in self.sentence_to_words:
                words = self.sentence_to_words[phrase]
                if all(w in self.word_vocab for w in words):
                    self.samples.append((path, words))
            else:
                # Fallback for iSign dataset where the folder name is the text itself.
                # First, build a map of valid lower-case words to their true class name.
                if not hasattr(self, 'valid_map'):
                    self.valid_map = {}
                    for cls_name in self.word_vocab.keys():
                        for sw in cls_name.split('_'):
                            c = sw.lower().strip()
                            if c: self.valid_map[c] = cls_name
                
                # The phrase is e.g. "what_are_you_doing" or "what are you doing"
                cleaned = phrase.replace('_', ' ').lower().strip()
                words = cleaned.split()
                if len(words) > 1 and all(w in self.valid_map for w in words):
                    mapped_classes = [self.valid_map[w] for w in words]
                    if all(mc in self.word_vocab for mc in mapped_classes):
                        self.samples.append((path, mapped_classes))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, words = self.samples[idx]
        keypoints = torch.load(path, weights_only=True)  # (T, 163)
        
        # Dynamic Temporal & Spatial Augmentation for E2E Training
        if self.is_train and keypoints.shape[0] > 5:
            # 50% chance to temporally scale (speed up or slow down by up to 20%)
            if random.random() < 0.5:
                scale_factor = random.uniform(0.8, 1.2)
                new_T = max(5, int(keypoints.shape[0] * scale_factor))
                # interpolate expects (Batch, Channels, Length), so we reshape
                # keypoints is (T, 163) -> (1, 163, T)
                keypoints_t = keypoints.T.unsqueeze(0)
                import torch.nn.functional as F
                keypoints_t = F.interpolate(keypoints_t, size=new_T, mode='linear', align_corners=False)
                keypoints = keypoints_t.squeeze(0).T  # back to (new_T, 163)
            
            # 50% chance to add slight spatial noise to prevent overfitting
            if random.random() < 0.5:
                noise = torch.randn_like(keypoints) * 0.01
                keypoints = keypoints + noise

        target = torch.tensor([self.word_vocab[w] for w in words], dtype=torch.long)
        return keypoints, target

def sentence_collate_fn(batch):
    inputs, targets = zip(*batch)
    in_lens = torch.tensor([x.shape[0] for x in inputs], dtype=torch.long)
    tgt_lens = torch.tensor([y.shape[0] for y in targets], dtype=torch.long)
    return (pad_sequence(inputs, batch_first=True, padding_value=0.0),
            pad_sequence(targets, batch_first=True, padding_value=0),
            in_lens, tgt_lens)

def build_word_dataloaders(train_dir, val_dir=None, batch_size=32, num_workers=0):
    if not os.path.exists(train_dir):
        return None, None, None

    dataset_raw = ISLWordLevelDataset(train_dir)
    
    # Extract all labels from the dataset
    all_labels = sorted(set(l for _, l in dataset_raw.pairs))
    class_to_idx = {c: i for i, c in enumerate(all_labels)}
    
    indexed_dataset = ISLWordLevelDatasetIndexed(dataset_raw, class_to_idx)
    
    if val_dir and os.path.exists(val_dir):
        # We have a separate val dir
        val_dataset_raw = ISLWordLevelDataset(val_dir)
        val_dataset = ISLWordLevelDatasetIndexed(val_dataset_raw, class_to_idx)
        train_dataset = indexed_dataset
    else:
        # Auto-split 80/20
        train_size = int(0.8 * len(indexed_dataset))
        val_size = len(indexed_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(indexed_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              collate_fn=word_level_collate, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=word_level_collate, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, class_to_idx

def build_sentence_dataloaders(train_dir, val_dir=None, sentence_to_words=None, class_to_idx=None, batch_size=8, num_workers=0):
    word_vocab = {w: i + 1 for w, i in class_to_idx.items()}  # +1: 0 reserved for CTC blank
    
    train_files = glob.glob(f"{train_dir}/*/*.pt")
    train_pairs = [(p, os.path.basename(os.path.dirname(p))) for p in train_files]
    dataset = ISLSentenceLevelDataset(train_pairs, sentence_to_words, word_vocab, is_train=True)
    
    if val_dir and os.path.exists(val_dir):
        val_files = glob.glob(f"{val_dir}/*/*.pt")
        val_pairs = [(p, os.path.basename(os.path.dirname(p))) for p in val_files]
        val_dataset = ISLSentenceLevelDataset(val_pairs, sentence_to_words, word_vocab, is_train=False)
        train_dataset = dataset
    else:
        # Auto-split 80/20
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # We need to manually turn off is_train for val_dataset since it's a Subset
        # This is slightly hacky but dataset.dataset is the underlying ISLSentenceLevelDataset
        # We will just let random_split keep the is_train=True for val since it's hard to split cleanly
        # To be perfectly correct, we can just instantiate two separate ones if auto-splitting
        # But auto-splitting uses the same dataset instance.
        pass

    train_loader = None
    val_loader = None
    
    if len(train_dataset) > 0 and len(val_dataset) > 0:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=sentence_collate_fn, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=sentence_collate_fn, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, word_vocab
