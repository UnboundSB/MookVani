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
    inputs = torch.stack([x for x, _ in batch])   # (B, 1, 163)
    labels = torch.stack([y for _, y in batch])   # (B,) 0-indexed class ids
    return inputs, labels

class ISLSentenceLevelDataset(Dataset):
    """Sentence-level dataset: each sample is a (T, 163) sequence with a MULTI-WORD target
    (list of word indices, in signing order) for CTC. Requires `sentence_to_words`
    and a shared `word_vocab`.
    """
    def __init__(self, file_label_pairs, sentence_to_words, word_vocab):
        self.sentence_to_words = sentence_to_words
        self.word_vocab = word_vocab  # word -> idx (1-indexed, 0 reserved for CTC blank)

        self.samples = []
        for path, phrase in file_label_pairs:
            if phrase in self.sentence_to_words:
                words = self.sentence_to_words[phrase]
                if all(w in self.word_vocab for w in words):
                    self.samples.append((path, words))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, words = self.samples[idx]
        keypoints = torch.load(path, weights_only=True)  # (T, 163)
        target = torch.tensor([self.word_vocab[w] for w in words], dtype=torch.long)
        return keypoints, target

def sentence_collate_fn(batch):
    inputs, targets = zip(*batch)
    in_lens = torch.tensor([x.shape[0] for x in inputs], dtype=torch.long)
    tgt_lens = torch.tensor([y.shape[0] for y in targets], dtype=torch.long)
    return (pad_sequence(inputs, batch_first=True, padding_value=0.0),
            pad_sequence(targets, batch_first=True, padding_value=0),
            in_lens, tgt_lens)

def build_word_dataloaders(train_dir, val_dir, batch_size=32, num_workers=0):
    if not (os.path.exists(train_dir) and os.path.exists(val_dir)):
        return None, None, None

    train_dataset_raw = ISLWordLevelDataset(train_dir)
    val_dataset_raw = ISLWordLevelDataset(val_dir)

    all_labels = sorted(set(l for _, l in train_dataset_raw.pairs) | set(l for _, l in val_dataset_raw.pairs))
    class_to_idx = {c: i for i, c in enumerate(all_labels)}
    
    train_dataset = ISLWordLevelDatasetIndexed(train_dataset_raw, class_to_idx)
    val_dataset = ISLWordLevelDatasetIndexed(val_dataset_raw, class_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              collate_fn=word_level_collate, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=word_level_collate, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, class_to_idx

def build_sentence_dataloaders(train_dir, val_dir, sentence_to_words, class_to_idx, batch_size=8, num_workers=0):
    word_vocab = {w: i + 1 for w, i in class_to_idx.items()}  # +1: 0 reserved for CTC blank
    
    train_files = glob.glob(f"{train_dir}/*/*.pt")
    val_files = glob.glob(f"{val_dir}/*/*.pt")
    
    train_pairs = [(p, os.path.basename(os.path.dirname(p))) for p in train_files]
    val_pairs = [(p, os.path.basename(os.path.dirname(p))) for p in val_files]

    train_dataset = ISLSentenceLevelDataset(train_pairs, sentence_to_words, word_vocab)
    val_dataset = ISLSentenceLevelDataset(val_pairs, sentence_to_words, word_vocab)

    train_loader = None
    val_loader = None
    
    if len(train_dataset) > 0 and len(val_dataset) > 0:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=sentence_collate_fn, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=sentence_collate_fn, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, word_vocab
