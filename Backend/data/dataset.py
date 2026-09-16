import math
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class SaneSkeletalAugmentation:
    def __init__(self, max_rot_degrees=10.0, max_shift=0.05, noise_std=0.01, apply_prob=0.7):
        self.max_rot_rad = math.radians(max_rot_degrees)
        self.max_shift = max_shift
        self.noise_std = noise_std
        self.apply_prob = apply_prob

    def __call__(self, tensor):
        if torch.rand(1).item() > self.apply_prob:
            return tensor
        aug = tensor.clone()
        coords = aug[0, :159].view(53, 3)  # 53 pts = lh21+rh21+arms4+face7

        angle = (torch.rand(1).item() * 2 - 1) * self.max_rot_rad
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        x, y = coords[:, 0].clone(), coords[:, 1].clone()
        coords[:, 0] = x * cos_a - y * sin_a
        coords[:, 1] = x * sin_a + y * cos_a

        coords[:, 0] += (torch.rand(1).item() * 2 - 1) * self.max_shift
        coords[:, 1] += (torch.rand(1).item() * 2 - 1) * self.max_shift
        coords += torch.randn_like(coords) * self.noise_std

        aug[0, :159] = coords.view(159)
        return aug


class ISLWordLevelDataset(Dataset):
    """Word-level dataset built from a pre-split, pre-balanced file list."""
    def __init__(self, file_label_pairs, class_to_idx, transform=None):
        self.pairs = file_label_pairs
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path, label = self.pairs[idx]
        tensor = torch.load(path)  # (1, 163)
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, torch.tensor(self.class_to_idx[label], dtype=torch.long)


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
        keypoints = torch.load(path)  # (T, 163)
        target = torch.tensor([self.word_vocab[w] for w in words], dtype=torch.long)
        return keypoints, target


def sentence_collate_fn(batch):
    inputs, targets = zip(*batch)
    in_lens = torch.tensor([x.shape[0] for x in inputs], dtype=torch.long)
    tgt_lens = torch.tensor([y.shape[0] for y in targets], dtype=torch.long)
    return (pad_sequence(inputs, batch_first=True, padding_value=0.0),
            pad_sequence(targets, batch_first=True, padding_value=0),
            in_lens, tgt_lens)
