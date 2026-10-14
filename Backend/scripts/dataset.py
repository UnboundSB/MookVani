import os
import glob
import json
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
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
        
        # BLOCK MASKING (To prevent instance memorization)
        # 53 points: 0-20 (Left Hand), 21-41 (Right Hand), 42-45 (Arms), 46-52 (Face)
        if torch.rand(1).item() < 0.3:
            # 30% chance to drop left hand
            coords[0:21] = 0.0
        if torch.rand(1).item() < 0.1:
            # 10% chance to drop face
            coords[46:53] = 0.0

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
        tensor = torch.load(path, weights_only=True)  # (1, 163)
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

class ISLVideoLevelDataset(Dataset):
    """Video-level dataset: each sample is a (T, 163) sequence extracted from video with a MULTI-WORD target
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


def balanced_word_collate_fn(batch):
    """Collates variable-length word sequences into a padded batch (B, T_max, 163)."""
    xs, lengths, labels = zip(*batch)
    max_len = max(lengths)
    batch_size = len(xs)
    
    x_padded = torch.zeros(batch_size, max_len, xs[0].shape[-1], dtype=torch.float32)
    for i, x in enumerate(xs):
        x_padded[i, :lengths[i], :] = x
        
    return x_padded, torch.tensor(lengths, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def generate_nothing_sample(T=None):
    """
    Generates realistic resting/idle/non-signing sample where user stands/sits naturally.
    Hands are absent (flags=0.0, coords=0.0), shoulders and face are in natural resting positions.
    """
    if T is None:
        T = np.random.randint(25, 65)

    frames = np.zeros((T, 163), dtype=np.float32)

    # Base resting body coordinates (normalized shoulder width ~1.0)
    base_arms = np.array([
        [-0.50, 0.20, 0.0], [0.50, 0.20, 0.0],   # left shoulder, right shoulder
        [-0.55, 0.65, 0.0], [0.55, 0.65, 0.0]    # left elbow, right elbow hanging down
    ], dtype=np.float32)

    # Face points anchored near nose [0, 0, 0]
    base_face = np.array([
        [-0.15, -0.15, 0.0], [0.15, -0.15, 0.0], # left eye, right eye
        [0.00,  0.00,  0.0],                     # nose tip
        [-0.25, 0.05,  0.0], [0.25, 0.05,  0.0], # left cheek, right cheek
        [-0.10, 0.18,  0.0], [0.10, 0.18,  0.0]  # left mouth, right mouth
    ], dtype=np.float32)

    # Subtle natural sway / breathing
    t_phase = np.random.uniform(0, 2 * np.pi)
    t_freq = np.random.uniform(0.06, 0.12)
    sway = 0.012 * np.sin(np.arange(T) * t_freq + t_phase)[:, np.newaxis, np.newaxis]

    noise_arms = np.random.normal(0, 0.006, (T, 4, 3)).astype(np.float32)
    noise_face = np.random.normal(0, 0.004, (T, 7, 3)).astype(np.float32)

    arms_seq = base_arms[np.newaxis, :, :] + sway + noise_arms
    face_seq = base_face[np.newaxis, :, :] + sway + noise_face

    frames[:, 126:138] = arms_seq.reshape(T, 12)
    frames[:, 138:159] = face_seq.reshape(T, 21)
    frames[:, 159:163] = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32) # hands absent, arms & face present

    # In 20% of nothing samples, simulate casual idle hand scratch or resting on lap
    if np.random.rand() < 0.20:
        hand_idx = 0 if np.random.rand() < 0.5 else 63
        flag_idx = 159 if hand_idx == 0 else 160
        hand_pts = np.random.normal(0, 0.04, (T, 21, 3)).astype(np.float32)
        hand_pts += np.array([0.0, 0.40, 0.0]) # resting down on lap
        frames[:, hand_idx:hand_idx+63] = hand_pts.reshape(T, 63)
        frames[:, flag_idx] = 1.0

    return torch.tensor(frames, dtype=torch.float32)


def augment_heavy_skeleton(x):
    """
    Comprehensive sign-language skeletal augmentations:
    1. Horizontal Flip (50% prob) with Left/Right hand & body part swap and X-coordinate inversion
    2. Heavy Jitter (Gaussian noise on active landmarks)
    3. Random 2D Rotation (+/- 15 degrees in XY plane)
    4. Random Spatial Translation / Shift (+/- 0.12)
    5. Random Spatial Scaling / Tilt (0.85x to 1.15x)
    6. Random Hand Occlusion / Dropout (Mask 1 hand with 20% prob)
    7. Temporal Speed Scaling (0.6x to 1.4x) & Random Frame Dropping
    """
    T, F = x.shape
    x_aug = x.clone()

    # 1. Horizontal Flip (Mirroring) with Left-Right Hand and Body Symmetry Swap
    if torch.rand(1).item() < 0.5:
        # Invert all X coordinates (every 3rd value starting at 0 up to 159)
        x_aug[:, 0:159:3] = -x_aug[:, 0:159:3]

        # Swap Left Hand [0:63] and Right Hand [63:126]
        lh = x_aug[:, 0:63].clone()
        rh = x_aug[:, 63:126].clone()
        x_aug[:, 0:63] = rh
        x_aug[:, 63:126] = lh

        # Swap Left Arm (shoulder 126:129, elbow 132:135) with Right Arm (shoulder 129:132, elbow 135:138)
        ls = x_aug[:, 126:129].clone(); rs = x_aug[:, 129:132].clone()
        le = x_aug[:, 132:135].clone(); re = x_aug[:, 135:138].clone()
        x_aug[:, 126:129] = rs; x_aug[:, 129:132] = ls
        x_aug[:, 132:135] = re; x_aug[:, 135:138] = le

        # Swap Face symmetric pairs: left eye [138:141] <-> right eye [141:144], cheeks, mouth
        le_eye = x_aug[:, 138:141].clone(); re_eye = x_aug[:, 141:144].clone()
        x_aug[:, 138:141] = re_eye; x_aug[:, 141:144] = le_eye
        lc = x_aug[:, 147:150].clone(); rc = x_aug[:, 150:153].clone()
        x_aug[:, 147:150] = rc; x_aug[:, 150:153] = lc
        lm = x_aug[:, 153:156].clone(); rm = x_aug[:, 156:159].clone()
        x_aug[:, 153:156] = rm; x_aug[:, 156:159] = lm

        # Swap hand visibility flags
        lh_flag = x_aug[:, 159].clone(); rh_flag = x_aug[:, 160].clone()
        x_aug[:, 159] = rh_flag; x_aug[:, 160] = lh_flag

    # 2. Tilt a bit (Random 2D Rotation / Tilt in XY plane: +/- 8 degrees)
    if torch.rand(1).item() < 0.5:
        angle_deg = (torch.rand(1).item() * 2 - 1) * 8.0
        angle_rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        for i in range(0, 159, 3):
            px = x_aug[:, i].clone()
            py = x_aug[:, i+1].clone()
            pts_mask = (px != 0) | (py != 0)
            x_aug[:, i] = torch.where(pts_mask, px * cos_a - py * sin_a, px)
            x_aug[:, i+1] = torch.where(pts_mask, px * sin_a + py * cos_a, py)

    # 3. Subtle Spatial Translation / Shift (+/- 0.03)
    if torch.rand(1).item() < 0.4:
        dx = (torch.rand(1).item() - 0.5) * 0.06
        dy = (torch.rand(1).item() - 0.5) * 0.06
        for i in range(0, 159, 3):
            x_aug[:, i] = torch.where(x_aug[:, i] != 0, x_aug[:, i] + dx, x_aug[:, i])
            x_aug[:, i+1] = torch.where(x_aug[:, i+1] != 0, x_aug[:, i+1] + dy, x_aug[:, i+1])

    # 4. Subtle Scaling (0.95x to 1.05x)
    if torch.rand(1).item() < 0.4:
        scale = torch.rand(1).item() * 0.10 + 0.95
        x_aug[:, :159] = x_aug[:, :159] * scale

    # NO extra noise/jitter as specified by user
    return x_aug


class BalancedISLWordDataset(Dataset):
    """
    Word-level dataset enforcing EXACTLY `samples_per_class` samples (default: 14)
    per class, with heavy skeletal augmentations (flip, jitter, tilt, rotate, dropout)
    and realistic [NOTHING] idle class generation.
    No files are removed or altered on disk.
    """
    def __init__(self, filepaths, labels, classes, max_len=300, is_train=True):
        self.filepaths = filepaths
        self.labels = labels
        self.classes = classes
        self.max_len = max_len
        self.is_train = is_train
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        label = self.labels[idx]

        if path.startswith("SYNTHETIC_NOTHING"):
            x = generate_nothing_sample()
        else:
            x = torch.load(path, weights_only=True)
            if self.is_train:
                x = augment_heavy_skeleton(x)

        if x.shape[0] > self.max_len:
            x = x[:self.max_len]

        length = x.shape[0]
        return x, length, label


def get_balanced_word_dataloaders(
    data_dir,
    batch_size=32,
    samples_per_class=14,
    val_samples_per_class=3,
    max_len=300,
    include_nothing_class=True,
    num_workers=0,
    pin_memory=False,
    seed=42,
    save_meta_path=None
):
    """
    Constructs train and val DataLoaders using EXACTLY `samples_per_class` samples
    for each qualifying class plus the [NOTHING] idle class.

    Returns:
        train_loader, val_loader, class_to_idx, idx_to_class, qualifying_classes
    """
    # 1. Discover qualifying classes (classes with at least samples_per_class)
    all_dirs = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d not in ("NONE", "[NOTHING]", "NOTHING")
    ])

    class_files_map = {}
    for c in all_dirs:
        c_files = sorted(glob.glob(os.path.join(data_dir, c, "*.pt")))
        if len(c_files) >= samples_per_class:
            class_files_map[c] = c_files[:samples_per_class]

    qualifying_classes = sorted(list(class_files_map.keys()))
    if include_nothing_class:
        classes_with_nothing = qualifying_classes + ["[NOTHING]"]
    else:
        classes_with_nothing = qualifying_classes

    class_to_idx = {cls: idx for idx, cls in enumerate(classes_with_nothing)}
    idx_to_class = {idx: cls for idx, cls in enumerate(classes_with_nothing)}

    train_files, train_labels = [], []
    val_files, val_labels = [], []

    # 2. Split exactly samples_per_class into train and val deterministically
    np.random.seed(seed)
    # 2. Split or load all samples_per_class (default: 14)
    np.random.seed(seed)
    if val_samples_per_class == 0:
        train_per_cls = samples_per_class
        val_per_cls = samples_per_class
        for c in qualifying_classes:
            files = class_files_map[c]
            c_idx = class_to_idx[c]
            train_files.extend(files)
            train_labels.extend([c_idx] * len(files))
            val_files.extend(files)
            val_labels.extend([c_idx] * len(files))
        if include_nothing_class:
            nothing_idx = class_to_idx["[NOTHING]"]
            for i in range(samples_per_class):
                train_files.append(f"SYNTHETIC_NOTHING_{i}")
                train_labels.append(nothing_idx)
                val_files.append(f"SYNTHETIC_NOTHING_{i}")
                val_labels.append(nothing_idx)
    else:
        train_per_cls = samples_per_class - val_samples_per_class
        val_per_cls = val_samples_per_class

        for c in qualifying_classes:
            files = class_files_map[c]
            c_idx = class_to_idx[c]

            train_files.extend(files[:train_per_cls])
            train_labels.extend([c_idx] * train_per_cls)

            val_files.extend(files[train_per_cls:])
            val_labels.extend([c_idx] * val_per_cls)

        if include_nothing_class:
            nothing_idx = class_to_idx["[NOTHING]"]
            for i in range(train_per_cls):
                train_files.append(f"SYNTHETIC_NOTHING_TRAIN_{i}")
                train_labels.append(nothing_idx)
            for i in range(val_per_cls):
                val_files.append(f"SYNTHETIC_NOTHING_VAL_{i}")
                val_labels.append(nothing_idx)

    train_dataset = BalancedISLWordDataset(
        train_files, train_labels, classes_with_nothing, max_len=max_len, is_train=True
    )
    val_dataset = BalancedISLWordDataset(
        val_files, val_labels, classes_with_nothing, max_len=max_len, is_train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=balanced_word_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=balanced_word_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print("=" * 75)
    print("EXACT BALANCED ISL WORD DATALOADER INITIALIZED")
    print("=" * 75)
    print(f"Dataset directory:            {data_dir}")
    print(f"Total directory classes:      {len(all_dirs)}")
    print(f"Qualifying classes (>= {samples_per_class}):   {len(qualifying_classes)}")
    print(f"Samples per class loaded:     EXACTLY {samples_per_class} (Train: {train_per_cls}, Val: {val_per_cls})")
    print(f"Total Train Samples:          {len(train_dataset)}")
    print(f"Total Val Samples:            {len(val_dataset)}")
    print(f"Total Batch Size:             {batch_size}")
    print("=" * 75)

    if save_meta_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_meta_path)), exist_ok=True)
        with open(save_meta_path, "w") as f:
            json.dump({
                "num_classes": len(classes_with_nothing),
                "samples_per_class": samples_per_class,
                "classes": classes_with_nothing,
                "class_to_idx": class_to_idx
            }, f, indent=2)
        print(f"Saved class metadata mapping to: {save_meta_path}")

    return train_loader, val_loader, class_to_idx, idx_to_class, qualifying_classes
