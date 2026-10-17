"""
Phase 1 Sentence Trainer: Synthetic ISL Sentences with Frozen Phase 6 Backbone
==============================================================================
Features:
1. Dynamic Word Count: Randomly picks n in [1, 10] words per sentence.
2. Unique Sampling: Picks n distinct words (no repetition within a sentence).
3. Co-articulation & Joining: Smooth linear transition frames between consecutive words.
4. Uniform Whole-Sentence Augmentation:
   - Full horizontal flip (with left/right hand & arm landmark swap, mouth corners 51/52 swap)
   - Uniform translation shift (dx in [-0.08, 0.08], dy in [-0.03, 0.03])
   - Missing-hand sentinel coordinate re-clamping ([-1.5, 3.0, 0])
   - Uniform tilt rotation & temporal time-stretch
   - SpecAugment temporal frame masking
5. Full Dataset Coverage: Loads ALL 262 classes with ALL available .pt files (no sample capping).
6. Transfer Learning with Frozen Backbone:
   - Initializes from models/phase6_word_model/best_phase6_model.pth
   - Freezes input projection, BiGRU, Transformer, and TCNN layers
   - Trains only the CTC Sentence Head for 10 epochs
7. Tracks CTC Loss, Word Error Rate (WER), Exact Match Accuracy, and Macro F1.
"""

import os
import sys
import glob
import json
import math
import random
import argparse
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# =========================================================================
# PATH CONFIGURATION
# =========================================================================
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_DEFAULT = os.path.join(BACKEND_DIR, "data", "tensors_include50_word_level")
PHASE6_CKPT_DEFAULT = os.path.join(BACKEND_DIR, "models", "phase6_word_model", "best_phase6_model.pth")
PHASE6_CLASSES_DEFAULT = os.path.join(BACKEND_DIR, "models", "phase6_word_model", "classes.json")
OUTPUT_DIR_DEFAULT = os.path.join(BACKEND_DIR, "models", "islsentence_synthetic")

# Sentinel coordinates for absent hands
MISSING_HAND_VALS = torch.tensor([-1.5, 3.0, 0.0], dtype=torch.float32)

# =========================================================================
# 1. EVALUATION METRICS (WER, F1, ACCURACY)
# =========================================================================
def compute_wer(ref, hyp):
    """Computes Word Error Rate using Levenshtein distance."""
    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=np.int32)
    for i in range(len(ref) + 1): d[i][0] = i
    for j in range(len(hyp) + 1): d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i-1] == hyp[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j] + 1,      # Deletion
                              d[i][j-1] + 1,      # Insertion
                              d[i-1][j-1] + 1)    # Substitution
    return d[len(ref)][len(hyp)] / max(len(ref), 1)

def compute_f1_prec_rec(ref, hyp):
    """Computes token set Precision, Recall, and F1."""
    set_ref = set(ref)
    set_hyp = set(hyp)
    if len(set_hyp) == 0 and len(set_ref) == 0:
        return 1.0, 1.0, 1.0
    if len(set_hyp) == 0 or len(set_ref) == 0:
        return 0.0, 0.0, 0.0
    true_pos = len(set_ref.intersection(set_hyp))
    precision = true_pos / len(set_hyp)
    recall = true_pos / len(set_ref)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, precision, recall

# =========================================================================
# 2. UNIFORM SENTENCE-LEVEL AUGMENTATIONS
# =========================================================================
def flip_pose(seq):
    """Flips entire sentence horizontally with accurate landmark swapping."""
    seq = seq.clone()
    T = seq.shape[0]
    coords = seq[:, :159].view(T, 53, 3).clone()
    coords[:, :, 0] = -coords[:, :, 0]

    # Swap Left and Right hands: [0..21) <-> [21..42)
    lh = coords[:, 0:21, :].clone()
    rh = coords[:, 21:42, :].clone()
    coords[:, 0:21, :] = rh
    coords[:, 21:42, :] = lh

    # Swap Arm landmarks: [42..47) <-> [47..52)
    l_arm = coords[:, 42:47, :].clone()
    r_arm = coords[:, 47:52, :].clone()
    coords[:, 42:47, :] = r_arm
    coords[:, 47:52, :] = l_arm

    # Swap Mouth corner landmarks: 51 (lip left) <-> 52 (lip right)
    m_left = coords[:, 51, :].clone()
    m_right = coords[:, 52, :].clone()
    coords[:, 51, :] = m_right
    coords[:, 52, :] = m_left

    seq[:, :159] = coords.view(T, 159)

    # Swap presence flags: [159]=lh_flag, [160]=rh_flag
    lh_flag = seq[:, 159].clone()
    rh_flag = seq[:, 160].clone()
    seq[:, 159] = rh_flag
    seq[:, 160] = lh_flag

    # Re-clamp missing hand sentinel positions
    for t in range(T):
        if seq[t, 159] == 0:
            seq[t, 0:63] = MISSING_HAND_VALS.repeat(21)
        if seq[t, 160] == 0:
            seq[t, 63:126] = MISSING_HAND_VALS.repeat(21)
    return seq

def tilt_pose(seq, deg=12.0):
    """Uniform rotation tilt across all frames."""
    seq = seq.clone()
    T = seq.shape[0]
    angle = math.radians(random.uniform(-deg, deg))
    c, s = math.cos(angle), math.sin(angle)
    R = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)

    coords = seq[:, :159].view(T, 53, 3)
    xy = coords[:, :, :2].reshape(-1, 2)
    coords[:, :, :2] = (xy @ R.T).view(T, 53, 2)
    seq[:, :159] = coords.view(T, 159)

    # Re-clamp missing hand sentinels
    for t in range(T):
        if seq[t, 159] == 0:
            seq[t, 0:63] = MISSING_HAND_VALS.repeat(21)
        if seq[t, 160] == 0:
            seq[t, 63:126] = MISSING_HAND_VALS.repeat(21)
    return seq

def shift_pose(seq, dx_max=0.08, dy_max=0.03):
    """Uniform translation offset across all frames."""
    seq = seq.clone()
    T = seq.shape[0]
    dx = random.uniform(-dx_max, dx_max)
    dy = random.uniform(-dy_max, dy_max)

    coords = seq[:, :159].view(T, 53, 3)
    coords[:, :, 0] += dx
    coords[:, :, 1] += dy
    seq[:, :159] = coords.view(T, 159)

    # Re-clamp missing hand sentinels
    for t in range(T):
        if seq[t, 159] == 0:
            seq[t, 0:63] = MISSING_HAND_VALS.repeat(21)
        if seq[t, 160] == 0:
            seq[t, 63:126] = MISSING_HAND_VALS.repeat(21)
    return seq

def time_stretch(seq, lo=0.85, hi=1.15):
    """Uniform speed/time scaling across the full sentence."""
    T, D = seq.shape
    factor = random.uniform(lo, hi)
    new_T = max(10, round(T * factor))
    if new_T == T:
        return seq
    s = F.interpolate(seq.T.unsqueeze(0), size=new_T, mode="linear", align_corners=True)
    return s.squeeze(0).T

def temporal_frame_mask(seq, p=0.30, max_mask=3):
    """SpecAugment temporal masking: zeroing 1-3 random contiguous frames."""
    if random.random() > p or seq.shape[0] <= 10:
        return seq
    seq = seq.clone()
    T = seq.shape[0]
    mask_len = random.randint(1, min(max_mask, T // 4))
    start_t = random.randint(0, T - mask_len)
    seq[start_t:start_t + mask_len, :] = 0.0
    return seq

def augment_sentence(seq):
    """Applies all sentence-level augmentations uniformly."""
    if random.random() < 0.50:
        seq = flip_pose(seq)
    if random.random() < 0.70:
        seq = shift_pose(seq, dx_max=0.08, dy_max=0.03)
    if random.random() < 0.60:
        seq = tilt_pose(seq, deg=12.0)
    if random.random() < 0.50:
        seq = time_stretch(seq, lo=0.85, hi=1.15)
    seq = temporal_frame_mask(seq, p=0.30, max_mask=3)
    if random.random() < 0.50:
        noise = torch.randn_like(seq) * 0.005
        seq = seq + noise
    return seq

# =========================================================================
# 3. SYNTHETIC SENTENCE DATASET (1-10 WORDS, NO REPETITION, UNCAPPED)
# =========================================================================
class SyntheticSentenceDataset(Dataset):
    def __init__(self, data_dir, classes, samples_per_epoch=1200, is_train=True):
        self.data_dir = data_dir
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.samples_per_epoch = samples_per_epoch
        self.is_train = is_train

        # Load all available files per class without capping
        self.class_files = {}
        self.rest_segments = []
        valid_classes = []

        for cls in classes:
            cls_dir = os.path.join(data_dir, cls)
            if os.path.isdir(cls_dir):
                files = sorted(glob.glob(os.path.join(cls_dir, "*.pt")))
                if len(files) > 0:
                    self.class_files[cls] = files
                    valid_classes.append(cls)
                    # Harvest natural resting poses from edge frames
                    for f in files[:4]:
                        try:
                            x = torch.load(f, weights_only=True)
                            if x.shape[0] >= 20:
                                self.rest_segments.append(x[:8].float())
                                self.rest_segments.append(x[-8:].float())
                        except Exception:
                            pass

        self.valid_classes = valid_classes
        if len(self.rest_segments) == 0:
            dummy_rest = torch.zeros(10, 163)
            dummy_rest[:, 0:63] = MISSING_HAND_VALS.repeat(21)
            dummy_rest[:, 63:126] = MISSING_HAND_VALS.repeat(21)
            self.rest_segments.append(dummy_rest)

    def __len__(self):
        return self.samples_per_epoch

    def get_real_rest_frames(self, num_frames):
        """Samples natural resting segment for sentence boundary padding."""
        idx = random.randint(0, len(self.rest_segments) - 1)
        seg = self.rest_segments[idx]
        while seg.shape[0] < num_frames:
            nxt = self.rest_segments[random.randint(0, len(self.rest_segments) - 1)]
            seg = torch.cat([seg, nxt], dim=0)
        start = random.randint(0, seg.shape[0] - num_frames)
        return seg[start:start + num_frames].clone()

    def __getitem__(self, idx):
        # 1. Randomly pick n in [1, 10] words
        n = random.randint(1, 10)

        # 2. Pick n distinct words without repetition
        n_available = min(n, len(self.valid_classes))
        chosen_words = random.sample(self.valid_classes, n_available)

        tensors = []
        labels = []

        # Optional prefix resting frames (6 to 14 frames)
        n_prefix = random.randint(6, 14) if self.is_train else 8
        tensors.append(self.get_real_rest_frames(n_prefix))

        # 3. Concatenate gesture words with smooth co-articulation transitions
        for i, word in enumerate(chosen_words):
            filepath = random.choice(self.class_files[word])
            x = torch.load(filepath, weights_only=True).float()

            # Smooth co-articulation transition between consecutive words (3-5 frames)
            if i > 0 and len(tensors) > 0:
                prev_last = tensors[-1][-1:]
                curr_first = x[:1]
                n_trans = random.randint(3, 5) if self.is_train else 4
                alphas = torch.linspace(0.0, 1.0, steps=n_trans + 2)[1:-1].unsqueeze(1)
                transition = (1.0 - alphas) * prev_last + alphas * curr_first
                tensors.append(transition)

            tensors.append(x)
            labels.append(self.class_to_idx[word])

        # Optional suffix resting frames (4 to 10 frames)
        n_suffix = random.randint(4, 10) if self.is_train else 6
        tensors.append(self.get_real_rest_frames(n_suffix))

        full_tensor = torch.cat(tensors, dim=0)

        # 4. Apply uniform augmentation across the entire sentence
        if self.is_train:
            full_tensor = augment_sentence(full_tensor)

        return full_tensor, full_tensor.shape[0], torch.tensor(labels, dtype=torch.long)

def collate_fn_sentence(batch):
    """Pads sentences to max length in the batch for batch processing."""
    xs, lengths, labels = zip(*batch)
    max_len = max(lengths)
    batch_size = len(xs)

    x_padded = torch.zeros(batch_size, max_len, xs[0].shape[-1], dtype=torch.float32)
    for i, x in enumerate(xs):
        x_padded[i, :lengths[i], :] = x

    lengths = torch.tensor(lengths, dtype=torch.long)
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    targets = torch.cat(labels)

    return x_padded, lengths, targets, target_lengths

# =========================================================================
# 4. PHASE 1 SENTENCE MODEL (FROZEN PHASE 6 BACKBONE + CTC HEAD)
# =========================================================================
class PosEnc(nn.Module):
    def __init__(self, d, maxlen=4096):
        super().__init__()
        self.d = d
        pe = self._build_pe(maxlen, d)
        self.register_buffer("pe", pe)

    def _build_pe(self, maxlen, d):
        pe = torch.zeros(maxlen, d, dtype=torch.float32)
        pos = torch.arange(0, maxlen).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    def forward(self, x):
        T = x.size(1)
        if T > self.pe.size(1):
            new_maxlen = max(T + 256, self.pe.size(1) * 2)
            self.pe = self._build_pe(new_maxlen, self.d).to(x.device)
        return x + self.pe[:, :T, :]

class TCNN(nn.Module):
    def __init__(self, ic, oc, k=3, drop=0.10):
        super().__init__()
        p = k // 2
        self.c1 = nn.Conv1d(ic, oc, k, padding=p)
        self.b1 = nn.BatchNorm1d(oc)
        self.g1 = nn.GELU()
        self.d1 = nn.Dropout(drop)
        self.c2 = nn.Conv1d(oc, oc, k, padding=p)
        self.b2 = nn.BatchNorm1d(oc)
        self.g2 = nn.GELU()
        self.d2 = nn.Dropout(drop)
        self.res = nn.Conv1d(ic, oc, 1) if ic != oc else nn.Identity()

    def forward(self, x):
        r = self.res(x)
        o = self.d1(self.g1(self.b1(self.c1(x))))
        o = self.d2(self.g2(self.b2(self.c2(o))))
        return o + r

class Phase1SentenceModel(nn.Module):
    def __init__(self, num_classes, d=128, gru_layers=2, n_head=4, n_trans=2, drop=0.15):
        super().__init__()
        self.num_classes = num_classes
        self.ctc_blank_id = num_classes  # Blank token is at index num_classes

        # Frozen Phase 6 Backbone
        self.proj = nn.Sequential(
            nn.Linear(163, d), nn.LayerNorm(d), nn.Dropout(drop))
        self.gru = nn.GRU(d, d // 2, gru_layers, batch_first=True, bidirectional=True,
                          dropout=drop if gru_layers > 1 else 0.0)
        self.gnorm = nn.LayerNorm(d)
        self.pe = PosEnc(d)
        el = nn.TransformerEncoderLayer(d, n_head, d * 2, drop, "gelu", batch_first=True)
        self.trans = nn.TransformerEncoder(el, n_trans, enable_nested_tensor=False)
        self.tnorm = nn.LayerNorm(d)
        self.t1 = TCNN(d, d, drop=drop)
        self.t2 = TCNN(d, d * 2, drop=drop)

        # Trainable CTC Sentence Head (projects frame features d*2=256 -> num_classes+1)
        self.ctc_head = nn.Sequential(
            nn.Conv1d(d * 2, d * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(d * 2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv1d(d * 2, num_classes + 1, kernel_size=1)
        )

    def load_phase6_backbone(self, ckpt_path):
        """Loads pretrained weights from Phase 6 and freezes backbone layers."""
        print(f"[INFO] Loading Phase 6 checkpoint: {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # Drop the classification head and fixed 512-length pe buffer from Phase 6
        filtered_sd = {k: v for k, v in sd.items() if not k.startswith("head.") and k != "pe.pe"}
        missing, unexpected = self.load_state_dict(filtered_sd, strict=False)
        print(f"[INFO] Loaded backbone weights. (Head dropped, {len(filtered_sd)} keys restored)")

        # Freeze all backbone layers
        frozen_count = 0
        trainable_count = 0
        for name, param in self.named_parameters():
            if "ctc_head" in name:
                param.requires_grad = True
                trainable_count += param.numel()
            else:
                param.requires_grad = False
                frozen_count += param.numel()

        print(f"[FREEZE] Backbone frozen successfully:")
        print(f"         - Frozen Parameters:    {frozen_count:,} (100% frozen)")
        print(f"         - Trainable Parameters: {trainable_count:,} (CTC Sentence Head)")

    def forward(self, x, lengths):
        # Flatten GRU parameters for contiguity
        if hasattr(self.gru, "flatten_parameters"):
            self.gru.flatten_parameters()

        # 1. Input projection
        h = self.proj(x)

        # 2. BiGRU temporal context
        g, _ = self.gru(h)
        h = self.gnorm(h + g)

        # 3. Transformer global attention
        h = self.pe(h)
        h = self.trans(h)
        h = self.tnorm(h)

        # 4. TCNN local temporal refinement
        # Conv1d expects (B, C, T)
        t = h.transpose(1, 2)
        t = self.t1(t)
        t = self.t2(t)  # (B, 256, T)

        # 5. CTC Head frame logits
        logits = self.ctc_head(t)  # (B, num_classes+1, T)
        logits = logits.transpose(1, 2)  # (B, T, num_classes+1)

        # Log-softmax for CTC loss: requires shape (T, B, C)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

# =========================================================================
# 5. CTC DECODING & HELPERS
# =========================================================================
def ctc_decode(log_probs, lengths, blank_id):
    """Greedy CTC collapse: argmax -> collapse adjacent duplicates -> remove blank."""
    preds = log_probs.argmax(dim=-1).cpu().numpy()  # (B, T)
    lengths = lengths.cpu().numpy()

    decoded = []
    for i in range(preds.shape[0]):
        pred_seq = preds[i, :lengths[i]]
        collapsed = []
        for j in range(len(pred_seq)):
            tok = pred_seq[j]
            if tok != blank_id:
                if j == 0 or tok != pred_seq[j - 1]:
                    collapsed.append(tok)
        decoded.append(collapsed)
    return decoded

def get_targets_list(targets, target_lengths):
    """Splits flattened targets tensor into list of lists."""
    targets = targets.cpu().numpy()
    target_lengths = target_lengths.cpu().numpy()
    true_seqs = []
    idx = 0
    for l in target_lengths:
        true_seqs.append(targets[idx:idx + l].tolist())
        idx += l
    return true_seqs

# =========================================================================
# 6. PLOTTING DIAGNOSTICS
# =========================================================================
def save_sentence_plots(history, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # Plot 1: 4-Panel Metric Dashboard
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    axs[0, 0].plot(epochs, history["train_loss"], label="CTC Train Loss", color="#1f77b4", lw=2)
    axs[0, 0].plot(epochs, history["val_loss"], label="CTC Val Loss", color="#ff7f0e", lw=2, linestyle="--")
    axs[0, 0].set_title("CTC Loss (Lower is Better)")
    axs[0, 0].set_xlabel("Epoch"); axs[0, 0].grid(True, linestyle="--", alpha=0.6); axs[0, 0].legend()

    axs[0, 1].plot(epochs, history["wer"], label="Word Error Rate (WER)", color="#d62728", lw=2)
    axs[0, 1].set_title("Word Error Rate (Lower is Better)")
    axs[0, 1].set_xlabel("Epoch"); axs[0, 1].grid(True, linestyle="--", alpha=0.6); axs[0, 1].legend()

    axs[1, 0].plot(epochs, history["acc"], label="Exact Sentence Acc", color="#2ca02c", lw=2)
    axs[1, 0].set_title("Exact Match Sentence Accuracy (Higher is Better)")
    axs[1, 0].set_xlabel("Epoch"); axs[1, 0].grid(True, linestyle="--", alpha=0.6); axs[1, 0].legend()

    axs[1, 1].plot(epochs, history["f1"], label="Sequence F1", color="#9467bd", lw=2)
    axs[1, 1].set_title("Sequence F1 Score (Higher is Better)")
    axs[1, 1].set_xlabel("Epoch"); axs[1, 1].grid(True, linestyle="--", alpha=0.6); axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "ctc_metrics.png"), dpi=150)
    plt.close()

    # Plot 2: Loss & WER Progression
    fig2, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(epochs, history["train_loss"], color="#1f77b4", label="Train Loss", lw=2)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("CTC Loss", color="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(epochs, history["wer"], color="#d62728", label="WER", lw=2)
    ax2.set_ylabel("Word Error Rate", color="#d62728")
    plt.title("Phase 1 Sentence: CTC Loss and WER Progression")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "loss_and_wer.png"), dpi=150)
    plt.close()
    print(f"[SAVED] Diagnostic plots -> {plots_dir}")

# =========================================================================
# 7. MAIN TRAINING PIPELINE
# =========================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1: Synthetic ISL Sentence CTC Training")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR_DEFAULT, help="Path to word tensors directory")
    parser.add_argument("--phase6_ckpt", type=str, default=PHASE6_CKPT_DEFAULT, help="Phase 6 best model checkpoint")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR_DEFAULT, help="Directory to save sentence model artifacts")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs to run")
    parser.add_argument("--train_samples", type=int, default=1200, help="Synthetic sentences per train epoch")
    parser.add_argument("--val_samples", type=int, default=200, help="Synthetic sentences per val epoch")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1.0e-3, help="Learning rate for trainable CTC head")
    parser.add_argument("--backbone_lr", type=float, default=1.0e-5, help="Learning rate for backbone when unfrozen")
    parser.add_argument("--unfreeze_backbone", action="store_true", help="Unfreeze all backbone parameters for end-to-end training")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in output_dir")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Explicit checkpoint to resume from")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0 for Windows)")
    return parser.parse_args()

def train_phase1_sentence():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    CHECKPOINT_FILE = os.path.join(args.output_dir, "phase1_sentence_checkpoint.pth")
    BEST_MODEL_FILE = os.path.join(args.output_dir, "phase1_sentence_best.pth")
    FINAL_MODEL_FILE = os.path.join(args.output_dir, "phase1_sentence_final.pth")
    CLASSES_FILE = os.path.join(args.output_dir, "classes.json")
    HISTORY_FILE = os.path.join(args.output_dir, "history.json")

    # Load Classes from Phase 6 (all 262 classes)
    if os.path.exists(PHASE6_CLASSES_DEFAULT):
        with open(PHASE6_CLASSES_DEFAULT, "r") as f:
            classes = json.load(f)
    else:
        classes = sorted([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))])

    num_classes = len(classes)
    CTC_BLANK_ID = num_classes  # Blank index is 262

    with open(CLASSES_FILE, "w") as f:
        json.dump({"classes": classes, "num_classes": num_classes, "ctc_blank_id": CTC_BLANK_ID}, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("PHASE 1 SENTENCE TRAINER - SYNTHETIC ISL SENTENCES (CTC)")
    print(f"Device:            {device}")
    print(f"Word Classes:      {num_classes} (CTC Blank ID: {CTC_BLANK_ID})")
    print(f"Pretrained Model:  {args.phase6_ckpt}")
    print(f"Output Directory:  {args.output_dir}")
    print(f"Backbone Status:   {'UNFROZEN (End-to-End Fine-Tuning)' if args.unfreeze_backbone else 'FROZEN (Training CTC Head Only)'}")
    print(f"Sentence Range:    1 to 10 words (No repetition within sentence)")
    print(f"Stage Epochs:      {args.epochs} | Train Samples: {args.train_samples} | Val: {args.val_samples} | Batch: {args.batch_size}")
    print("=" * 65)

    # Initialize Model
    model = Phase1SentenceModel(num_classes=num_classes, d=128, gru_layers=2, n_head=4, n_trans=2, drop=0.15).to(device)

    start_epoch = 0
    best_wer = float("inf")
    best_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "wer": [], "acc": [], "f1": []}

    resume_path = None
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        resume_path = args.checkpoint_path
    elif args.resume and os.path.exists(CHECKPOINT_FILE):
        resume_path = CHECKPOINT_FILE

    if resume_path:
        print(f"[RESUME] Loading weights from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            start_epoch = ckpt.get("epoch", 0)
            best_wer = ckpt.get("val_wer", float("inf"))
            best_loss = ckpt.get("val_loss", float("inf"))
            history = ckpt.get("history", history)
            print(f"[RESUME] Resuming from Epoch {start_epoch} (Best WER: {best_wer*100:.2f}%)")
        else:
            model.load_state_dict(ckpt)
            print(f"[RESUME] Loaded model weights from {resume_path}")
    elif os.path.exists(args.phase6_ckpt):
        model.load_phase6_backbone(args.phase6_ckpt)
    else:
        print(f"[WARNING] Pretrained Phase 6 checkpoint not found at {args.phase6_ckpt}! Starting CTC head with random backbone.")

    # Configure Freezing vs Unfreezing
    if args.unfreeze_backbone:
        for p in model.parameters():
            p.requires_grad = True
        backbone_params = [p for n, p in model.named_parameters() if not n.startswith("ctc_head.")]
        head_params = [p for n, p in model.named_parameters() if n.startswith("ctc_head.")]
        optimizer = optim.AdamW([
            {"params": backbone_params, "lr": args.backbone_lr, "weight_decay": 1e-4},
            {"params": head_params, "lr": args.lr, "weight_decay": 1e-3}
        ])
        print(f"[UNFREEZE] All layers trainable:")
        print(f"           - Backbone Parameters: {sum(p.numel() for p in backbone_params):,} (LR: {args.backbone_lr})")
        print(f"           - CTC Head Parameters: {sum(p.numel() for p in head_params):,} (LR: {args.lr})")
    else:
        for name, param in model.named_parameters():
            if "ctc_head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-3)
        print(f"[FREEZE] Backbone frozen. Training only CTC Head (LR: {args.lr}).")

    total_epochs = start_epoch + args.epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CTCLoss(blank=CTC_BLANK_ID, zero_infinity=True)

    # Datasets & DataLoaders
    train_ds = SyntheticSentenceDataset(args.data_dir, classes, samples_per_epoch=args.train_samples, is_train=True)
    val_ds = SyntheticSentenceDataset(args.data_dir, classes, samples_per_epoch=args.val_samples, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn_sentence, num_workers=args.num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn_sentence, num_workers=args.num_workers, pin_memory=False)

    for epoch in range(start_epoch, total_epochs):
        # Training Phase
        model.train()
        if not args.unfreeze_backbone:
            # Keep frozen backbone submodules in eval mode to prevent running stat drift
            model.proj.eval()
            model.gru.eval()
            model.gnorm.eval()
            model.trans.eval()
            model.tnorm.eval()
            model.t1.eval()
            model.t2.eval()
        model.ctc_head.train()

        total_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1:02d}/{total_epochs:02d} [Train Sentences]")

        for x, lengths, targets, target_lengths in pbar:
            x, lengths = x.to(device), lengths.to(device)
            targets, target_lengths = targets.to(device), target_lengths.to(device)

            optimizer.zero_grad()
            log_probs = model(x, lengths)  # (B, T, C)
            log_probs_t = log_probs.transpose(0, 1)  # (T, B, C) for CTCLoss

            loss = criterion(log_probs_t, targets, lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.ctc_head.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            pbar.set_postfix({"ctc_loss": f"{loss.item():.4f}"})

        train_loss = total_train_loss / len(train_loader)

        # Validation Phase
        model.eval()
        total_val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, lengths, targets, target_lengths in tqdm(val_loader, desc=f"Ep {epoch+1:02d}/{total_epochs:02d} [Val Decode]"):
                x, lengths = x.to(device), lengths.to(device)
                targets_dev, target_lengths_dev = targets.to(device), target_lengths.to(device)

                log_probs = model(x, lengths)
                log_probs_t = log_probs.transpose(0, 1)
                v_loss = criterion(log_probs_t, targets_dev, lengths, target_lengths_dev)
                total_val_loss += v_loss.item()

                decoded = ctc_decode(log_probs, lengths, blank_id=CTC_BLANK_ID)
                true_seqs = get_targets_list(targets, target_lengths)
                all_preds.extend(decoded)
                all_targets.extend(true_seqs)

        val_loss = total_val_loss / max(1, len(val_loader))

        # Compute Validation Metrics
        epoch_wer = 0.0
        epoch_acc = 0.0
        epoch_f1 = 0.0
        N = max(1, len(all_targets))

        for ref, hyp in zip(all_targets, all_preds):
            epoch_wer += compute_wer(ref, hyp)
            epoch_acc += 1.0 if ref == hyp else 0.0
            f1, _, _ = compute_f1_prec_rec(ref, hyp)
            epoch_f1 += f1

        val_wer = epoch_wer / N
        val_acc = epoch_acc / N
        val_f1 = epoch_f1 / N

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["wer"].append(val_wer)
        history["acc"].append(val_acc)
        history["f1"].append(val_f1)

        scheduler.step()

        print(f"\n[EPOCH {epoch+1:02d}/{total_epochs:02d} SUMMARY]")
        print(f"  Train CTC Loss: {train_loss:.4f} | Val CTC Loss: {val_loss:.4f}")
        print(f"  Word Error Rate (WER): {val_wer * 100:.2f}% | Exact Sentence Acc: {val_acc * 100:.2f}% | F1: {val_f1:.4f}")

        # Checkpoint Saving
        checkpoint_data = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "ctc_head_state_dict": model.ctc_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_wer": val_wer,
            "val_f1": val_f1,
            "history": history,
            "classes": classes,
            "num_classes": num_classes,
            "ctc_blank_id": CTC_BLANK_ID
        }
        torch.save(checkpoint_data, CHECKPOINT_FILE)

        if val_wer < best_wer or (val_wer == best_wer and val_loss < best_loss):
            best_wer = val_wer
            best_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_FILE)
            print(f"  * [NEW BEST CHECKPOINT] Saved to {BEST_MODEL_FILE} (WER: {val_wer * 100:.2f}%)")

        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)

        save_sentence_plots(history, plots_dir)
        gc.collect()
        torch.cuda.empty_cache()

    torch.save(model.state_dict(), FINAL_MODEL_FILE)
    print("\n" + "=" * 65)
    print("PHASE 1 SENTENCE TRAINING COMPLETE!")
    print(f"Best WER Model:  {BEST_MODEL_FILE}")
    print(f"Final Model:     {FINAL_MODEL_FILE}")
    print(f"Plots Saved:     {plots_dir}")
    print("=" * 65)

if __name__ == "__main__":
    train_phase1_sentence()
