"""
Phase 2+ Sentence Trainer: Unfrozen End-to-End Sentence CTC Cycling
===================================================================
Features & Architecture:
1. Dynamic Auto-Phase Cycling:
   - Scans models/ for existing sentence phases (e.g., islsentence_synthetic -> phase 2 -> phase 3 ...).
   - Automatically identifies the latest completed phase K.
   - Loads weights from Phase K and automatically creates and outputs into islsentence_synthetic_phase{K+1}/.
   - Can be repeatedly executed; every run increments and cycles to the next phase seamlessly.
2. Unfrozen End-to-End Fine-Tuning:
   - All layers (Input Proj, BiGRU, Transformer, TCNN, CTC Head) are unfrozen.
   - Uses discriminative learning rates: low LR on backbone to preserve pretrained spatial-temporal
     representations, higher LR on CTC head for continuous alignment.
3. Advanced Training Regimes:
   - Cosine Annealing with Warm Restarts or Standard Cosine Annealing schedule.
   - Model Exponential Moving Average (EMA) shadow tracking (smooths optimization valleys).
   - CTC Loss with Blank Entropy Regularization (prevents blank token dominance).
   - Gradient clipping (1.0) and SpecAugment temporal masking.
4. Uniform Sentence Augmentation:
   - Horizontal flip with landmark & hand swapping.
   - Tilt rotation, translation shifting, missing-hand sentinel clamping.
   - Smooth co-articulation interpolation between continuous sign words.
5. Auto-saves checkpoints, classes.json, history.json, structure.md, and diagnostic plots.
"""

import os
import sys
import glob
import re
import json
import math
import copy
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
# PATH CONFIGURATION & DEFAULTS
# =========================================================================
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_DEFAULT = os.path.join(BACKEND_DIR, "data", "tensors_include50_word_level")
MODELS_ROOT = os.path.join(BACKEND_DIR, "models")
PHASE1_DIR_DEFAULT = os.path.join(MODELS_ROOT, "islsentence_synthetic")
PHASE6_CLASSES_DEFAULT = os.path.join(MODELS_ROOT, "word_vocabulary", "classes.json")

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from utils.pose_augmentation import (  # noqa: E402
    AugmentConfig,
    augment_pose_sequence as augment_pose_sequence_canonical,
    mirror_pose,
    shift_pose as canonical_shift,
    tilt_pose as canonical_tilt,
    temporal_mask as canonical_mask,
    time_stretch as canonical_time_stretch,
)
from utils.dataloader_word import MAX_SAMPLES_PER_CLASS  # noqa: E402
from utils.checkpoint_utils import extract_state_dict, load_flexible  # noqa: E402

# Sentinel coordinates for absent hands
MISSING_HAND_VALS = torch.tensor([-1.5, 3.0, 0.0], dtype=torch.float32)

# =========================================================================
# 1. AUTO-DETECTION OF LATEST SENTENCE PHASE
# =========================================================================
def detect_sentence_phases(models_dir=MODELS_ROOT):
    """
    Scans models directory to find all existing sentence phases and determine
    the latest source phase and the next target phase.
    """
    phases = {}
    # Check Phase 1 base directory
    p1_dir = os.path.join(models_dir, "islsentence_synthetic")
    if os.path.isdir(p1_dir):
        phases[1] = p1_dir

    # Check numbered phase directories (islsentence_synthetic_phase2, phase3, ...)
    pattern = re.compile(r"^islsentence_synthetic_phase(\d+)$")
    if os.path.isdir(models_dir):
        for entry in os.listdir(models_dir):
            m = pattern.match(entry)
            if m:
                p_num = int(m.group(1))
                phases[p_num] = os.path.join(models_dir, entry)

    if not phases:
        # Fallback if no prior sentence model exists
        return 0, None, 1, os.path.join(models_dir, "islsentence_synthetic")

    latest_phase = max(phases.keys())
    source_dir = phases[latest_phase]
    next_phase = latest_phase + 1
    target_dir = os.path.join(models_dir, f"islsentence_synthetic_phase{next_phase}")

    return latest_phase, source_dir, next_phase, target_dir

def find_best_weights_in_phase(phase_dir, phase_num):
    """Locates the best weight checkpoint file in a given phase directory."""
    candidates = [
        os.path.join(phase_dir, f"phase{phase_num}_sentence_ema_best.pth"),
        os.path.join(phase_dir, f"phase{phase_num}_sentence_best.pth"),
        os.path.join(phase_dir, f"phase{phase_num}_sentence_final.pth"),
        os.path.join(phase_dir, f"phase{phase_num}_sentence_checkpoint.pth"),
        os.path.join(phase_dir, "phase1_sentence_best.pth"),
        os.path.join(phase_dir, "phase1_sentence_final.pth"),
        os.path.join(phase_dir, "phase1_sentence_checkpoint.pth"),
        os.path.join(phase_dir, "reconformer_synthetic_sentence_best.pth"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    
    # Generic search for .pth
    pths = sorted(glob.glob(os.path.join(phase_dir, "*.pth")), key=os.path.getmtime, reverse=True)
    return pths[0] if pths else None

# =========================================================================
# 2. EVALUATION METRICS (WER, F1, ACCURACY)
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
# 3. SENTENCE-LEVEL AUGMENTATIONS  (canonical implementation)
# =========================================================================
# The augmentations live in Backend/utils/pose_augmentation.py so that the
# word-level and the sentence-level models see exactly the same geometry.
# The old local copy swapped the 5-landmark block 42:47 against 47:52, which
# in the 53-landmark view mixes the arm landmarks with the eye landmarks, and
# it tilted around the origin instead of around the neck.
SENTENCE_AUGMENT = AugmentConfig(
    flip_prob=0.50,
    tilt_prob=0.70,
    tilt_deg=10.0,
    shift_prob=0.70,
    shift_dx=0.06,
    shift_dy=0.02,
    speed_prob=0.50,
    speed_lo=0.90,
    speed_hi=1.10,
    mask_prob=0.40,
    mask_max=3,
    mask_count=2,
    noise_std=0.0,
)


def flip_pose(seq):
    """Mirror around the signer's own axis (canonical version)."""
    return mirror_pose(seq)


def tilt_pose(seq, deg=12.0):
    """Rotation around the neck (canonical version)."""
    return canonical_tilt(seq, deg)


def shift_pose(seq, max_dx=0.08, max_dy=0.03):
    """Bounded left/right shift (canonical version)."""
    return canonical_shift(seq, max_dx, max_dy)


def time_stretch_pose(seq, low=0.85, high=1.15):
    """Speed up / slow down the clip (canonical version)."""
    return canonical_time_stretch(seq, low, high)


def temporal_masking(seq, max_mask_frames=4, num_masks=2):
    """SpecAugment temporal cutout (canonical version)."""
    return canonical_mask(seq, max_mask_frames, num_masks)


def augment_sentence(seq):
    """Applies the canonical sentence-level augmentation set."""
    return augment_pose_sequence_canonical(seq, SENTENCE_AUGMENT)

# =========================================================================
# 4. SYNTHETIC SENTENCE DATASET
# =========================================================================
class SyntheticSentenceDataset(Dataset):
    def __init__(self, data_dir, classes, samples_per_epoch=1200, is_train=True):
        self.data_dir = data_dir
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.samples_per_epoch = samples_per_epoch
        self.is_train = is_train

        self.class_files = {}
        for c in classes:
            c_dir = os.path.join(data_dir, c)
            if os.path.isdir(c_dir):
                files = sorted(glob.glob(os.path.join(c_dir, "*.pt")))
                # the same per-class clip budget as the word-level loader
                if len(files) > MAX_SAMPLES_PER_CLASS:
                    files = files[:MAX_SAMPLES_PER_CLASS]
                if files:
                    self.class_files[c] = files

        self.valid_classes = list(self.class_files.keys())
        if not self.valid_classes:
            raise ValueError(f"No valid .pt files found in {data_dir} across {len(classes)} classes!")

        self.rest_segments = []
        for c in random.sample(self.valid_classes, min(30, len(self.valid_classes))):
            f = random.choice(self.class_files[c])
            try:
                t = torch.load(f, weights_only=True).float()
                if t.shape[0] >= 6:
                    self.rest_segments.append(t[:4])
                    self.rest_segments.append(t[-4:])
            except Exception:
                pass

        if not self.rest_segments:
            dummy_rest = torch.zeros(10, 163, dtype=torch.float32)
            dummy_rest[:, 0:63] = MISSING_HAND_VALS.repeat(21)
            dummy_rest[:, 63:126] = MISSING_HAND_VALS.repeat(21)
            self.rest_segments.append(dummy_rest)

    def __len__(self):
        return self.samples_per_epoch

    def get_real_rest_frames(self, num_frames):
        idx = random.randint(0, len(self.rest_segments) - 1)
        seg = self.rest_segments[idx]
        while seg.shape[0] < num_frames:
            nxt = self.rest_segments[random.randint(0, len(self.rest_segments) - 1)]
            seg = torch.cat([seg, nxt], dim=0)
        start = random.randint(0, seg.shape[0] - num_frames)
        return seg[start:start + num_frames].clone()

    def __getitem__(self, idx):
        n = random.randint(1, 10)
        n_available = min(n, len(self.valid_classes))
        chosen_words = random.sample(self.valid_classes, n_available)

        tensors = []
        labels = []

        n_prefix = random.randint(6, 14) if self.is_train else 8
        tensors.append(self.get_real_rest_frames(n_prefix))

        for i, word in enumerate(chosen_words):
            filepath = random.choice(self.class_files[word])
            x = torch.load(filepath, weights_only=True).float()

            if i > 0 and len(tensors) > 0:
                prev_last = tensors[-1][-1:]
                curr_first = x[:1]
                n_trans = random.randint(3, 5) if self.is_train else 4
                alphas = torch.linspace(0.0, 1.0, steps=n_trans + 2)[1:-1].unsqueeze(1)
                transition = (1.0 - alphas) * prev_last + alphas * curr_first
                tensors.append(transition)

            tensors.append(x)
            labels.append(self.class_to_idx[word])

        n_suffix = random.randint(4, 10) if self.is_train else 6
        tensors.append(self.get_real_rest_frames(n_suffix))

        full_tensor = torch.cat(tensors, dim=0)

        if self.is_train:
            full_tensor = augment_sentence(full_tensor)

        return full_tensor, full_tensor.shape[0], torch.tensor(labels, dtype=torch.long)

def collate_fn_sentence(batch):
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
# 5. MODEL ARCHITECTURE (POSENC, TCNN, SENTENCE CONFORMER)
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

class ISLContinuousSentenceModel(nn.Module):
    """
    Continuous ISL Sentence Recognizer with BiGRU, Transformer Encoder,
    Temporal CNN (TCNN), and CTC Projection Head.
    """
    def __init__(self, num_classes, d=128, gru_layers=2, n_head=4, n_trans=2, drop=0.15):
        super().__init__()
        self.num_classes = num_classes
        self.ctc_blank_id = num_classes

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

        self.ctc_head = nn.Sequential(
            nn.Conv1d(d * 2, d * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(d * 2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv1d(d * 2, num_classes + 1, kernel_size=1)
        )

    def forward(self, x, lengths):
        if hasattr(self.gru, "flatten_parameters"):
            self.gru.flatten_parameters()

        h = self.proj(x)
        g, _ = self.gru(h)
        h = self.gnorm(h + g)

        h = self.pe(h)
        h = self.trans(h)
        h = self.tnorm(h)

        t = h.transpose(1, 2)
        t = self.t1(t)
        t = self.t2(t)  # (B, 256, T)

        logits = self.ctc_head(t)  # (B, num_classes+1, T)
        logits = logits.transpose(1, 2)  # (B, T, num_classes+1)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

# =========================================================================
# 6. MODEL EMA (EXPONENTIAL MOVING AVERAGE)
# =========================================================================
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.decay = decay
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        for ema_p, p in zip(self.model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)
        for ema_b, b in zip(self.model.buffers(), model.buffers()):
            ema_b.copy_(b)

# =========================================================================
# 7. CTC LOSS & REGULARIZATION
# =========================================================================
class RegularizedCTCLoss(nn.Module):
    """
    Standard CTC Loss augmented with Blank Entropy Regularization
    to prevent blank-token dominance and encourage expressive non-blank activations.
    """
    def __init__(self, blank_id, blank_penalty=0.01, zero_infinity=True):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank_id, zero_infinity=zero_infinity)
        self.blank_id = blank_id
        self.blank_penalty = blank_penalty

    def forward(self, log_probs_t, targets, input_lengths, target_lengths):
        # log_probs_t: (T, B, C)
        ctc_loss = self.ctc(log_probs_t, targets, input_lengths, target_lengths)
        if self.blank_penalty > 0.0:
            # Penalize excessively confident blank log probabilities: mean(P(blank))
            blank_probs = torch.exp(log_probs_t[:, :, self.blank_id])
            reg = torch.mean(blank_probs)
            return ctc_loss + (self.blank_penalty * reg)
        return ctc_loss

# =========================================================================
# 8. DECODING & HELPERS
# =========================================================================
def ctc_decode(log_probs, lengths, blank_id):
    """Greedy CTC collapse: argmax -> collapse adjacent duplicates -> remove blank."""
    preds = log_probs.argmax(dim=-1).cpu().numpy()
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
    targets = targets.cpu().numpy()
    target_lengths = target_lengths.cpu().numpy()
    true_seqs = []
    idx = 0
    for l in target_lengths:
        true_seqs.append(targets[idx:idx + l].tolist())
        idx += l
    return true_seqs

# =========================================================================
# 9. PLOTTING DIAGNOSTICS & STRUCTURE.MD GENERATION
# =========================================================================
def save_sentence_plots(history, plots_dir, phase_num):
    os.makedirs(plots_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # 1. 4-Panel Dashboard
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    axs[0, 0].plot(epochs, history["train_loss"], label="Train CTC Loss", color="#1f77b4", lw=2)
    axs[0, 0].plot(epochs, history["val_loss"], label="Val CTC Loss (Std)", color="#ff7f0e", lw=2, linestyle="--")
    if "val_loss_ema" in history and history["val_loss_ema"]:
        axs[0, 0].plot(epochs, history["val_loss_ema"], label="Val CTC Loss (EMA)", color="#2ca02c", lw=2, linestyle=":")
    axs[0, 0].set_title(f"Phase {phase_num}: CTC Loss (Lower is Better)")
    axs[0, 0].set_xlabel("Epoch"); axs[0, 0].grid(True, linestyle="--", alpha=0.6); axs[0, 0].legend()

    axs[0, 1].plot(epochs, [w * 100 for w in history["wer"]], label="WER (Std)", color="#d62728", lw=2)
    if "wer_ema" in history and history["wer_ema"]:
        axs[0, 1].plot(epochs, [w * 100 for w in history["wer_ema"]], label="WER (EMA)", color="#9467bd", lw=2, linestyle=":")
    axs[0, 1].set_title(f"Phase {phase_num}: Word Error Rate % (Lower is Better)")
    axs[0, 1].set_xlabel("Epoch"); axs[0, 1].grid(True, linestyle="--", alpha=0.6); axs[0, 1].legend()

    axs[1, 0].plot(epochs, [a * 100 for a in history["acc"]], label="Exact Sentence Acc", color="#2ca02c", lw=2)
    axs[1, 0].set_title("Exact Match Sentence Accuracy % (Higher is Better)")
    axs[1, 0].set_xlabel("Epoch"); axs[1, 0].grid(True, linestyle="--", alpha=0.6); axs[1, 0].legend()

    axs[1, 1].plot(epochs, history["f1"], label="Sequence F1", color="#8c564b", lw=2)
    axs[1, 1].set_title("Sequence F1 Score (Higher is Better)")
    axs[1, 1].set_xlabel("Epoch"); axs[1, 1].grid(True, linestyle="--", alpha=0.6); axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "ctc_metrics.png"), dpi=150)
    plt.close()

    # 2. Loss & WER Progression
    fig2, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(epochs, history["train_loss"], color="#1f77b4", label="Train Loss", lw=2)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("CTC Loss", color="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(epochs, [w * 100 for w in history["wer"]], color="#d62728", label="WER (%)", lw=2)
    ax2.set_ylabel("Word Error Rate (%)", color="#d62728")
    plt.title(f"Phase {phase_num} Sentence: CTC Loss & WER Progression")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "loss_and_wer.png"), dpi=150)
    plt.close()

    # 3. Learning Rate Progression
    if "lr_backbone" in history and history["lr_backbone"]:
        fig3, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, history["lr_backbone"], label="Backbone LR", color="#1f77b4")
        ax.plot(epochs, history["lr_head"], label="CTC Head LR", color="#ff7f0e")
        ax.set_title("Cosine Annealing LR Schedule")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate"); ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.6); ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "lr_schedule.png"), dpi=150)
        plt.close()

def write_phase_structure_md(target_dir, phase_num, source_dir):
    """Generates structure.md in the new phase output directory per project rules."""
    content = f"""# Directory Structure: Backend/models/islsentence_synthetic_phase{phase_num}

Model checkpoints, training history, and diagnostics for Phase {phase_num} unfrozen continuous sentence recognition.

## Files
- `classes.json`: Vocabulary gloss mapping and CTC blank token index.
- `history.json`: Epoch-by-epoch loss, Word Error Rate (WER), exact match accuracy, F1, and learning rates.
- `phase{phase_num}_sentence_best.pth`: Best checkpoint model weights achieving the lowest validation WER.
- `phase{phase_num}_sentence_ema_best.pth`: Exponential Moving Average (EMA) shadow model weights.
- `phase{phase_num}_sentence_checkpoint.pth`: Complete training state (optimizer, scheduler, epoch, EMA).
- `phase{phase_num}_sentence_final.pth`: Final model weights after completing all training epochs.
- `structure.md`: Directory structure documentation for this phase directory.

## Immediate Subdirectories
- `plots/`: Metric dashboards, loss curves, and WER progression charts.
"""
    with open(os.path.join(target_dir, "structure.md"), "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")

# =========================================================================
# 10. MAIN TRAINING SCRIPT
# =========================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2+ Continuous Sentence Unfrozen Trainer")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR_DEFAULT, help="Path to word tensors directory")
    parser.add_argument("--source_dir", type=str, default=None, help="Directory of source phase to initialize from (auto-detected if None)")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save next phase outputs (auto-detected if None)")
    parser.add_argument("--phase", type=int, default=None, help="Explicit target phase number (auto-detected if None)")
    parser.add_argument("--weights_path", type=str, default=None, help="Explicit weight file to load")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs to run in this phase")
    parser.add_argument("--train_samples", type=int, default=1400, help="Synthetic sentences per train epoch")
    parser.add_argument("--val_samples", type=int, default=200, help="Synthetic sentences per val epoch")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--backbone_lr", type=float, default=2.5e-5, help="Learning rate for unfrozen backbone")
    parser.add_argument("--head_lr", type=float, default=2.0e-4, help="Learning rate for CTC head")
    parser.add_argument("--eta_min", type=float, default=1.0e-6, help="Minimum LR for Cosine Annealing")
    parser.add_argument("--blank_penalty", type=float, default=0.01, help="Regularization penalty on blank token probability")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="Exponential moving average decay factor")
    parser.add_argument("--resume", action="store_true", help="Resume training within the current target phase if checkpoint exists")
    parser.add_argument("--num_workers",     type=int, default=2, help="DataLoader workers (0 for Windows)")
    return parser.parse_args()

def run_phase_training():
    args = parse_args()

    # 1. Auto-detect phases if not explicitly overridden
    latest_phase, detected_source_dir, next_phase, detected_target_dir = detect_sentence_phases(MODELS_ROOT)

    phase_num = args.phase if args.phase is not None else next_phase
    source_dir = args.source_dir if args.source_dir is not None else detected_source_dir
    target_dir = args.output_dir if args.output_dir is not None else detected_target_dir

    if source_dir is None:
        raise FileNotFoundError(f"Could not find any prior sentence phase in {MODELS_ROOT} to initialize from!")

    os.makedirs(target_dir, exist_ok=True)
    plots_dir = os.path.join(target_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    CHECKPOINT_FILE = os.path.join(target_dir, f"phase{phase_num}_sentence_checkpoint.pth")
    BEST_MODEL_FILE = os.path.join(target_dir, f"phase{phase_num}_sentence_best.pth")
    EMA_BEST_MODEL_FILE = os.path.join(target_dir, f"phase{phase_num}_sentence_ema_best.pth")
    FINAL_MODEL_FILE = os.path.join(target_dir, f"phase{phase_num}_sentence_final.pth")
    CLASSES_FILE = os.path.join(target_dir, "classes.json")
    HISTORY_FILE = os.path.join(target_dir, "history.json")

    # 2. Load Class Vocabulary
    source_classes_file = os.path.join(source_dir, "classes.json")
    if os.path.exists(source_classes_file):
        with open(source_classes_file, "r") as f:
            c_data = json.load(f)
            classes = c_data["classes"] if isinstance(c_data, dict) and "classes" in c_data else c_data
    elif os.path.exists(PHASE6_CLASSES_DEFAULT):
        with open(PHASE6_CLASSES_DEFAULT, "r") as f:
            classes = json.load(f)
    else:
        classes = sorted([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))])

    num_classes = len(classes)
    CTC_BLANK_ID = num_classes

    with open(CLASSES_FILE, "w", encoding="utf-8") as f:
        json.dump({"classes": classes, "num_classes": num_classes, "ctc_blank_id": CTC_BLANK_ID}, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print(f"ISL CONTINUOUS SENTENCE TRAINER - PHASE {phase_num}")
    print(f"Device:             {device}")
    print(f"Mode:               UNFROZEN END-TO-END DISCRIMINATIVE FINE-TUNING")
    print(f"Source Directory:   {source_dir}")
    print(f"Target Directory:   {target_dir}")
    print(f"Word Vocabulary:    {num_classes} classes (Blank ID: {CTC_BLANK_ID})")
    print(f"Backbone LR:        {args.backbone_lr:.2e} | CTC Head LR: {args.head_lr:.2e}")
    print(f"Scheduler:          CosineAnnealingLR (eta_min={args.eta_min:.2e})")
    print(f"Regularization:     Blank Penalty ({args.blank_penalty}) + Model EMA ({args.ema_decay})")
    print(f"Epochs to Run:      {args.epochs} | Train Samples: {args.train_samples} | Val: {args.val_samples}")
    print("=" * 70)

    # 3. Instantiate Model
    model = ISLContinuousSentenceModel(num_classes=num_classes, d=128, gru_layers=2, n_head=4, n_trans=2, drop=0.15).to(device)

    # Determine weight checkpoint to load
    start_epoch = 0
    best_wer = float("inf")
    best_loss = float("inf")
    best_ema_wer = float("inf")
    history = {
        "train_loss": [], "val_loss": [], "val_loss_ema": [],
        "wer": [], "wer_ema": [], "acc": [], "f1": [],
        "lr_backbone": [], "lr_head": []
    }

    loaded_weights = False
    if args.resume and os.path.exists(CHECKPOINT_FILE):
        print(f"[RESUME] Loading in-progress checkpoint from {CHECKPOINT_FILE}")
        ckpt = torch.load(CHECKPOINT_FILE, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            load_flexible(model, extract_state_dict(ckpt), label="RESUME")
            start_epoch = ckpt.get("epoch", 0)
            best_wer = ckpt.get("val_wer", float("inf"))
            best_loss = ckpt.get("val_loss", float("inf"))
            best_ema_wer = ckpt.get("val_ema_wer", float("inf"))
            history = ckpt.get("history", history)
            loaded_weights = True
            print(f"[RESUME] Resuming from Epoch {start_epoch} with prior best WER: {best_wer*100:.2f}%")
    
    if not loaded_weights:
        weights_to_load = args.weights_path if args.weights_path else find_best_weights_in_phase(source_dir, latest_phase)
        if weights_to_load and os.path.exists(weights_to_load):
            print(f"[LOAD] Initializing Phase {phase_num} from source weights: {weights_to_load}")
            raw = torch.load(weights_to_load, map_location=device, weights_only=False)
            sd = raw["model_state_dict"] if (isinstance(raw, dict) and "model_state_dict" in raw) else raw
            filtered = {k: v for k, v in sd.items() if not k.startswith("head.") and k != "pe.pe"}
            missing, unexpected = load_flexible(model, filtered, label="SOURCE")
            print(f"[LOAD] Weights restored. ({len(filtered)} keys loaded, missing: {len(missing)})")
            loaded_weights = True
        else:
            print(f"[WARNING] No weights found in source {source_dir}! Training from scratch initialization.")

    # 4. Unfreeze All Layers with Differential Learning Rates
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        param.requires_grad = True
        if name.startswith("ctc_head."):
            head_params.append(param)
        else:
            backbone_params.append(param)

    n_bb = sum(p.numel() for p in backbone_params)
    n_head = sum(p.numel() for p in head_params)
    print(f"[OPTIMIZER] Unfrozen parameter breakdown:")
    print(f"            - Backbone: {n_bb:,} params @ LR={args.backbone_lr:.2e}")
    print(f"            - CTC Head: {n_head:,} params @ LR={args.head_lr:.2e}")

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": args.backbone_lr, "weight_decay": 1e-4},
        {"params": head_params, "lr": args.head_lr, "weight_decay": 1e-3}
    ])

    total_epochs = start_epoch + args.epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
    criterion = RegularizedCTCLoss(blank_id=CTC_BLANK_ID, blank_penalty=args.blank_penalty)

    # Initialize Model EMA
    ema = ModelEMA(model, decay=args.ema_decay)

    # 5. Datasets & Loaders
    train_ds = SyntheticSentenceDataset(args.data_dir, classes, samples_per_epoch=args.train_samples, is_train=True)
    val_ds = SyntheticSentenceDataset(args.data_dir, classes, samples_per_epoch=args.val_samples, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn_sentence, num_workers=args.num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn_sentence, num_workers=args.num_workers, pin_memory=False)

    write_phase_structure_md(target_dir, phase_num, source_dir)

    # 6. Training Loop
    for epoch in range(start_epoch, total_epochs):
        model.train()
        total_train_loss = 0.0

        cur_bb_lr = optimizer.param_groups[0]["lr"]
        cur_head_lr = optimizer.param_groups[1]["lr"]
        history["lr_backbone"].append(cur_bb_lr)
        history["lr_head"].append(cur_head_lr)

        pbar = tqdm(train_loader, desc=f"Ep {epoch+1:02d}/{total_epochs:02d} [Phase {phase_num} Train]")
        for x, lengths, targets, target_lengths in pbar:
            x, lengths = x.to(device), lengths.to(device)
            targets, target_lengths = targets.to(device), target_lengths.to(device)

            optimizer.zero_grad()
            log_probs = model(x, lengths)  # (B, T, C)
            log_probs_t = log_probs.transpose(0, 1)  # (T, B, C)

            loss = criterion(log_probs_t, targets, lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(model)

            total_train_loss += loss.item()
            pbar.set_postfix({"ctc_loss": f"{loss.item():.4f}", "bb_lr": f"{cur_bb_lr:.1e}"})

        train_loss = total_train_loss / max(1, len(train_loader))

        # Evaluation (Standard Model)
        model.eval()
        total_val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, lengths, targets, target_lengths in tqdm(val_loader, desc=f"Ep {epoch+1:02d}/{total_epochs:02d} [Std Val]"):
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

        # Compute Metrics (Standard)
        epoch_wer = sum(compute_wer(ref, hyp) for ref, hyp in zip(all_targets, all_preds)) / max(1, len(all_targets))
        epoch_acc = sum(1.0 for ref, hyp in zip(all_targets, all_preds) if ref == hyp) / max(1, len(all_targets))
        epoch_f1 = sum(compute_f1_prec_rec(ref, hyp)[0] for ref, hyp in zip(all_targets, all_preds)) / max(1, len(all_targets))

        # Evaluation (EMA Model)
        ema.model.eval()
        total_ema_loss = 0.0
        ema_preds = []
        with torch.no_grad():
            for x, lengths, targets, target_lengths in val_loader:
                x, lengths = x.to(device), lengths.to(device)
                targets_dev, target_lengths_dev = targets.to(device), target_lengths.to(device)

                log_probs = ema.model(x, lengths)
                log_probs_t = log_probs.transpose(0, 1)
                v_loss = criterion(log_probs_t, targets_dev, lengths, target_lengths_dev)
                total_ema_loss += v_loss.item()

                decoded = ctc_decode(log_probs, lengths, blank_id=CTC_BLANK_ID)
                ema_preds.extend(decoded)

        val_loss_ema = total_ema_loss / max(1, len(val_loader))
        epoch_wer_ema = sum(compute_wer(ref, hyp) for ref, hyp in zip(all_targets, ema_preds)) / max(1, len(all_targets))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_loss_ema"].append(val_loss_ema)
        history["wer"].append(epoch_wer)
        history["wer_ema"].append(epoch_wer_ema)
        history["acc"].append(epoch_acc)
        history["f1"].append(epoch_f1)

        scheduler.step()

        print(f"\n[PHASE {phase_num} - EPOCH {epoch+1:02d}/{total_epochs:02d} SUMMARY]")
        print(f"  Train CTC Loss: {train_loss:.4f} | Val CTC Loss: {val_loss:.4f} (EMA: {val_loss_ema:.4f})")
        print(f"  WER (Standard): {epoch_wer * 100:.2f}% | WER (EMA): {epoch_wer_ema * 100:.2f}%")
        print(f"  Exact Sentence Acc: {epoch_acc * 100:.2f}% | Sequence F1: {epoch_f1:.4f}")

        # Save Checkpoint State
        checkpoint_data = {
            "epoch": epoch + 1,
            "phase": phase_num,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "val_wer": epoch_wer,
            "val_ema_wer": epoch_wer_ema,
            "val_f1": epoch_f1,
            "history": history,
            "classes": classes,
            "num_classes": num_classes,
            "ctc_blank_id": CTC_BLANK_ID
        }
        torch.save(checkpoint_data, CHECKPOINT_FILE)

        # Track and Save Best Standard Model
        if epoch_wer < best_wer or (epoch_wer == best_wer and val_loss < best_loss):
            best_wer = epoch_wer
            best_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_FILE)
            print(f"  * [NEW BEST MODEL] Saved -> {BEST_MODEL_FILE} (WER: {epoch_wer * 100:.2f}%)")

        # Track and Save Best EMA Model
        if epoch_wer_ema < best_ema_wer:
            best_ema_wer = epoch_wer_ema
            torch.save(ema.model.state_dict(), EMA_BEST_MODEL_FILE)
            print(f"  * [NEW BEST EMA MODEL] Saved -> {EMA_BEST_MODEL_FILE} (WER: {epoch_wer_ema * 100:.2f}%)")

        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        save_sentence_plots(history, plots_dir, phase_num)
        gc.collect()
        torch.cuda.empty_cache()

    torch.save(model.state_dict(), FINAL_MODEL_FILE)
    write_phase_structure_md(target_dir, phase_num, source_dir)

    print("\n" + "=" * 70)
    print(f"PHASE {phase_num} CONTINUOUS SENTENCE TRAINING COMPLETE!")
    print(f"Best Standard Model: {BEST_MODEL_FILE} (Best WER: {best_wer*100:.2f}%)")
    print(f"Best EMA Model:      {EMA_BEST_MODEL_FILE} (Best WER: {best_ema_wer*100:.2f}%)")
    print(f"Final Weights:       {FINAL_MODEL_FILE}")
    print(f"Diagnostic Plots:    {plots_dir}")
    print("=" * 70)

if __name__ == "__main__":
    run_phase_training()
