"""Phase 6 - High-Precision Squeeze Trainer.

Designed to escape local minima from Phase 5/4 checkpoints and maximize accuracy in 20-30 epochs.

Key Architectural & Training Features:
-------------------------------------
1. Split Integrity:
   - Reuses the EXACT train/val split from models/phase4_word_model/split_paths.json.
   - Zero data leakage between phases.
2. Checkpoint Warm-Start:
   - Initializes directly from Phase 5's best checkpoint (best_phase5_model.pth).
3. Optimizer - Lookahead + Decoupled AdamW:
   - Lookahead (k=5, alpha=0.5) wrapping AdamW with separate weight decay for
     weights vs 1D normalization/biases.
   - Breaks out of sharp local minima into flatter, higher-generalization basins.
4. Annealed LR Combined with ReduceLROnPlateau:
   - Warmup (2 epochs) -> Cosine Annealing decay down to min_lr.
   - Real-time Plateau Detector: if validation accuracy does not improve for
     `patience=3` epochs, cuts learning rate by factor 0.70 to tighten gradient steps.
5. Model EMA (Exponential Moving Average):
   - Maintains an EMA shadow model (decay=0.995) during training.
   - Saves both best standard and best EMA checkpoints.
6. Regularization & Augmentation:
   - Time-stretch (0.75x - 1.25x)
   - Bilateral Pose Flip (with exact left/right joint swapping)
   - Random 2D Pose Tilt (+-15 deg) + subtle joint coordinate jitter
   - Temporal Frame Masking (SpecAugment for video sequence: drops 1-3 continuous frames)
   - Gentle Mixup (alpha=0.15, p=0.30)
   - Label smoothing (0.05) & gradient clipping (1.0)
7. Outputs & Plots:
   - Saved to: models/phase6_word_model/
   - Plots in: models/phase6_word_model/plots/
"""

import os, sys, glob, json, math, random, copy, collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# =========================================================================
# PATHS
# =========================================================================
BACKEND_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_JSON     = os.path.join(BACKEND_DIR, "models", "phase4_word_model", "split_paths.json")
PHASE5_CKPT    = os.path.join(BACKEND_DIR, "models", "phase5_word_model", "best_phase5_model.pth")
PHASE4_CKPT    = os.path.join(BACKEND_DIR, "models", "phase4_word_model", "best_phase4_model.pth")
SAVE_DIR       = os.path.join(BACKEND_DIR, "models", "phase6_word_model")
PLOTS_DIR      = os.path.join(SAVE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from utils.pose_augmentation import (  # noqa: E402
    AugmentConfig,
    augment_pose_sequence as augment_pose_sequence_canonical,
    mirror_pose as canonical_mirror,
    shift_pose as canonical_shift,
    tilt_pose as canonical_tilt,
    temporal_mask as canonical_mask,
    time_stretch as canonical_time_stretch,
)
from utils.checkpoint_utils import load_flexible  # noqa: E402
from utils.dataloader_word import get_word_classes  # noqa: E402
from utils.splits import load_canonical_split, split_fingerprint  # noqa: E402

# =========================================================================
# HYPER-PARAMETERS (25 EPOCHS SQUEEZE)
# =========================================================================
INPUT_DIM      = 163
D_MODEL        = 128
N_HEAD         = 4
N_TRANS        = 2
GRU_LAYERS     = 2

TOTAL_EPOCHS   = 25
WARMUP_EPOCHS  = 2
BATCH_SIZE     = 48
WEIGHT_DECAY   = 1e-4
LABEL_SMOOTH   = 0.05
GRAD_CLIP      = 1.0

# Differential Learning Rates
LR_TRANS       = 1.2e-5
LR_RNN_CNN     = 2.5e-5
LR_HEAD        = 5.0e-5
MIN_LR         = 1.0e-7

# Plateau Detection Parameters
PLATEAU_PATIENCE = 3
PLATEAU_FACTOR   = 0.70

# Lookahead Optimizer Parameters
LOOKAHEAD_K      = 5
LOOKAHEAD_ALPHA  = 0.50

# Model EMA Parameters
EMA_DECAY        = 0.995

# Augmentation Parameters
STRETCH_LO     = 0.75
STRETCH_HI     = 1.25
FLIP_P         = 0.35
TILT_DEG       = 15.0
SHIFT_X_MAX    = 0.08        # Horizontal left/right shift from centre per sequence
SHIFT_Y_MAX    = 0.03        # Subtle vertical shift per sequence
NOISE_STD      = 0.005
MIXUP_ALPHA    = 0.15
MIXUP_P        = 0.30
MASK_P         = 0.30
MAX_MASK_LEN   = 3

# The canonical augmentation set (Backend/utils/pose_augmentation.py).
PHASE6_AUGMENT = AugmentConfig(
    flip_prob=FLIP_P,
    tilt_prob=1.0,
    tilt_deg=TILT_DEG,
    shift_prob=0.5,
    shift_dx=SHIFT_X_MAX,
    shift_dy=SHIFT_Y_MAX,
    speed_prob=0.5,
    speed_lo=STRETCH_LO,
    speed_hi=STRETCH_HI,
    time_shift_prob=0.2,
    time_shift_max=3,
    mask_prob=MASK_P,
    mask_max=MAX_MASK_LEN,
    mask_count=1,
    noise_std=NOISE_STD,
)


# =========================================================================
# 1. AUGMENTATION
# =========================================================================

def time_stretch(seq):
    """Speed up / slow down the clip (canonical implementation).

    Only the 159 landmark channels are interpolated; the presence flags are
    resampled with nearest neighbour and re-thresholded, otherwise the flags
    become fractional and the missing-modality bookkeeping breaks.
    """
    return canonical_time_stretch(seq, STRETCH_LO, STRETCH_HI)


def flip_pose(seq):
    """Mirror around the signer's own axis (canonical implementation)."""
    return canonical_mirror(seq)


def tilt_pose(seq):
    """Rotate around the neck (canonical implementation)."""
    return canonical_tilt(seq, TILT_DEG)


def temporal_frame_mask(seq, max_mask_len=MAX_MASK_LEN, p=MASK_P):
    """SpecAugment-style temporal cutout (canonical implementation)."""
    if random.random() > p or seq.shape[0] <= 5:
        return seq
    return canonical_mask(seq, max_mask_len, 1)


def shift_pose(seq, shift_x_max=SHIFT_X_MAX, shift_y_max=SHIFT_Y_MAX):
    """Bounded left/right shift (canonical implementation).

    The magnitude is expressed in shoulder widths, so the landmarks can never
    be pushed out of the frame.
    """
    return canonical_shift(seq, shift_x_max, shift_y_max)


def augment(seq):
    """The canonical augmentation set (Backend/utils/pose_augmentation.py).

    The old local copy rotated around the origin (the corpus is anchor
    relative, so the body sat at x ~ +3.5 shoulder widths) and re-clamped the
    missing hands to the resting placeholder, which is NOT the encoding of
    this corpus (a missing modality owns a ZERO block here).
    """
    return augment_pose_sequence_canonical(seq, PHASE6_AUGMENT)


def mixup_criterion(crit, pred, y_a, y_b, lam):
    return lam * crit(pred, y_a) + (1.0 - lam) * crit(pred, y_b)


# =========================================================================
# 2. DATASET & COLLATE
# =========================================================================

def collate_pad(batch):
    xs, ys = zip(*batch)
    max_len = max(x.shape[0] for x in xs)
    feat_dim = xs[0].shape[1]
    padded = torch.zeros(len(xs), max_len, feat_dim, dtype=torch.float32)
    for i, x in enumerate(xs):
        padded[i, :x.shape[0]] = x
    return padded, torch.tensor(ys, dtype=torch.long)


class TrainDS(Dataset):
    def __init__(self, split: dict, c2i: dict):
        self.items = []
        for cls, paths in split["train"].items():
            idx = c2i[cls]
            for p in paths:
                self.items.append((p, idx))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        seq = torch.load(path, map_location="cpu", weights_only=True).float()
        seq = augment(seq)
        return seq, label


class ValDS(Dataset):
    def __init__(self, split: dict, c2i: dict):
        self.items = []
        for cls, paths in split["val"].items():
            idx = c2i[cls]
            for p in paths:
                self.items.append((p, idx))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        seq = torch.load(path, map_location="cpu", weights_only=True).float()
        return seq, label


# =========================================================================
# 3. MODEL ARCHITECTURE
# =========================================================================

class PosEnc(nn.Module):
    def __init__(self, d, maxlen=512):
        super().__init__()
        pe  = torch.zeros(maxlen, d)
        pos = torch.arange(0, maxlen).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TCNN(nn.Module):
    def __init__(self, ic, oc, k=3, drop=0.1):
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


class Phase6Model(nn.Module):
    def __init__(self, nc, d=D_MODEL, gru=GRU_LAYERS,
                 nh=N_HEAD, nt=N_TRANS, drop=0.15):
        super().__init__()
        self.proj  = nn.Sequential(
            nn.Linear(INPUT_DIM, d), nn.LayerNorm(d), nn.Dropout(drop))
        self.gru   = nn.GRU(d, d // 2, gru, batch_first=True, bidirectional=True,
                            dropout=drop if gru > 1 else 0.0)
        self.gnorm = nn.LayerNorm(d)
        self.pe    = PosEnc(d)
        el = nn.TransformerEncoderLayer(d, nh, d * 2, drop, "gelu", batch_first=True)
        self.trans = nn.TransformerEncoder(el, nt, enable_nested_tensor=False)
        self.tnorm = nn.LayerNorm(d)
        self.t1    = TCNN(d, d, drop=drop)
        self.t2    = TCNN(d, d * 2, drop=drop)
        self.ap    = nn.AdaptiveAvgPool1d(1)
        self.mp    = nn.AdaptiveMaxPool1d(1)
        self.head  = nn.Sequential(
            nn.Linear(d * 4, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Dropout(drop * 2), nn.Linear(256, nc))

    def transformer_params(self):
        return (list(self.pe.parameters()) + list(self.trans.parameters()) +
                list(self.tnorm.parameters()))

    def rnn_params(self):
        return (list(self.proj.parameters()) + list(self.gru.parameters()) +
                list(self.gnorm.parameters()))

    def cnn_params(self):
        return list(self.t1.parameters()) + list(self.t2.parameters())

    def head_params(self):
        return list(self.head.parameters())

    def forward(self, x):
        self.gru.flatten_parameters()
        h = self.proj(x)
        g, _ = self.gru(h)
        h = self.gnorm(h + g)
        h = self.pe(h)
        h = self.trans(h)
        h = self.tnorm(h)
        h = h.transpose(1, 2)
        h = self.t1(h)
        h = self.t2(h)
        p = torch.cat([self.ap(h).squeeze(-1), self.mp(h).squeeze(-1)], -1)
        return self.head(p)


# =========================================================================
# 4. LOOKAHEAD OPTIMIZER & MODEL EMA
# =========================================================================

class Lookahead(optim.Optimizer):
    """
    Lookahead Optimizer (Zhang et al.)
    Iteratively updates slow weights every k steps towards fast weights:
    theta_slow = theta_slow + alpha * (theta_fast - theta_slow)
    Helps escape sharp local minima and improves generalization.
    """
    def __init__(self, base_optimizer, k=5, alpha=0.5):
        self.optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = collections.defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0
        self.slow_weights = [
            [p.clone().detach() for p in group["params"]]
            for group in self.param_groups
        ]

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group, slow_weights in zip(self.param_groups, self.slow_weights):
            group["counter"] += 1
            if group["counter"] % self.k == 0:
                for p, q in zip(group["params"], slow_weights):
                    if p.grad is None:
                        continue
                    q.data.add_(p.data - q.data, alpha=self.alpha)
                    p.data.copy_(q.data)
        return loss

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)


class ModelEMA:
    """Maintains Exponential Moving Average of model parameters."""
    def __init__(self, model, decay=EMA_DECAY):
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.decay = decay
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)
            for ema_buf, buf in zip(self.ema_model.buffers(), model.buffers()):
                ema_buf.copy_(buf)


# =========================================================================
# 5. HYBRID ANNEALED + REDUCE LR ON PLATEAU SCHEDULER
# =========================================================================

class AnnealedPlateauScheduler:
    """
    Combines:
    1. Linear warmup for `warmup_epochs`.
    2. Cosine Annealing decay down to `min_lr` over `total_epochs`.
    3. ReduceLROnPlateau adaptive factor: if validation metric fails to improve
       for `patience` consecutive epochs, scales down current learning rates by
       `plateau_factor` to settle into finer minima.
    """
    def __init__(self, optimizer, total_epochs, warmup_epochs=WARMUP_EPOCHS,
                 min_lr=MIN_LR, patience=PLATEAU_PATIENCE,
                 plateau_factor=PLATEAU_FACTOR, mode='max'):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.patience = patience
        self.plateau_factor = plateau_factor
        self.mode = mode

        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.plateau_scale = 1.0
        self.best_metric = -float('inf') if mode == 'max' else float('inf')
        self.bad_epochs = 0
        self.current_epoch = 0

    def step_epoch(self, val_metric=None):
        self.current_epoch += 1
        plateau_triggered = False

        if val_metric is not None:
            improved = ((val_metric > self.best_metric + 1e-4) if self.mode == 'max'
                        else (val_metric < self.best_metric - 1e-4))
            if improved:
                self.best_metric = val_metric
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1
                if self.bad_epochs >= self.patience:
                    self.plateau_scale = max(0.05, self.plateau_scale * self.plateau_factor)
                    self.bad_epochs = 0
                    plateau_triggered = True

        # Cosine Annealing factor with Warmup
        if self.current_epoch <= self.warmup_epochs:
            warmup_factor = self.current_epoch / max(1, self.warmup_epochs)
            cosine_factor = 0.25 + 0.75 * warmup_factor
        else:
            progress = (self.current_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

        # Apply to each parameter group
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            lr = self.min_lr + (base_lr - self.min_lr) * cosine_factor * self.plateau_scale
            pg['lr'] = max(self.min_lr, lr)

        return plateau_triggered


# =========================================================================
# 6. WEIGHT LOADING
# =========================================================================

def build_model(nc):
    model = Phase6Model(nc)
    ckpt_path = None
    if os.path.exists(PHASE5_CKPT):
        ckpt_path = PHASE5_CKPT
        print(f"[INFO] Initializing from Phase 5 checkpoint: {PHASE5_CKPT}")
    elif os.path.exists(PHASE4_CKPT):
        ckpt_path = PHASE4_CKPT
        print(f"[INFO] Initializing from Phase 4 checkpoint: {PHASE4_CKPT}")
    else:
        print("[WARN] No prior checkpoint found -- starting from random initialization")
        return model

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    miss, unex = load_flexible(model, raw, label="PHASE5->PHASE6")
    matched = len(raw) - len(miss)
    print(f"[INFO] Weights successfully loaded: {matched}/{len(raw)} keys matched.")
    if miss:
        print(f"  Missing:    {miss}")
    if unex:
        print(f"  Unexpected: {unex}")
    return model


# =========================================================================
# 7. PLOTTING FUNCTIONS
# =========================================================================

BG  = "#0f1117"; PAN = "#1a1d27"; TXT = "#e6edf3"; MUT = "#8b949e"; BRD = "#30363d"
LEG = dict(facecolor="#21262d", labelcolor="#c9d1d9", edgecolor="#30363d", fontsize=9)

def sax(ax):
    ax.set_facecolor(PAN)
    ax.tick_params(colors=MUT)
    for s in ax.spines.values():
        s.set_edgecolor(BRD)
    ax.grid(True, ls="--", alpha=0.2, color=MUT)

def plot_loss_acc(h):
    ep = range(1, len(h["tl"]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BG)

    # Loss
    sax(a1)
    a1.plot(ep, h["tl"], label="Train Loss", color="#58a6ff", lw=2)
    a1.plot(ep, h["vl"], label="Val Loss",   color="#f78166", lw=2)
    a1.set_title("Cross-Entropy Loss (Label Smoothed)", color=TXT, fontsize=11, fontweight="bold")
    a1.set_xlabel("Epoch", color=MUT); a1.set_ylabel("Loss", color=MUT)
    a1.legend(**LEG)

    # Accuracy
    sax(a2)
    a2.plot(ep, [a * 100 for a in h["ta"]], label="Train Acc", color="#58a6ff", lw=1.5, ls="--")
    a2.plot(ep, [a * 100 for a in h["va"]], label="Val Acc (Std)", color="#3fb950", lw=2)
    if "ema_va" in h and h["ema_va"]:
        a2.plot(ep, [a * 100 for a in h["ema_va"]], label="Val Acc (EMA)", color="#d29922", lw=2)
    for p_ep in h.get("plateau_epochs", []):
        a2.axvline(p_ep, color="#f85149", ls=":", alpha=0.7, label="Plateau Drop" if p_ep == h["plateau_epochs"][0] else "")
    a2.set_title("Accuracy (%): Standard vs EMA", color=TXT, fontsize=11, fontweight="bold")
    a2.set_xlabel("Epoch", color=MUT); a2.set_ylabel("Accuracy (%)", color=MUT)
    a2.legend(**LEG)

    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "loss_acc.png")
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  [SAVED] {p}")

def plot_metrics(h):
    ep = range(1, len(h["vf1"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG); sax(ax)
    ax.plot(ep, h["vf1"], label="Weighted F1",  color="#bc8cff", lw=2)
    ax.plot(ep, h["vp"],  label="Precision",    color="#39d353", lw=1.5, ls="--")
    ax.plot(ep, h["vr"],  label="Recall",       color="#e3b341", lw=1.5, ls=":")
    ax.set_title("Validation Classification Metrics (Standard Model)", color=TXT, fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", color=MUT); ax.set_ylabel("Score", color=MUT)
    ax.legend(**LEG)
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "metrics.png")
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  [SAVED] {p}")

def plot_lr(h):
    ep = range(1, len(h["lr_head"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(BG); sax(ax)
    ax.plot(ep, h["lr_trans"], label="Transformer LR", color="#bc8cff", lw=2)
    ax.plot(ep, h["lr_rnn"],   label="RNN+CNN LR",     color="#58a6ff", lw=2)
    ax.plot(ep, h["lr_head"],  label="Head LR",        color="#f0883e", lw=2)
    for p_ep in h.get("plateau_epochs", []):
        ax.axvline(p_ep, color="#f85149", ls=":", alpha=0.7)
    ax.set_title("Annealed LR with ReduceLROnPlateau Drops", color=TXT, fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", color=MUT); ax.set_ylabel("Learning Rate", color=MUT)
    ax.set_yscale("log")
    ax.legend(**LEG)
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "lr_schedule.png")
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  [SAVED] {p}")

def plot_ema_compare(h):
    if "ema_va" not in h or not h["ema_va"]:
        return
    ep = range(1, len(h["va"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG); sax(ax)
    std_acc = [a * 100 for a in h["va"]]
    ema_acc = [a * 100 for a in h["ema_va"]]
    ax.plot(ep, std_acc, label="Standard Model Val Acc", color="#58a6ff", lw=2)
    ax.plot(ep, ema_acc, label="EMA Model Val Acc", color="#3fb950", lw=2)
    ax.fill_between(ep, std_acc, ema_acc, where=[e >= s for e, s in zip(ema_acc, std_acc)],
                    color="#3fb950", alpha=0.15, label="EMA Advantage")
    ax.set_title("EMA vs Standard Model Val Accuracy", color=TXT, fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", color=MUT); ax.set_ylabel("Accuracy (%)", color=MUT)
    ax.legend(**LEG)
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "ema_vs_standard.png")
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  [SAVED] {p}")

def plot_cm(yt, yp, classes, tag=""):
    cm = confusion_matrix(yt, yp)
    n = len(classes)
    sz = max(12, n * 0.07)
    fig, ax = plt.subplots(figsize=(sz, sz * 0.9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PAN)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title(f"Confusion Matrix — Phase 6 {tag} ({n} classes)",
                 color=TXT, fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted", color=MUT); ax.set_ylabel("True", color=MUT)
    fs = max(3.5, 6.5 - n * 0.012)
    ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n))
    ax.set_xticklabels(classes, rotation=90, fontsize=fs, color="#c9d1d9")
    ax.set_yticklabels(classes, fontsize=fs, color="#c9d1d9")
    plt.tight_layout()
    fname = f"confusion_matrix{'_' + tag if tag else ''}.png"
    p = os.path.join(PLOTS_DIR, fname)
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  [SAVED] {p}")

def gen_plots(h, yt_b, yp_b, classes, yt_e=None, yp_e=None):
    print("[INFO] Generating diagnostic plots ...")
    plot_loss_acc(h)
    plot_metrics(h)
    plot_lr(h)
    plot_ema_compare(h)
    plot_cm(yt_b, yp_b, classes, "best")
    if yt_e is not None and yp_e is not None:
        plot_cm(yt_e, yp_e, classes, "ema")


# =========================================================================
# 8. EVALUATION ROUTINE
# =========================================================================

def evaluate(model, loader, crit, device, dataset_len):
    model.eval()
    vl = 0.0
    vp = []
    vt_ = []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            lo = model(X)
            vl += crit(lo, y).item() * X.size(0)
            vp.extend(lo.argmax(-1).cpu().numpy())
            vt_.extend(y.cpu().numpy())
    vl /= max(1, dataset_len)
    va  = accuracy_score(vt_, vp)
    vpr, vrc, vf1, _ = precision_recall_fscore_support(
        vt_, vp, average="weighted", zero_division=0)
    return vl, va, vpr, vrc, vf1, vt_, vp


# =========================================================================
# 9. MAIN TRAINING LOOP
# =========================================================================

def train_phase6():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using Device: {device}")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # --- Load the CANONICAL split (the same assignment as every other phase) ---
    manifest = load_canonical_split()
    split = {"classes": manifest["classes"], "train": manifest["train"], "val": manifest["val"]}
    classes = split["classes"]
    c2i = {c: i for i, c in enumerate(classes)}
    nc  = len(classes)
    print(f"[SPLIT] Canonical manifest {split_fingerprint(manifest)[:12]}: "
          f"{nc} classes, {sum(len(v) for v in split['train'].values())} train, "
          f"{sum(len(v) for v in split['val'].values())} val")

    # The split defines the vocabulary of this phase, so it must match the
    # vocabulary of the word loader.  A stale split (from the old 262-class
    # corpus) would silently train a model on a different class list.
    expected = get_word_classes()
    if set(classes) != set(expected):
        raise RuntimeError(
            f"[ERROR] The Phase-4 split is stale: it holds {len(classes)} classes "
            f"but the word loader selects {len(expected)}.\n"
            f"       Regenerate it with:  python scripts/save_phase4_split.py"
        )

    n_train = sum(len(v) for v in split['train'].values())
    n_val   = sum(len(v) for v in split['val'].values())
    print(f"[INFO] Loaded split: {nc} classes | {n_train} train samples | {n_val} val samples")

    tr_ds = TrainDS(split, c2i)
    va_ds = ValDS(split, c2i)
    tr_ld = DataLoader(tr_ds, BATCH_SIZE, shuffle=True, drop_last=True,
                       num_workers=0, collate_fn=collate_pad)
    va_ld = DataLoader(va_ds, BATCH_SIZE, shuffle=False,
                       num_workers=0, collate_fn=collate_pad)

    # --- Model Setup ---
    model = build_model(nc).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Phase6Model total parameters: {total_p:,}")

    # --- Optimizers: Decoupled Weight Decay & Lookahead ---
    # Separate parameters with weight decay vs no decay (biases, norms)
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndimension() <= 1 or "norm" in name.lower() or "bias" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    base_opt = optim.AdamW([
        {"params": [p for p in model.transformer_params() if p in set(decay_params)],
         "lr": LR_TRANS, "weight_decay": WEIGHT_DECAY, "name": "trans"},
        {"params": [p for p in model.transformer_params() if p in set(no_decay_params)],
         "lr": LR_TRANS, "weight_decay": 0.0, "name": "trans_nd"},
        {"params": [p for p in model.rnn_params() if p in set(decay_params)],
         "lr": LR_RNN_CNN, "weight_decay": WEIGHT_DECAY, "name": "rnn"},
        {"params": [p for p in model.rnn_params() if p in set(no_decay_params)],
         "lr": LR_RNN_CNN, "weight_decay": 0.0, "name": "rnn_nd"},
        {"params": [p for p in model.cnn_params() if p in set(decay_params)],
         "lr": LR_RNN_CNN, "weight_decay": WEIGHT_DECAY, "name": "cnn"},
        {"params": [p for p in model.cnn_params() if p in set(no_decay_params)],
         "lr": LR_RNN_CNN, "weight_decay": 0.0, "name": "cnn_nd"},
        {"params": [p for p in model.head_params() if p in set(decay_params)],
         "lr": LR_HEAD, "weight_decay": WEIGHT_DECAY, "name": "head"},
        {"params": [p for p in model.head_params() if p in set(no_decay_params)],
         "lr": LR_HEAD, "weight_decay": 0.0, "name": "head_nd"},
    ])

    # Wrap with Lookahead to break out of sharp local minima
    opt = Lookahead(base_opt, k=LOOKAHEAD_K, alpha=LOOKAHEAD_ALPHA)

    # Hybrid Annealed + ReduceLROnPlateau Scheduler
    sch = AnnealedPlateauScheduler(
        opt,
        total_epochs=TOTAL_EPOCHS,
        warmup_epochs=WARMUP_EPOCHS,
        min_lr=MIN_LR,
        patience=PLATEAU_PATIENCE,
        plateau_factor=PLATEAU_FACTOR,
        mode='max'
    )

    # Model Exponential Moving Average
    ema = ModelEMA(model, decay=EMA_DECAY)

    crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    hist = {k: [] for k in ["tl", "ta", "vl", "va", "vp", "vr", "vf1",
                            "lr_trans", "lr_rnn", "lr_head", "ema_va", "plateau_epochs"]}
    best_std_acc = 0.0
    best_ema_acc = 0.0

    best_p = os.path.join(SAVE_DIR, "best_phase6_model.pth")
    ema_p  = os.path.join(SAVE_DIR, "ema_phase6_model.pth")
    final_p = os.path.join(SAVE_DIR, "final_phase6_model.pth")

    def _get_lr(name):
        for pg in opt.param_groups:
            if pg.get("name") == name:
                return pg["lr"]
        return opt.param_groups[0]["lr"]

    print(f"\n[START] Phase 6 Squeeze -- {TOTAL_EPOCHS} epochs | Lookahead(k={LOOKAHEAD_K}) | Model EMA | Annealed+Plateau LR\n")

    for ep in range(1, TOTAL_EPOCHS + 1):
        # Record current LRs
        hist["lr_trans"].append(_get_lr("trans"))
        hist["lr_rnn"].append(_get_lr("rnn"))
        hist["lr_head"].append(_get_lr("head"))

        # --- Training Loop ---
        model.train()
        tl = 0.0
        tc = 0
        tn = 0
        bar = tqdm(tr_ld, desc=f"E{ep:02d}/{TOTAL_EPOCHS} [Squeeze]", leave=False, ncols=95)

        for X, y in bar:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()

            # Gentle Mixup regularization
            if MIXUP_ALPHA > 0 and random.random() < MIXUP_P:
                idx2 = torch.randperm(X.size(0), device=device)
                X2, y2 = X[idx2], y[idx2]
                lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
                X_mix = lam * X + (1.0 - lam) * X2
                lo = model(X_mix)
                ls = mixup_criterion(crit, lo, y, y2, lam)
                with torch.no_grad():
                    lo_raw = model(X)
                tc += (lo_raw.argmax(-1) == y).sum().item()
            else:
                lo = model(X)
                ls = crit(lo, y)
                tc += (lo.argmax(-1) == y).sum().item()

            ls.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            # Update Model EMA
            ema.update(model)

            tl += ls.item() * X.size(0)
            tn += X.size(0)
            bar.set_postfix(loss=f"{ls.item():.3f}")

        tl /= max(1, tn)
        ta = tc / max(1, tn)

        # --- Validate Standard Model ---
        vl, va, vpr, vrc, vf1, vt_, vp_ = evaluate(model, va_ld, crit, device, len(va_ds))

        # --- Validate EMA Model ---
        _, ema_va, _, _, ema_vf1, _, _ = evaluate(ema.ema_model, va_ld, crit, device, len(va_ds))

        # Step scheduler using best of (va, ema_va)
        plateau_hit = sch.step_epoch(max(va, ema_va))
        if plateau_hit:
            hist["plateau_epochs"].append(ep)
            print(f"  [PLATEAU] Plateau detected at epoch {ep}! LR multiplied by {PLATEAU_FACTOR}")

        for k, v in [("tl", tl), ("ta", ta), ("vl", vl), ("va", va),
                     ("vp", vpr), ("vr", vrc), ("vf1", vf1), ("ema_va", ema_va)]:
            hist[k].append(v)

        print(f"Ep {ep:02d}/{TOTAL_EPOCHS} | "
              f"Train L={tl:.4f} Acc={ta*100:.1f}% | "
              f"Val Acc={va*100:.1f}% (EMA={ema_va*100:.1f}%) F1={vf1:.4f} | "
              f"LR t={_get_lr('trans'):.2e} h={_get_lr('head'):.2e}",
              flush=True)

        # Save Best Standard Model
        if va > best_std_acc:
            best_std_acc = va
            torch.save(model.state_dict(), best_p)
            print(f"  * New best standard checkpoint: {best_std_acc*100:.2f}%")

        # Save Best EMA Model
        if ema_va > best_ema_acc:
            best_ema_acc = ema_va
            torch.save(ema.ema_model.state_dict(), ema_p)
            print(f"  * New best EMA checkpoint:      {best_ema_acc*100:.2f}%")

    # --- Save Final Weights ---
    torch.save(model.state_dict(), final_p)

    # --- Final Evaluation & Reports ---
    print("\n" + "=" * 60)
    print(f"[SUMMARY] Best Standard Val Acc: {best_std_acc*100:.2f}%")
    print(f"[SUMMARY] Best EMA Val Acc:      {best_ema_acc*100:.2f}%")
    print("=" * 60)

    # Evaluate best standard
    model.load_state_dict(torch.load(best_p, weights_only=True))
    _, final_std_va, _, _, _, ft_b, fp_b = evaluate(model, va_ld, crit, device, len(va_ds))

    # Evaluate best EMA
    ema.ema_model.load_state_dict(torch.load(ema_p, weights_only=True))
    _, final_ema_va, _, _, _, ft_e, fp_e = evaluate(ema.ema_model, va_ld, crit, device, len(va_ds))

    # Save metadata
    with open(os.path.join(SAVE_DIR, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)
    with open(os.path.join(SAVE_DIR, "history.json"), "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)

    rep_b = classification_report(ft_b, fp_b, target_names=classes, zero_division=0)
    rep_e = classification_report(ft_e, fp_e, target_names=classes, zero_division=0)
    with open(os.path.join(SAVE_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Phase 6 — Best Standard Model (Val Acc: {final_std_va*100:.2f}%)\n{'='*60}\n{rep_b}\n\n")
        f.write(f"Phase 6 — Best EMA Model (Val Acc: {final_ema_va*100:.2f}%)\n{'='*60}\n{rep_e}\n")

    # Generate all plots
    gen_plots(hist, ft_b, fp_b, classes, ft_e, fp_e)

    print(f"\n[SAVED] Checkpoints -> {SAVE_DIR}")
    print(f"[SAVED] Diagnostic Plots -> {PLOTS_DIR}")


# =========================================================================
# 10. ENTRY POINT
# =========================================================================

if __name__ == "__main__":
    train_phase6()
