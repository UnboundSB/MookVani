"""Phase 2 Fine-tuning: Load pretrained backbone into ISLSentenceReconformer (BiGRU→Transformer→TCNN).

Strategy:
  - Backbone weights (input_dense, transformer) loaded from best_pretrained_model.pth.
  - BiGRU is new; frozen backbone unfreezes after FREEZE_EPOCHS (differential LR).
  - Curriculum fps: 1 fps → 25 fps linearly; center-out frame selection.
  - Classes with < MIN_SAMPLES files excluded from train/val (NOT deleted from disk).
  - Max MIN_SAMPLES per class loaded (balanced dataset, random subset each run).
  - Train/val split FIRST; augmentation only on train split.
  - Output: models/pretrained_phase_2_model/

Architecture notes:
  - ISLSentenceReconformer uses d_model=128 to match pretrained backbone.
  - input_dense: Linear(163→128) + GELU + Dropout  — maps to input_proj in pretrained.
  - rnn: BiGRU(128→64*2=128) — NEW, randomly initialised.
  - transformer: TransformerEncoder(d_model=128, nhead=4) — LOADED from pretrained.
  - cnn: Conv1d TCNN — LOADED from pretrained (except final head layer).
  - Weights mapped: input_proj.* → input_dense.*, transformer.* → transformer.*.
    cnn dim mismatch handled by strict=False (head re-init).
"""

import os
import sys
import glob
import json
import math
import random
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
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ── Paths ──────────────────────────────────────────────────────────────────────
BACKEND_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR        = os.path.join(BACKEND_DIR, "data", "tensors_include50_word_level")
PRETRAINED_PATH = os.path.join(BACKEND_DIR, "models", "pretrained_model", "best_pretrained_model.pth")
SAVE_DIR        = os.path.join(BACKEND_DIR, "models", "pretrained_phase_2_model")
PLOTS_DIR       = os.path.join(SAVE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from utils.dataloader_word import (  # noqa: E402
    discover_word_classes,
    load_split_tensors,
    split_word_classes,
)
from utils.pose_augmentation import (  # noqa: E402
    AugmentConfig,
    augment_pose_sequence as augment_pose_sequence_canonical,
)

# The augmentation set of this phase: mirror / tilt / bounded shift / scale.
PHASE2_AUGMENT = AugmentConfig(
    flip_prob=0.5,
    tilt_prob=0.6,
    tilt_deg=15.0,
    shift_prob=0.5,
    shift_dx=0.05,
    shift_dy=0.05,
    scale_prob=0.4,
    scale_lo=0.90,
    scale_hi=1.10,
    speed_prob=0.0,
    noise_std=0.007,
)

# ── Config ─────────────────────────────────────────────────────────────────────
MIN_SAMPLES    = 14       # a class below this tier is eliminated from the loader
MAX_SAMPLES    = 15       # hard budget of clips per class
SOURCE_FPS     = 25.0      # reference rate the corpus was measured at (NOT a cap)
START_FPS      = 1.0
END_FPS        = 50.0      # the curriculum may go ABOVE the source rate
MEDIAN_T       = 62       # median frames in tensors_include50 dataset
INPUT_DIM      = 163
D_MODEL        = 128      # must match pretrained (pretrainer uses d_model=128)
N_HEAD         = 4
N_TRANSFORMER  = 2
FREEZE_EPOCHS  = 5        # freeze backbone; train BiGRU+head only


# ==============================================================================
# 1. TEMPORAL SAMPLING
# ==============================================================================

def center_out_indices(T: int, n: int) -> list:
    """Center-out frame picker: middle → right → left alternately."""
    n = min(n, T)
    mid = T // 2
    candidates = [mid]
    r, l = mid + 1, mid - 1
    while len(candidates) < n:
        if r < T and len(candidates) < n:
            candidates.append(r); r += 1
        if l >= 0 and len(candidates) < n:
            candidates.append(l); l -= 1
    return sorted(candidates[:n])


def fps_for_epoch(epoch: int, total: int) -> float:
    progress = (epoch - 1) / max(total - 1, 1)
    return START_FPS + (END_FPS - START_FPS) * progress


def epoch_target_T(fps: float) -> int:
    """Number of frames a clip is resampled to at the given frame rate.

    ``SOURCE_FPS`` is only the REFERENCE rate the corpus was measured at - it
    is NOT a cap: the curriculum is allowed to go above it, and the clip is
    then upsampled to the denser target.
    """
    return max(1, round(MEDIAN_T * fps / SOURCE_FPS))


def sample_and_normalize(tensor: torch.Tensor, fps: float) -> torch.Tensor:
    """Center-out sample (when downsampling) then interpolate to the epoch length.

    The SOURCE frame rate is not a ceiling: above ``SOURCE_FPS`` every source
    frame is kept and the clip is interpolated UP to the denser target, so no
    frame of the original sign is ever thrown away.
    """
    T = tensor.shape[0]
    tgt = epoch_target_T(fps)
    n_pick = max(1, round(T * fps / SOURCE_FPS))
    if n_pick < T:
        idxs   = center_out_indices(T, n_pick)
        sampled = tensor[idxs]
    else:
        sampled = tensor                      # no cap: keep every source frame
    if sampled.shape[0] == tgt:
        return sampled
    s = sampled.T.unsqueeze(0)
    s = F.interpolate(s, size=tgt, mode='linear', align_corners=True)
    return s.squeeze(0).T


# ==============================================================================
# 2. AUGMENTATION
# ==============================================================================

def augment_pose_sequence(pose_seq: torch.Tensor, **kwargs) -> torch.Tensor:
    """Geometric augmentation - train split only.

    Delegates to the canonical implementation in
    ``Backend/utils/pose_augmentation.py``.  The old local copy rotated around
    the ORIGIN (the corpus is anchored at the image origin, so the neck sits at
    x ~ +3.5 shoulder widths and an origin-centred rotation swung the whole
    body out of frame) and it never swapped the mouth corners.
    """
    return augment_pose_sequence_canonical(pose_seq, PHASE2_AUGMENT)


# ==============================================================================
# 3. DATASETS
# ==============================================================================

class Phase2TrainDataset(Dataset):
    def __init__(self, class_samples: dict, init_fps: float = START_FPS):
        self.class_samples = class_samples
        self.current_fps   = init_fps
        self.items = [(cls, i)
                      for cls, tensors in class_samples.items()
                      for i in range(len(tensors))]

    def set_fps(self, fps: float):
        self.current_fps = fps

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        cls_idx, s_idx = self.items[idx]
        raw = self.class_samples[cls_idx][s_idx]
        sampled   = sample_and_normalize(raw, self.current_fps)
        augmented = augment_pose_sequence(sampled)
        return augmented, cls_idx


class Phase2ValDataset(Dataset):
    def __init__(self, class_samples: dict, init_fps: float = END_FPS):
        self.class_samples = class_samples
        self.current_fps   = init_fps
        self.items = [(cls, i)
                      for cls, tensors in class_samples.items()
                      for i in range(len(tensors))]

    def set_fps(self, fps: float):
        self.current_fps = fps

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        cls_idx, s_idx = self.items[idx]
        raw = self.class_samples[cls_idx][s_idx]
        return sample_and_normalize(raw, self.current_fps), cls_idx


# ==============================================================================
# 4. MODEL: ISLSentenceReconformer with d_model=128
# ==============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class Phase2Reconformer(nn.Module):
    """
    BiGRU → Transformer → TCNN for word-level pose classification.

    Weight loading:
      pretrained input_proj.0.* → self.input_dense.0.*   (Linear 163→128)
      pretrained transformer.*  → self.transformer.*      (exact match)
      pretrained tcnn.*         → self.cnn conv layers    (mapped manually)
      BiGRU and final head are always randomly initialised.
    """
    def __init__(self, num_classes: int, d_model: int = D_MODEL,
                 gru_layers: int = 2, n_head: int = N_HEAD,
                 n_transformer: int = N_TRANSFORMER, dropout: float = 0.15):
        super().__init__()
        self.d_model = d_model

        # 0. Input projection (mirrors pretrained input_proj)
        self.input_dense = nn.Sequential(
            nn.Linear(INPUT_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        # 1. BiGRU — NEW; not in pretrained
        self.bigru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0
        )
        self.gru_norm = nn.LayerNorm(d_model)

        # 2. Transformer — LOADED from pretrained
        self.pos_encoder = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_model * 2,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer      = nn.TransformerEncoder(enc_layer, num_layers=n_transformer,
                                                       enable_nested_tensor=False)
        self.transformer_norm = nn.LayerNorm(d_model)

        # 3. TCNN — conv dims LOADED from pretrained; final head re-init
        # Mirrors pretrainer tcnn_channels=(128, 256):
        #   tcnn.0: TemporalCNNBlock(128 → 128)   ← block1
        #   tcnn.1: TemporalCNNBlock(128 → 256)   ← block2
        self.tcnn_block1 = _TCNNBlock(d_model,   d_model,       dropout=dropout)  # 128→128
        self.tcnn_block2 = _TCNNBlock(d_model,   d_model * 2,   dropout=dropout)  # 128→256

        # Dual pool + classification head
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        pooled_dim    = d_model * 2 * 2  # 256*2 = 512

        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout * 2),
            nn.Linear(256, num_classes)
        )

    # ── Param groups ─────────────────────────────────────────────────
    def backbone_parameters(self):
        """Pretrained params: input_dense + transformer + tcnn."""
        return (list(self.input_dense.parameters()) +
                list(self.transformer.parameters()) +
                list(self.transformer_norm.parameters()) +
                list(self.tcnn_block1.parameters()) +
                list(self.tcnn_block2.parameters()))

    def new_parameters(self):
        """New params: BiGRU + positional encoding + head."""
        return (list(self.bigru.parameters()) +
                list(self.gru_norm.parameters()) +
                list(self.pos_encoder.parameters()) +
                list(self.classifier.parameters()))

    # ── Forward ──────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 163) → logits: (B, num_classes)"""
        B, T, _ = x.shape

        # 0. Project input
        h = self.input_dense(x)                         # (B, T, 128)

        # 1. BiGRU temporal context
        gru_out, _ = self.bigru(h)                      # (B, T, 128)
        h = self.gru_norm(h + gru_out)                  # residual

        # 2. Transformer global attention
        h = self.pos_encoder(h)
        h = self.transformer(h)
        h = self.transformer_norm(h)                    # (B, T, 128)

        # 3. TCNN
        h = h.transpose(1, 2)                           # (B, 128, T)
        h = self.tcnn_block1(h)                         # (B, 256, T)
        h = self.tcnn_block2(h)                         # (B, 256, T)

        # 4. Dual global pool + classify
        pooled = torch.cat([self.avg_pool(h).squeeze(-1),
                            self.max_pool(h).squeeze(-1)], dim=-1)  # (B, 512)
        return self.classifier(pooled)


class _TCNNBlock(nn.Module):
    """1D Temporal CNN Block with residual — mirrors TemporalCNNBlock in pretrainer."""
    def __init__(self, in_ch, out_ch, kernel_size=3, dropout=0.1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1    = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.bn1      = nn.BatchNorm1d(out_ch)
        self.act1     = nn.GELU()
        self.drop1    = nn.Dropout(dropout)
        self.conv2    = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.bn2      = nn.BatchNorm1d(out_ch)
        self.act2     = nn.GELU()
        self.drop2    = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        out = self.drop1(self.act1(self.bn1(self.conv1(x))))
        out = self.drop2(self.act2(self.bn2(self.conv2(out))))
        return out + res


# ==============================================================================
# 5. WEIGHT LOADING — map pretrained keys to Phase2Reconformer
# ==============================================================================

def build_model(num_classes: int, pretrained_path: str) -> Phase2Reconformer:
    """
    Build Phase2Reconformer and load matching pretrained weights.

    Key remapping (pretrainer → Phase2Reconformer):
      input_proj.0.*    → input_dense.0.*
      input_proj.1.*    → input_dense.1.*
      transformer.*     → transformer.*
      transformer_norm  → transformer_norm.*
      tcnn.0.*          → tcnn_block1.*
      tcnn.1.*          → tcnn_block2.*
      classifier.*      → DROPPED (different num_classes / different head)
      pos_embedding     → DROPPED (replaced by PositionalEncoding)
    """
    model = Phase2Reconformer(num_classes=num_classes)

    if not os.path.exists(pretrained_path):
        print(f"[WARN] {pretrained_path} not found — training from scratch.")
        return model

    raw_sd = torch.load(pretrained_path, map_location='cpu', weights_only=True)

    remap = {}
    DROP  = {'pos_embedding',
             'classifier.0.weight', 'classifier.0.bias',
             'classifier.1.weight', 'classifier.1.bias',
             'classifier.4.weight', 'classifier.4.bias'}

    for k, v in raw_sd.items():
        if k in DROP:
            continue
        # input_proj → input_dense
        if k.startswith('input_proj.'):
            new_k = k.replace('input_proj.', 'input_dense.')
            remap[new_k] = v
        # tcnn.0.* → tcnn_block1.*
        elif k.startswith('tcnn.0.'):
            new_k = k.replace('tcnn.0.', 'tcnn_block1.')
            remap[new_k] = v
        # tcnn.1.* → tcnn_block2.*
        elif k.startswith('tcnn.1.'):
            new_k = k.replace('tcnn.1.', 'tcnn_block2.')
            remap[new_k] = v
        else:
            remap[k] = v

    missing, unexpected = model.load_state_dict(remap, strict=False)
    loaded = len(remap) - len(missing)
    print(f"[INFO] Pretrained weights loaded: {loaded} keys matched.")
    if missing:
        print(f"  Missing  (new/random): {missing}")
    if unexpected:
        print(f"  Unexpected (ignored): {unexpected}")
    return model


# ==============================================================================
# 6. DATA LOADING + STRATIFIED SPLIT
# ==============================================================================

def load_and_split(data_dir: str, min_samples: int, val_frac: float = 0.2):
    """Load the word corpus with the shared selection rule.

    A class that owns fewer than ``min_samples`` clips is ELIMINATED from the
    loader (the files stay on disk), and every surviving class is capped at
    ``max_samples`` clips.  The class list is ordered by descending clip count
    (the "top most frequent" classes).
    """
    print(f"[INFO] Scanning {data_dir} ...")
    infos = discover_word_classes(
        data_dir, min_samples_per_class=min_samples, max_samples_per_class=MAX_SAMPLES
    )
    classes = [i.name for i in infos]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    print(f"[INFO] {num_classes} classes loaded (each capped at {MAX_SAMPLES} samples).")

    train_items, val_items = split_word_classes(infos, val_fraction=val_frac, seed=42)
    train_data = load_split_tensors(train_items)
    val_data = load_split_tensors(val_items)

    total_train = sum(len(v) for v in train_data.values())
    total_val = sum(len(v) for v in val_data.values())
    print(f"[INFO] Split: {total_train} train | {total_val} val | {num_classes} classes")
    return train_data, val_data, classes


# ==============================================================================
# 7. PLOTS
# ==============================================================================

def generate_plots(history: dict, targets: list, preds: list, classes: list):
    print("[INFO] Generating evaluation plots ...")
    ep = range(1, len(history['train_loss']) + 1)

    # ── Loss & Acc ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0f1117')
    for ax in (ax1, ax2):
        ax.set_facecolor('#1a1d27')
        ax.tick_params(colors='#c9d1d9')
        for sp in ax.spines.values(): sp.set_edgecolor('#30363d')

    ax1.plot(ep, history['train_loss'], color='#f85149', lw=2, label='Train Loss')
    ax1.plot(ep, history['val_loss'],   color='#ff9500', lw=2, linestyle='--', label='Val Loss')
    ax1.set_title('Loss (curriculum fps)', color='#e6edf3', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch', color='#8b949e'); ax1.set_ylabel('CE Loss', color='#8b949e')
    ax1.legend(facecolor='#21262d', labelcolor='#c9d1d9', edgecolor='#30363d', fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.2, color='#8b949e')

    ax2.plot(ep, [a*100 for a in history['train_acc']], color='#58a6ff', lw=2, label='Train Acc %')
    ax2.plot(ep, [a*100 for a in history['val_acc']],   color='#3fb950', lw=2, linestyle='--', label='Val Acc %')
    ax2.set_title('Accuracy (curriculum fps)', color='#e6edf3', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch', color='#8b949e'); ax2.set_ylabel('Accuracy (%)', color='#8b949e')
    ax2.legend(facecolor='#21262d', labelcolor='#c9d1d9', edgecolor='#30363d', fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.2, color='#8b949e')

    plt.suptitle('Phase 2 — BiGRU→Transformer→TCNN', color='#e6edf3', fontsize=13, fontweight='bold')
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, 'acc_vs_loss.png')
    plt.savefig(p, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor()); plt.close()
    print(f"  [SAVED] {p}")

    # ── F1/P/R ──
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0f1117'); ax.set_facecolor('#1a1d27')
    ax.tick_params(colors='#c9d1d9')
    for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
    ax.plot(ep, history['val_f1'],        color='#bc8cff', lw=2.5, label='Weighted F1')
    ax.plot(ep, history['val_precision'], color='#58a6ff', lw=2, linestyle=':', label='Precision')
    ax.plot(ep, history['val_recall'],    color='#3fb950', lw=2, linestyle='--', label='Recall')
    ax.set_title('Validation Metrics — Phase 2', color='#e6edf3', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch', color='#8b949e'); ax.set_ylabel('Score', color='#8b949e')
    ax.legend(facecolor='#21262d', labelcolor='#c9d1d9', edgecolor='#30363d', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.2, color='#8b949e')
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, 'metrics.png')
    plt.savefig(p, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor()); plt.close()
    print(f"  [SAVED] {p}")

    # ── FPS schedule ──
    if history.get('fps_per_epoch'):
        fps_list = history['fps_per_epoch']
        fig, ax = plt.subplots(figsize=(10, 3))
        fig.patch.set_facecolor('#0f1117'); ax.set_facecolor('#1a1d27')
        ax.tick_params(colors='#c9d1d9')
        for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
        ax.plot(range(1, len(fps_list)+1), fps_list, color='#f0c040', lw=2.5)
        ax.fill_between(range(1, len(fps_list)+1), fps_list, alpha=0.2, color='#f0c040')
        ax.set_title('Curriculum FPS Schedule', color='#e6edf3', fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch', color='#8b949e'); ax.set_ylabel('FPS', color='#8b949e')
        ax.grid(True, linestyle='--', alpha=0.2, color='#8b949e')
        plt.tight_layout()
        p = os.path.join(PLOTS_DIR, 'fps_schedule.png')
        plt.savefig(p, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor()); plt.close()
        print(f"  [SAVED] {p}")

    # ── Confusion matrix ──
    cm = confusion_matrix(targets, preds)
    n  = len(classes)
    sz = max(12, n * 0.07)
    fig, ax = plt.subplots(figsize=(sz, sz * 0.9))
    fig.patch.set_facecolor('#0f1117'); ax.set_facecolor('#1a1d27')
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title(f'Confusion Matrix — Phase 2 ({n} classes, full 25fps)',
                 color='#e6edf3', fontsize=10, fontweight='bold')
    ax.set_xlabel('Predicted', color='#8b949e'); ax.set_ylabel('True', color='#8b949e')
    fs = max(3.5, 6.5 - n * 0.012)
    ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n))
    ax.set_xticklabels(classes, rotation=90, fontsize=fs, color='#c9d1d9')
    ax.set_yticklabels(classes, fontsize=fs, color='#c9d1d9')
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, 'confusion_matrix.png')
    plt.savefig(p, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor()); plt.close()
    print(f"  [SAVED] {p}")


# ==============================================================================
# 8. TRAINING LOOP
# ==============================================================================

def train_phase2(epochs=40, batch_size=64, head_lr=5e-4, backbone_lr=5e-5,
                 freeze_epochs=FREEZE_EPOCHS, val_frac=0.20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    train_data, val_data, classes = load_and_split(DATA_DIR, MIN_SAMPLES, val_frac)
    num_classes = len(classes)

    train_ds = Phase2TrainDataset(train_data, init_fps=START_FPS)
    val_ds   = Phase2ValDataset(val_data,   init_fps=START_FPS)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # Build model with pretrained backbone
    model = build_model(num_classes, PRETRAINED_PATH).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Phase2Reconformer: {total:,} params | {num_classes} classes")

    # Freeze backbone for warmup
    for p in model.backbone_parameters():
        p.requires_grad = False
    optimizer = optim.AdamW(model.new_parameters(), lr=head_lr, weight_decay=1e-4)
    print(f"[INFO] Epochs 1-{freeze_epochs}: backbone FROZEN — training BiGRU+head only")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.06)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = {k: [] for k in ['train_loss','train_acc','val_loss','val_acc',
                                'val_precision','val_recall','val_f1','fps_per_epoch']}
    best_val_acc = 0.0
    best_path  = os.path.join(SAVE_DIR, 'best_phase2_model.pth')
    final_path = os.path.join(SAVE_DIR, 'final_phase2_model.pth')

    print(f"\n[START] Phase 2 — {epochs} epochs, 1fps→25fps curriculum\n")

    for epoch in range(1, epochs + 1):
        # Unfreeze backbone after warmup
        if epoch == freeze_epochs + 1:
            print(f"\n[UNFREEZE] Epoch {epoch}: releasing backbone @ backbone_lr={backbone_lr}")
            for p in model.backbone_parameters():
                p.requires_grad = True
            optimizer = optim.AdamW([
                {'params': model.backbone_parameters(), 'lr': backbone_lr},
                {'params': model.new_parameters(),      'lr': head_lr * 0.5},
            ], weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - freeze_epochs, eta_min=1e-6)

        # Curriculum fps
        cur_fps = fps_for_epoch(epoch, epochs)
        cur_T   = epoch_target_T(cur_fps)
        train_ds.set_fps(cur_fps)
        val_ds.set_fps(cur_fps)
        history['fps_per_epoch'].append(round(cur_fps, 2))

        # ── Train ──
        model.train()
        total_loss, correct, total_n = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"E{epoch:02d} [{cur_fps:.1f}fps T={cur_T}]",
                    leave=False, ncols=88)
        for poses, labels in pbar:
            poses, labels = poses.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(poses)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += loss.item() * poses.size(0)
            correct    += (logits.argmax(dim=-1) == labels).sum().item()
            total_n    += poses.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        scheduler.step()
        train_loss = total_loss / total_n
        train_acc  = correct / total_n

        # ── Val ──
        model.eval()
        vl_sum = 0.0
        vp, vt = [], []
        with torch.no_grad():
            for poses, labels in val_loader:
                poses, labels = poses.to(device), labels.to(device)
                logits = model(poses)
                vl_sum += criterion(logits, labels).item() * poses.size(0)
                vp.extend(logits.argmax(dim=-1).cpu().numpy())
                vt.extend(labels.cpu().numpy())

        val_loss = vl_sum / len(val_ds)
        val_acc  = accuracy_score(vt, vp)
        vprec, vrec, vf1, _ = precision_recall_fscore_support(vt, vp, average='weighted', zero_division=0)

        for k, v in [('train_loss', train_loss), ('train_acc', train_acc),
                     ('val_loss', val_loss), ('val_acc', val_acc),
                     ('val_precision', vprec), ('val_recall', vrec), ('val_f1', vf1)]:
            history[k].append(v)

        print(f"Ep {epoch:02d}/{epochs} | fps={cur_fps:.1f} T={cur_T:3d} | "
              f"Train L={train_loss:.4f} Acc={train_acc*100:.1f}% | "
              f"Val L={val_loss:.4f} Acc={val_acc*100:.1f}% F1={vf1:.4f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

    torch.save(model.state_dict(), final_path)
    print(f"\n[DONE] Best Val Acc: {best_val_acc*100:.2f}%")

    with open(os.path.join(SAVE_DIR, 'classes.json'), 'w', encoding='utf-8') as f:
        json.dump(classes, f, indent=2)
    with open(os.path.join(SAVE_DIR, 'history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    # Final full-fps evaluation
    print("\n[EVAL] Final evaluation @ full 25fps ...")
    model.load_state_dict(torch.load(best_path, weights_only=True))
    model.eval()
    val_ds.set_fps(END_FPS)
    fp, ft = [], []
    with torch.no_grad():
        for poses, labels in val_loader:
            poses, labels = poses.to(device), labels.to(device)
            fp.extend(model(poses).argmax(dim=-1).cpu().numpy())
            ft.extend(labels.cpu().numpy())

    full_acc = accuracy_score(ft, fp)
    print(f"[EVAL] Full-fps Val Acc: {full_acc*100:.2f}%")
    report = classification_report(ft, fp, target_names=classes, zero_division=0)
    with open(os.path.join(SAVE_DIR, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Full 25-fps Evaluation\n{'='*60}\n{report}")

    generate_plots(history, ft, fp, classes)
    print(f"\n[SAVED] Model → {SAVE_DIR}")
    print(f"[SAVED] Plots  → {PLOTS_DIR}")


# ==============================================================================
# 9. ENTRY POINT
# ==============================================================================

if __name__ == '__main__':
    train_phase2(
        epochs        = 40,
        batch_size    = 64,
        head_lr       = 5e-4,
        backbone_lr   = 5e-5,
        freeze_epochs = FREEZE_EPOCHS,
        val_frac      = 0.20,
    )
