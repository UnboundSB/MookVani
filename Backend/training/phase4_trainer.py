"""Phase 4 - Full-FPS fine-tuning from Phase-3 checkpoint.

Key design decisions
--------------------
* No frame clipping / curriculum FPS.  Tensors are used AT NATIVE length.
* Augmentation applied ONLY to the train split (after train/val split).
  - Random time-stretch  : squeeze or stretch frames by 0.75-1.25x
  - Random flip          : mirror x-coordinates, swap left/right landmarks
  - Random tilt (rotate) : 2-D rotation of all joint coords, uniform angle
* 50-epoch schedule:
    Epochs  1-30  Transformer FROZEN  — RNN (BiGRU) + CNN (TCNN) + head train
    Epochs 31-50  Full model unfreeze — all params train with lower backbone LR
* Separate AdamW param-groups for (transformer) vs (rnn+cnn+head).
* CosineAnnealingWarmRestarts scheduler (T_0=15) for the first phase and
  CosineAnnealingLR(T_max=20) for the second phase.
* Label-smoothed CrossEntropyLoss throughout.
* Per-epoch plots: loss, accuracy, F1/P/R, LR schedule, confusion matrix.
* Saves to: models/phase4_word_model/
"""

import os, sys, glob, json, math, random
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
BACKEND_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(BACKEND_DIR, "data", "tensors_include50_word_level")
PHASE3_CKPT  = os.path.join(BACKEND_DIR, "models", "phase3_word_model",
                             "best_phase3_model.pth")
PHASE3_CLASSES = os.path.join(BACKEND_DIR, "models", "phase3_word_model", "classes.json")
SAVE_DIR     = os.path.join(BACKEND_DIR, "models", "phase4_word_model")
PLOTS_DIR    = os.path.join(SAVE_DIR, "plots")
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
    mirror_pose,
    tilt_pose as canonical_tilt,
)
from utils.checkpoint_utils import load_flexible  # noqa: E402

# =========================================================================
# HYPER-PARAMETERS
# =========================================================================
MIN_SAMPLES       = 14          # a class below this tier is eliminated
MAX_SAMPLES       = 15          # hard budget of clips per class
INPUT_DIM        = 163
D_MODEL          = 128
N_HEAD           = 4
N_TRANS          = 2
GRU_LAYERS       = 2

TOTAL_EPOCHS     = 50
FREEZE_EPOCHS    = 30          # transformer frozen for first 30
BATCH_SIZE       = 64
VAL_FRAC         = 0.20

# -- Optimizer LRs ---
LR_RNN_CNN       = 3e-4        # BiGRU + TCNN during frozen phase
LR_HEAD_FROZEN   = 5e-4        # head during frozen phase
LR_TRANS_UNFREEZE = 5e-5       # transformer after unfreeze
LR_RNN_CNN_UNFREEZE = 1e-4     # BiGRU + TCNN after unfreeze
LR_HEAD_UNFREEZE = 2e-4        # head after unfreeze

WEIGHT_DECAY     = 1e-4
LABEL_SMOOTHING  = 0.05
GRAD_CLIP        = 2.0

# -- Augmentation params ---
STRETCH_RANGE    = (0.75, 1.25)   # time-stretch factor
FLIP_PROB        = 0.5
TILT_RANGE       = (-20.0, 20.0)  # degrees, uniform
TILT_MAX         = max(abs(TILT_RANGE[0]), abs(TILT_RANGE[1]))

# The canonical augmentation set (Backend/utils/pose_augmentation.py).
PHASE4_AUGMENT = AugmentConfig(
    flip_prob=FLIP_PROB,
    tilt_prob=1.0,
    tilt_deg=TILT_MAX,
    shift_prob=0.4,
    shift_dx=0.05,
    shift_dy=0.02,
    speed_prob=0.5,
    speed_lo=STRETCH_RANGE[0],
    speed_hi=STRETCH_RANGE[1],
    time_shift_prob=0.2,
    time_shift_max=3,
    mask_prob=0.0,
    noise_std=0.0,
)


# =========================================================================
# 1. TIME-STRETCH AUGMENTATION
#    Randomly speeds up or slows down by resampling the temporal axis.
# =========================================================================

def time_stretch(seq: torch.Tensor) -> torch.Tensor:
    """Randomly stretch or compress the temporal dimension."""
    T, D = seq.shape
    factor = random.uniform(*STRETCH_RANGE)
    new_T  = max(1, round(T * factor))
    if new_T == T:
        return seq
    # F.interpolate expects (B, C, L)
    s = seq.T.unsqueeze(0)                               # (1, D, T)
    s = F.interpolate(s, size=new_T, mode="linear", align_corners=True)
    return s.squeeze(0).T                                # (new_T, D)


# =========================================================================
# 2. FLIP + TILT AUGMENTATION
# =========================================================================

def flip_pose(seq: torch.Tensor) -> torch.Tensor:
    """Horizontal mirror around the signer's own axis (canonical version)."""
    return mirror_pose(seq)


def tilt_pose(seq: torch.Tensor) -> torch.Tensor:
    """Rotation around the neck (canonical version)."""
    return canonical_tilt(seq, TILT_MAX)


def augment(seq: torch.Tensor) -> torch.Tensor:
    """Compose: time-stretch -> optional mirror -> tilt -> bounded shift.

    The canonical implementation in ``Backend/utils/pose_augmentation.py`` is
    used, so the mouth corners are swapped as well (the old local copy left
    the mouth asymmetric) and the rotation pivots on the neck instead of the
    origin.
    """
    return augment_pose_sequence_canonical(seq, PHASE4_AUGMENT)


# =========================================================================
# 3. COLLATE  (variable-length sequences -> padded batch)
# =========================================================================

def collate_pad(batch):
    """Pad variable-length sequences to the max T in the batch."""
    seqs, labels = zip(*batch)
    T_max = max(s.shape[0] for s in seqs)
    D     = seqs[0].shape[1]
    padded = torch.zeros(len(seqs), T_max, D)
    for i, s in enumerate(seqs):
        padded[i, :s.shape[0], :] = s
    return padded, torch.tensor(labels, dtype=torch.long)


# =========================================================================
# 4. DATASETS
# =========================================================================

class TrainDS(Dataset):
    def __init__(self, data: dict):
        self.data  = data
        self.items = [(c, i) for c, ts in data.items() for i in range(len(ts))]

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        c, i = self.items[idx]
        seq  = self.data[c][i].clone()
        seq  = augment(seq)           # augmentation applied here (train only)
        return seq, c


class ValDS(Dataset):
    def __init__(self, data: dict):
        self.data  = data
        self.items = [(c, i) for c, ts in data.items() for i in range(len(ts))]

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        c, i = self.items[idx]
        return self.data[c][i].clone(), c   # NO augmentation on val


# =========================================================================
# 5. MODEL  (identical to Phase3Model — same architecture)
# =========================================================================

class PosEnc(nn.Module):
    def __init__(self, d, maxlen=512):
        super().__init__()
        pe  = torch.zeros(maxlen, d)
        pos = torch.arange(0, maxlen).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.)/d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x): return x + self.pe[:, :x.size(1), :]


class TCNN(nn.Module):
    def __init__(self, ic, oc, k=3, drop=.1):
        super().__init__(); p = k // 2
        self.c1 = nn.Conv1d(ic, oc, k, padding=p); self.b1 = nn.BatchNorm1d(oc)
        self.g1 = nn.GELU();                        self.d1 = nn.Dropout(drop)
        self.c2 = nn.Conv1d(oc, oc, k, padding=p); self.b2 = nn.BatchNorm1d(oc)
        self.g2 = nn.GELU();                        self.d2 = nn.Dropout(drop)
        self.res = nn.Conv1d(ic, oc, 1) if ic != oc else nn.Identity()

    def forward(self, x):
        r = self.res(x)
        o = self.d1(self.g1(self.b1(self.c1(x))))
        o = self.d2(self.g2(self.b2(self.c2(o))))
        return o + r


class Phase4Model(nn.Module):
    """
    Identical architecture to Phase3Model.

    Param-group naming for Phase 4 freeze/unfreeze:
        transformer_params()   — self.pe, self.trans, self.tnorm
        rnn_params()           — self.proj, self.gru, self.gnorm
        cnn_params()           — self.t1, self.t2
        head_params()          — self.head
    """
    def __init__(self, nc, d=D_MODEL, gru=GRU_LAYERS,
                 nh=N_HEAD, nt=N_TRANS, drop=0.15):
        super().__init__()
        self.proj  = nn.Sequential(
            nn.Linear(INPUT_DIM, d), nn.LayerNorm(d), nn.Dropout(drop))
        self.gru   = nn.GRU(d, d//2, gru, batch_first=True, bidirectional=True,
                            dropout=drop if gru > 1 else 0.)
        self.gnorm = nn.LayerNorm(d)
        self.pe    = PosEnc(d)
        el = nn.TransformerEncoderLayer(d, nh, d*2, drop, "gelu", batch_first=True)
        self.trans = nn.TransformerEncoder(el, nt, enable_nested_tensor=False)
        self.tnorm = nn.LayerNorm(d)
        self.t1    = TCNN(d, d, drop=drop)
        self.t2    = TCNN(d, d*2, drop=drop)
        self.ap    = nn.AdaptiveAvgPool1d(1)
        self.mp    = nn.AdaptiveMaxPool1d(1)
        self.head  = nn.Sequential(
            nn.Linear(d*4, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Dropout(drop*2), nn.Linear(256, nc))

    def transformer_params(self):
        return (list(self.pe.parameters()) +
                list(self.trans.parameters()) +
                list(self.tnorm.parameters()))

    def rnn_params(self):
        return (list(self.proj.parameters()) +
                list(self.gru.parameters()) +
                list(self.gnorm.parameters()))

    def cnn_params(self):
        return list(self.t1.parameters()) + list(self.t2.parameters())

    def head_params(self):
        return list(self.head.parameters())

    def forward(self, x):
        h = self.proj(x)
        g, _ = self.gru(h);  h = self.gnorm(h + g)
        h = self.pe(h);      h = self.trans(h); h = self.tnorm(h)
        h = h.transpose(1, 2)
        h = self.t1(h); h = self.t2(h)
        p = torch.cat([self.ap(h).squeeze(-1), self.mp(h).squeeze(-1)], -1)
        return self.head(p)


# =========================================================================
# 6. WEIGHT LOADING from Phase-3
# =========================================================================

def build_model(nc: int, ckpt: str) -> Phase4Model:
    model = Phase4Model(nc)
    if not os.path.exists(ckpt):
        print(f"[WARN] Phase-3 checkpoint not found: {ckpt} -- random init")
        return model
    raw = torch.load(ckpt, map_location="cpu", weights_only=True)
    # Phase-3 uses identical key names (proj/gru/gnorm/pe/trans/tnorm/t1/t2/head)
    miss, unex = load_flexible(model, raw, label="PHASE3->PHASE4")
    loaded = len(raw) - len(miss)
    print(f"[INFO] Phase-3 weights loaded: {loaded}/{len(raw)} keys matched.")
    if miss:  print(f"  Missing (new):    {miss}")
    if unex:  print(f"  Unexpected (ign): {unex}")
    return model


# =========================================================================
# 7. DATA LOADING + SPLIT
# =========================================================================

def load_data(data_dir: str, phase3_classes_json: str,
              max_per: int = MAX_SAMPLES, vfrac: float = VAL_FRAC):
    """
    Load the word corpus with the shared selection rule.

    A class that owns fewer than MIN_SAMPLES clips is eliminated from the
    loader (the files stay on disk) and every surviving class is capped at
    MAX_SAMPLES clips.  The class ORDER comes from the shared loader, so it is
    identical to the order Phase 3 produced (both use the same rule) and the
    head of the warm-started checkpoint keeps its meaning.
    """
    print(f"[INFO] Scanning {data_dir} ...")

    infos = discover_word_classes(data_dir, min_samples_per_class=MIN_SAMPLES,
                                  max_samples_per_class=max_per)
    classes = [i.name for i in infos]

    # The previous phase wrote its class list; it is only used to CHECK that
    # the vocabulary did not change under our feet.
    if os.path.exists(phase3_classes_json):
        with open(phase3_classes_json, encoding="utf-8") as f:
            payload = json.load(f)
        prev_classes = payload.get("classes", []) if isinstance(payload, dict) else payload
        if list(prev_classes) != classes:
            print(f"[WARN] The previous phase trained on {len(prev_classes)} classes "
                  f"but the loader selects {len(classes)}.")
            print("       The head of the warm-started checkpoint will be re-learned "
                  "(its shape changed).")
    else:
        print("[INFO] No previous-phase classes.json found; using the top-most "
              "frequent classes of the corpus.")

    print(f"[INFO] {len(classes)} classes loaded (capped at {max_per} each).")

    train_items, val_items = split_word_classes(infos, val_fraction=vfrac, seed=42)
    tr = load_split_tensors(train_items)
    va = load_split_tensors(val_items)
    tot_tr = sum(len(v) for v in tr.values())
    tot_va = sum(len(v) for v in va.values())
    print(f"[INFO] Split: {tot_tr} train | {tot_va} val")
    return tr, va, classes


# =========================================================================
# 8. PLOTS
# =========================================================================
BG  = "#0f1117"; PAN = "#1a1d27"; TXT = "#e6edf3"; MUT = "#8b949e"; BRD = "#30363d"
LEG = dict(facecolor="#21262d", labelcolor="#c9d1d9", edgecolor="#30363d", fontsize=9)


def sax(ax):
    ax.set_facecolor(PAN); ax.tick_params(colors=MUT)
    for s in ax.spines.values(): s.set_edgecolor(BRD)
    ax.grid(True, ls="--", alpha=.2, color=MUT)


def plot_loss_acc(h: dict):
    ep = range(1, len(h["tl"]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5)); fig.patch.set_facecolor(BG)
    sax(a1); sax(a2)
    a1.plot(ep, h["tl"], color="#f85149", lw=2, label="Train Loss")
    a1.plot(ep, h["vl"], color="#ff9500", lw=2, ls="--", label="Val Loss")
    a1.axvline(FREEZE_EPOCHS + 0.5, color="#58a6ff", ls=":", lw=1.5,
               label=f"Unfreeze (Ep {FREEZE_EPOCHS+1})")
    a1.set_title("Cross-Entropy Loss (label smooth=0.05)", color=TXT,
                 fontsize=12, fontweight="bold")
    a1.set_xlabel("Epoch", color=MUT); a1.set_ylabel("Loss", color=MUT)
    a1.legend(**LEG)
    a2.plot(ep, [a*100 for a in h["ta"]], color="#58a6ff", lw=2, label="Train Acc%")
    a2.plot(ep, [a*100 for a in h["va"]], color="#3fb950", lw=2, ls="--", label="Val Acc%")
    a2.axvline(FREEZE_EPOCHS + 0.5, color="#f0c040", ls=":", lw=1.5,
               label=f"Unfreeze (Ep {FREEZE_EPOCHS+1})")
    a2.set_title("Accuracy (full FPS, no clipping)", color=TXT,
                 fontsize=12, fontweight="bold")
    a2.set_xlabel("Epoch", color=MUT); a2.set_ylabel("Acc %", color=MUT)
    a2.legend(**LEG)
    plt.suptitle("Phase 4 — Full-FPS Fine-tuning", color=TXT, fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "loss_acc.png")
    plt.savefig(p, dpi=200, bbox_inches="tight", facecolor=BG); plt.close()
    print(f"  [SAVED] {p}")


def plot_metrics(h: dict):
    ep = range(1, len(h["vf1"]) + 1)
    fig, ax = plt.subplots(figsize=(12, 5)); fig.patch.set_facecolor(BG); sax(ax)
    ax.plot(ep, h["vf1"], color="#bc8cff", lw=2.5, label="Weighted F1")
    ax.plot(ep, h["vp"],  color="#58a6ff", lw=2,   ls=":",  label="Weighted Precision")
    ax.plot(ep, h["vr"],  color="#3fb950", lw=2,   ls="--", label="Weighted Recall")
    ax.axvline(FREEZE_EPOCHS + 0.5, color="#f0c040", ls=":", lw=1.5,
               label=f"Unfreeze (Ep {FREEZE_EPOCHS+1})")
    ax.set_title("Validation Metrics — Phase 4", color=TXT, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", color=MUT); ax.set_ylabel("Score (0-1)", color=MUT)
    ax.set_ylim(0, 1); ax.legend(**LEG); plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "metrics.png")
    plt.savefig(p, dpi=200, bbox_inches="tight", facecolor=BG); plt.close()
    print(f"  [SAVED] {p}")


def plot_lr(h: dict):
    ep = range(1, len(h["lr_rnn"]) + 1)
    fig, ax = plt.subplots(figsize=(12, 3)); fig.patch.set_facecolor(BG); sax(ax)
    ax.plot(ep, h["lr_rnn"],   color="#58a6ff", lw=2,   label="RNN+CNN LR")
    ax.plot(ep, h["lr_head"],  color="#3fb950", lw=2,   ls="--", label="Head LR")
    ax.plot(ep, h["lr_trans"], color="#f85149", lw=2,   ls=":", label="Transformer LR")
    ax.axvline(FREEZE_EPOCHS + 0.5, color="#f0c040", ls=":", lw=1.5,
               label=f"Unfreeze (Ep {FREEZE_EPOCHS+1})")
    ax.set_title("Learning Rate Schedule (CosineAnnealingWarmRestarts / CosineAnnealing)",
                 color=TXT, fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", color=MUT); ax.set_ylabel("LR", color=MUT)
    ax.set_yscale("log"); ax.legend(**LEG); plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "lr_schedule.png")
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG); plt.close()
    print(f"  [SAVED] {p}")


def plot_cm(yt, yp, classes):
    cm = confusion_matrix(yt, yp); n = len(classes); sz = max(12, n * 0.07)
    fig, ax = plt.subplots(figsize=(sz, sz * 0.9)); fig.patch.set_facecolor(BG)
    ax.set_facecolor(PAN)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title(f"Confusion Matrix — Phase 4 ({n} classes, full FPS)",
                 color=TXT, fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted", color=MUT); ax.set_ylabel("True", color=MUT)
    fs = max(3.5, 6.5 - n * 0.012)
    ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n))
    ax.set_xticklabels(classes, rotation=90, fontsize=fs, color="#c9d1d9")
    ax.set_yticklabels(classes, fontsize=fs, color="#c9d1d9")
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG); plt.close()
    print(f"  [SAVED] {p}")


def gen_plots(h, yt, yp, classes):
    print("[INFO] Generating plots ...")
    plot_loss_acc(h); plot_metrics(h); plot_lr(h); plot_cm(yt, yp, classes)


# =========================================================================
# 9. TRAINING LOOP
# =========================================================================

def train_phase4():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    # --- Data ---
    tr_data, va_data, classes = load_data(DATA_DIR, PHASE3_CLASSES)
    nc = len(classes)
    if nc == 0: raise RuntimeError(f"[ERROR] No classes loaded from {DATA_DIR}")

    tr_ds = TrainDS(tr_data); va_ds = ValDS(va_data)
    if len(tr_ds) == 0:
        raise RuntimeError("[ERROR] Training set is empty after split.")

    tr_ld = DataLoader(tr_ds, BATCH_SIZE, shuffle=True, drop_last=True,
                       num_workers=0, collate_fn=collate_pad)
    va_ld = DataLoader(va_ds, BATCH_SIZE, shuffle=False,
                       num_workers=0, collate_fn=collate_pad)

    # --- Model ---
    model = build_model(nc, PHASE3_CKPT).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Phase4Model: {total_p:,} params | {nc} classes")

    # =========================================================
    # PHASE A  (Epochs 1-30): Transformer FROZEN
    #   Optimise: RNN + CNN + head
    #   Scheduler: CosineAnnealingWarmRestarts(T_0=15, T_mult=1)
    # =========================================================
    for p in model.transformer_params():
        p.requires_grad = False

    crit   = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    opt_a  = optim.AdamW([
        {"params": model.rnn_params(),  "lr": LR_RNN_CNN,   "name": "rnn_cnn"},
        {"params": model.cnn_params(),  "lr": LR_RNN_CNN,   "name": "cnn"},
        {"params": model.head_params(), "lr": LR_HEAD_FROZEN,"name": "head"},
    ], weight_decay=WEIGHT_DECAY)
    sch_a  = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt_a, T_0=15, T_mult=1, eta_min=1e-6)

    print(f"[INFO] Epochs 1-{FREEZE_EPOCHS}: transformer FROZEN — "
          f"RNN+CNN lr={LR_RNN_CNN}, head lr={LR_HEAD_FROZEN}")

    hist = {k: [] for k in ["tl","ta","vl","va","vp","vr","vf1",
                             "lr_rnn","lr_head","lr_trans"]}
    best_acc = 0.
    best_p   = os.path.join(SAVE_DIR, "best_phase4_model.pth")
    final_p  = os.path.join(SAVE_DIR, "final_phase4_model.pth")

    print(f"\n[START] Phase 4 -- {TOTAL_EPOCHS} epochs, full FPS (no clipping)\n")

    def _get_lr(optimizer, name):
        for pg in optimizer.param_groups:
            if pg.get("name") == name:
                return pg["lr"]
        return optimizer.param_groups[0]["lr"]

    for ep in range(1, TOTAL_EPOCHS + 1):

        # -----------------------------------------------
        # Switch to Phase B at epoch FREEZE_EPOCHS+1
        # -----------------------------------------------
        if ep == FREEZE_EPOCHS + 1:
            print(f"\n[UNFREEZE] Epoch {ep}: releasing transformer")
            for p in model.transformer_params():
                p.requires_grad = True
            opt   = optim.AdamW([
                {"params": model.transformer_params(), "lr": LR_TRANS_UNFREEZE,    "name": "trans"},
                {"params": model.rnn_params(),         "lr": LR_RNN_CNN_UNFREEZE,  "name": "rnn_cnn"},
                {"params": model.cnn_params(),         "lr": LR_RNN_CNN_UNFREEZE,  "name": "cnn"},
                {"params": model.head_params(),        "lr": LR_HEAD_UNFREEZE,     "name": "head"},
            ], weight_decay=WEIGHT_DECAY)
            sch   = optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=TOTAL_EPOCHS - FREEZE_EPOCHS, eta_min=1e-6)
        else:
            opt = opt_a; sch = sch_a

        # Record LRs (before step for logging)
        rnn_lr   = _get_lr(opt, "rnn_cnn")
        head_lr  = _get_lr(opt, "head")
        trans_lr = _get_lr(opt, "trans") if ep > FREEZE_EPOCHS else 0.0
        hist["lr_rnn"].append(rnn_lr)
        hist["lr_head"].append(head_lr)
        hist["lr_trans"].append(trans_lr)

        # --- Train ---
        model.train(); tl = tc = tn = 0
        bar = tqdm(tr_ld,
                   desc=f"E{ep:02d}/{TOTAL_EPOCHS} "
                        f"[{'FROZEN' if ep<=FREEZE_EPOCHS else 'FULL'}]",
                   leave=False, ncols=95)
        for X, y in bar:
            X, y = X.to(device), y.to(device); opt.zero_grad()
            lo = model(X); ls = crit(lo, y); ls.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            tl += ls.item() * X.size(0)
            tc += (lo.argmax(-1) == y).sum().item()
            tn += X.size(0)
            bar.set_postfix(loss=f"{ls.item():.3f}")

        sch.step()
        tl /= tn; ta = tc / tn

        # --- Validate ---
        model.eval(); vl = 0; vp = []; vt_ = []
        with torch.no_grad():
            for X, y in va_ld:
                X, y = X.to(device), y.to(device); lo = model(X)
                vl += crit(lo, y).item() * X.size(0)
                vp.extend(lo.argmax(-1).cpu().numpy())
                vt_.extend(y.cpu().numpy())
        vl /= len(va_ds); va = accuracy_score(vt_, vp)
        vprc, vrc, vf1, _ = precision_recall_fscore_support(
            vt_, vp, average="weighted", zero_division=0)

        for k, v in [("tl",tl),("ta",ta),("vl",vl),("va",va),
                     ("vp",vprc),("vr",vrc),("vf1",vf1)]:
            hist[k].append(v)

        phase_tag = "FROZEN" if ep <= FREEZE_EPOCHS else "FULL  "
        print(f"Ep {ep:02d}/{TOTAL_EPOCHS} [{phase_tag}] | "
              f"Train L={tl:.4f} Acc={ta*100:.1f}% | "
              f"Val L={vl:.4f} Acc={va*100:.1f}% F1={vf1:.4f} | "
              f"LR rnn={rnn_lr:.2e} head={head_lr:.2e} trans={trans_lr:.2e}",
              flush=True)

        if va > best_acc:
            best_acc = va
            torch.save(model.state_dict(), best_p)
            print(f"  * New best val acc: {best_acc*100:.2f}%")

    # --- Save ---
    torch.save(model.state_dict(), final_p)
    print(f"\n[DONE] Best Val Acc: {best_acc*100:.2f}%")

    with open(os.path.join(SAVE_DIR, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)
    with open(os.path.join(SAVE_DIR, "history.json"), "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)

    # --- Final evaluation ---
    print("\n[EVAL] Final evaluation (best checkpoint) ...")
    model.load_state_dict(torch.load(best_p, weights_only=True))
    model.eval(); fp2 = []; ft2 = []
    with torch.no_grad():
        for X, y in va_ld:
            X, y = X.to(device), y.to(device)
            fp2.extend(model(X).argmax(-1).cpu().numpy())
            ft2.extend(y.cpu().numpy())
    fa = accuracy_score(ft2, fp2)
    print(f"[EVAL] Final Val Acc: {fa*100:.2f}%")

    rep = classification_report(ft2, fp2, target_names=classes, zero_division=0)
    with open(os.path.join(SAVE_DIR, "classification_report.txt"),
              "w", encoding="utf-8") as f:
        f.write(f"Phase 4 — Full-FPS Evaluation\n{'='*60}\n{rep}")

    gen_plots(hist, ft2, fp2, classes)
    print(f"\n[SAVED] Checkpoints -> {SAVE_DIR}")
    print(f"[SAVED] Plots       -> {PLOTS_DIR}")


# =========================================================================
# 10. ENTRY
# =========================================================================

if __name__ == "__main__":
    train_phase4()
