"""Phase 5 - Maximum accuracy squeeze from Phase-4 checkpoint.

Design decisions
----------------
* Exact same train/val split as Phase 4 (loaded from split_paths.json).
  No data leakage between phases.
* All layers unfrozen from epoch 1 (no freezing schedule).
* Very low differential LRs:
    Transformer : 8e-6
    RNN + CNN   : 2e-5
    Head        : 3e-5
* OneCycleLR per epoch (pct_start=0.1) for the first 30 epochs.
* SWA (Stochastic Weight Averaging) kicks in at epoch 31, runs 15 epochs.
  SWA model is saved separately alongside the best standard checkpoint.
* Augmentation (train split only, after split):
    - Time-stretch  : 0.60 - 1.40x (stronger than Phase 4)
    - Flip          : p=0.5
    - Tilt          : +-20 deg
    - Gaussian noise: std=0.008 on joint coords
    - Mixup         : alpha=0.3 (between random training pairs)
* Label smoothing: 0.04
* Plots:
    loss_acc.png, metrics.png, lr_schedule.png,
    swa_vs_best.png, confusion_matrix.png
* Saves to: models/phase5_word_model/
"""

import os, sys, glob, json, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
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
PHASE4_CKPT    = os.path.join(BACKEND_DIR, "models", "phase4_word_model", "best_phase4_model.pth")
SAVE_DIR       = os.path.join(BACKEND_DIR, "models", "phase5_word_model")
PLOTS_DIR      = os.path.join(SAVE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# =========================================================================
# HYPER-PARAMETERS
# =========================================================================
INPUT_DIM     = 163
D_MODEL       = 128
N_HEAD        = 4
N_TRANS       = 2
GRU_LAYERS    = 2

TOTAL_EPOCHS  = 45
SWA_START     = 31          # SWA kicks in at this epoch
BATCH_SIZE    = 48          # slightly smaller for more gradient steps
WEIGHT_DECAY  = 5e-5
LABEL_SMOOTH  = 0.04
GRAD_CLIP     = 1.5

# LRs (all layers unfrozen from start)
LR_TRANS      = 8e-6
LR_RNN_CNN    = 2e-5
LR_HEAD       = 3e-5
SWA_LR        = 5e-6        # flat SWA learning rate

# Augmentation
STRETCH_LO    = 0.60
STRETCH_HI    = 1.40
FLIP_P        = 0.50
TILT_DEG      = 20.0
NOISE_STD     = 0.008
MIXUP_ALPHA   = 0.30


# =========================================================================
# 1. AUGMENTATION
# =========================================================================

def time_stretch(seq):
    T, D = seq.shape
    factor = random.uniform(STRETCH_LO, STRETCH_HI)
    new_T = max(1, round(T * factor))
    if new_T == T: return seq
    s = F.interpolate(seq.T.unsqueeze(0), size=new_T, mode="linear",
                      align_corners=True)
    return s.squeeze(0).T


def flip_pose(seq):
    seq = seq.clone(); T = seq.shape[0]
    coords = seq[:, :159].view(T, 53, 3).clone()
    coords[:, :, 0] = -coords[:, :, 0]
    lh = coords[:, 0:21, :].clone(); rh = coords[:, 21:42, :].clone()
    coords[:, 0:21, :] = rh;         coords[:, 21:42, :] = lh
    la = coords[:, [42,44], :].clone(); ra = coords[:, [43,45], :].clone()
    coords[:, [42,44], :] = ra;         coords[:, [43,45], :] = la
    lf = coords[:, [46,49], :].clone(); rf = coords[:, [47,50], :].clone()
    coords[:, [46,49], :] = rf;         coords[:, [47,50], :] = lf
    l2 = seq[:, 159].clone(); r2 = seq[:, 160].clone()
    seq[:, 159] = r2; seq[:, 160] = l2
    seq[:, :159] = coords.view(T, 159)
    return seq


def tilt_pose(seq):
    seq = seq.clone(); T = seq.shape[0]
    a = math.radians(random.uniform(-TILT_DEG, TILT_DEG))
    ca, sa = math.cos(a), math.sin(a)
    coords = seq[:, :159].view(T, 53, 3).clone()
    x, y = coords[:,:,0].clone(), coords[:,:,1].clone()
    coords[:,:,0] = x*ca - y*sa
    coords[:,:,1] = x*sa + y*ca
    if NOISE_STD > 0:
        coords += torch.randn_like(coords) * NOISE_STD
    seq[:, :159] = coords.view(T, 159)
    return seq


def augment(seq):
    seq = time_stretch(seq)
    if random.random() < FLIP_P:
        seq = flip_pose(seq)
    seq = tilt_pose(seq)
    return seq


def mixup(x1, y1, x2, y2, alpha=MIXUP_ALPHA):
    """Mixup two (padded) sequences of the same temporal length."""
    lam = np.random.beta(alpha, alpha)
    # Match lengths by padding the shorter one
    T1, T2 = x1.shape[0], x2.shape[0]
    T = max(T1, T2)
    D = x1.shape[1]
    xm1 = torch.zeros(T, D); xm1[:T1] = x1
    xm2 = torch.zeros(T, D); xm2[:T2] = x2
    x_mix = lam * xm1 + (1 - lam) * xm2
    return x_mix, y1, y2, lam


# =========================================================================
# 2. COLLATE
# =========================================================================

def collate_pad(batch):
    seqs, labels = zip(*batch)
    T_max = max(s.shape[0] for s in seqs)
    D     = seqs[0].shape[1]
    padded = torch.zeros(len(seqs), T_max, D)
    for i, s in enumerate(seqs):
        padded[i, :s.shape[0]] = s
    return padded, torch.tensor(labels, dtype=torch.long)


# =========================================================================
# 3. DATASETS  (load from saved split_paths.json)
# =========================================================================

class TrainDS(Dataset):
    def __init__(self, split: dict, c2i: dict):
        self.items = []
        for cls, paths in split["train"].items():
            idx = c2i[cls]
            for p in paths:
                self.items.append((p, idx))
        random.shuffle(self.items)

    def __len__(self): return len(self.items)

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

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        seq = torch.load(path, map_location="cpu", weights_only=True).float()
        return seq, label   # NO augmentation


# =========================================================================
# 4. MODEL  (identical to Phase 4 architecture)
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


class Phase5Model(nn.Module):
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
        h = self.proj(x)
        g, _ = self.gru(h); h = self.gnorm(h + g)
        h = self.pe(h); h = self.trans(h); h = self.tnorm(h)
        h = h.transpose(1, 2)
        h = self.t1(h); h = self.t2(h)
        p = torch.cat([self.ap(h).squeeze(-1), self.mp(h).squeeze(-1)], -1)
        return self.head(p)


# =========================================================================
# 5. WEIGHT LOADING
# =========================================================================

def build_model(nc, ckpt):
    model = Phase5Model(nc)
    if not os.path.exists(ckpt):
        print(f"[WARN] Phase-4 checkpoint not found: {ckpt} -- random init")
        return model
    raw = torch.load(ckpt, map_location="cpu", weights_only=True)
    miss, unex = model.load_state_dict(raw, strict=False)
    loaded = len(raw) - len(miss)
    print(f"[INFO] Phase-4 weights loaded: {loaded}/{len(raw)} keys matched.")
    if miss:  print(f"  Missing:    {miss}")
    if unex:  print(f"  Unexpected: {unex}")
    return model


# =========================================================================
# 6. MIXUP LOSS HELPER
# =========================================================================

def mixup_criterion(crit, pred, y_a, y_b, lam):
    return lam * crit(pred, y_a) + (1 - lam) * crit(pred, y_b)


# =========================================================================
# 7. PLOTS
# =========================================================================
BG  = "#0f1117"; PAN = "#1a1d27"; TXT = "#e6edf3"; MUT = "#8b949e"; BRD = "#30363d"
LEG = dict(facecolor="#21262d", labelcolor="#c9d1d9", edgecolor="#30363d", fontsize=9)

def sax(ax):
    ax.set_facecolor(PAN); ax.tick_params(colors=MUT)
    for s in ax.spines.values(): s.set_edgecolor(BRD)
    ax.grid(True, ls="--", alpha=.2, color=MUT)

def plot_loss_acc(h):
    ep = range(1, len(h["tl"]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5)); fig.patch.set_facecolor(BG)
    sax(a1); sax(a2)
    a1.plot(ep, h["tl"], color="#f85149", lw=2, label="Train Loss (w/ Mixup)")
    a1.plot(ep, h["vl"], color="#ff9500", lw=2, ls="--", label="Val Loss")
    a1.axvline(SWA_START - 0.5, color="#58a6ff", ls=":", lw=1.5,
               label=f"SWA start (Ep {SWA_START})")
    a1.set_title("Loss — Phase 5 (Mixup + SWA)", color=TXT, fontsize=12, fontweight="bold")
    a1.set_xlabel("Epoch", color=MUT); a1.set_ylabel("Loss", color=MUT); a1.legend(**LEG)
    a2.plot(ep, [a*100 for a in h["ta"]], color="#58a6ff", lw=2, label="Train Acc%")
    a2.plot(ep, [a*100 for a in h["va"]], color="#3fb950", lw=2, ls="--", label="Val Acc%")
    a2.axvline(SWA_START - 0.5, color="#f0c040", ls=":", lw=1.5,
               label=f"SWA start (Ep {SWA_START})")
    a2.set_title("Accuracy — Phase 5 (full FPS)", color=TXT, fontsize=12, fontweight="bold")
    a2.set_xlabel("Epoch", color=MUT); a2.set_ylabel("Acc %", color=MUT); a2.legend(**LEG)
    plt.suptitle("Phase 5 — Maximum Accuracy Squeeze", color=TXT, fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "loss_acc.png")
    plt.savefig(p, dpi=200, bbox_inches="tight", facecolor=BG); plt.close(); print(f"  [SAVED] {p}")

def plot_metrics(h):
    ep = range(1, len(h["vf1"]) + 1)
    fig, ax = plt.subplots(figsize=(12, 5)); fig.patch.set_facecolor(BG); sax(ax)
    ax.plot(ep, h["vf1"], color="#bc8cff", lw=2.5, label="Weighted F1")
    ax.plot(ep, h["vp"],  color="#58a6ff", lw=2,   ls=":",  label="Precision")
    ax.plot(ep, h["vr"],  color="#3fb950", lw=2,   ls="--", label="Recall")
    ax.axvline(SWA_START - 0.5, color="#f0c040", ls=":", lw=1.5, label=f"SWA start (Ep {SWA_START})")
    ax.set_title("Validation Metrics — Phase 5", color=TXT, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", color=MUT); ax.set_ylabel("Score", color=MUT)
    ax.set_ylim(0, 1); ax.legend(**LEG); plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "metrics.png")
    plt.savefig(p, dpi=200, bbox_inches="tight", facecolor=BG); plt.close(); print(f"  [SAVED] {p}")

def plot_lr(h):
    ep = range(1, len(h["lr_trans"]) + 1)
    fig, ax = plt.subplots(figsize=(12, 3)); fig.patch.set_facecolor(BG); sax(ax)
    ax.plot(ep, h["lr_trans"],   color="#f85149", lw=2,   label="Transformer LR")
    ax.plot(ep, h["lr_rnn"],     color="#58a6ff", lw=2,   ls="--", label="RNN+CNN LR")
    ax.plot(ep, h["lr_head"],    color="#3fb950", lw=2,   ls=":", label="Head LR")
    ax.axvline(SWA_START - 0.5, color="#f0c040", ls=":", lw=1.5, label=f"SWA LR={SWA_LR:.0e}")
    ax.set_title("LR Schedule (OneCycleLR -> SWA flat)", color=TXT, fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", color=MUT); ax.set_ylabel("LR", color=MUT)
    ax.set_yscale("log"); ax.legend(**LEG); plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "lr_schedule.png")
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG); plt.close(); print(f"  [SAVED] {p}")

def plot_swa_compare(h):
    ep = range(1, len(h["va"]) + 1)
    fig, ax = plt.subplots(figsize=(12, 4)); fig.patch.set_facecolor(BG); sax(ax)
    ax.plot(ep, [a*100 for a in h["va"]],    color="#3fb950", lw=2,   label="Standard Val Acc%")
    swa_accs = h.get("swa_va", [])
    if swa_accs:
        swa_ep = range(SWA_START, SWA_START + len(swa_accs))
        ax.plot(swa_ep, [a*100 for a in swa_accs], color="#bc8cff", lw=2.5,
                ls="--", label="SWA Val Acc%")
    ax.axvline(SWA_START - 0.5, color="#f0c040", ls=":", lw=1.5, label="SWA start")
    ax.set_title("Standard Model vs SWA Model — Validation Accuracy", color=TXT,
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", color=MUT); ax.set_ylabel("Acc %", color=MUT)
    ax.legend(**LEG); plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "swa_vs_best.png")
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG); plt.close(); print(f"  [SAVED] {p}")

def plot_cm(yt, yp, classes, tag=""):
    cm = confusion_matrix(yt, yp); n = len(classes); sz = max(12, n * 0.07)
    fig, ax = plt.subplots(figsize=(sz, sz * 0.9)); fig.patch.set_facecolor(BG)
    ax.set_facecolor(PAN)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title(f"Confusion Matrix — Phase 5 {tag} ({n} classes)",
                 color=TXT, fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted", color=MUT); ax.set_ylabel("True", color=MUT)
    fs = max(3.5, 6.5 - n * 0.012)
    ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n))
    ax.set_xticklabels(classes, rotation=90, fontsize=fs, color="#c9d1d9")
    ax.set_yticklabels(classes, fontsize=fs, color="#c9d1d9")
    plt.tight_layout()
    fname = f"confusion_matrix{'_' + tag if tag else ''}.png"
    p = os.path.join(PLOTS_DIR, fname)
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG); plt.close(); print(f"  [SAVED] {p}")

def gen_plots(h, yt, yp, classes, yt_s=None, yp_s=None):
    print("[INFO] Generating plots ...")
    plot_loss_acc(h); plot_metrics(h); plot_lr(h); plot_swa_compare(h)
    plot_cm(yt, yp, classes, "best")
    if yt_s is not None:
        plot_cm(yt_s, yp_s, classes, "SWA")


# =========================================================================
# 8. TRAINING LOOP
# =========================================================================

def evaluate(model, loader, crit, device, dataset_len):
    model.eval(); vl = 0; vp = []; vt_ = []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            lo = model(X)
            vl += crit(lo, y).item() * X.size(0)
            vp.extend(lo.argmax(-1).cpu().numpy())
            vt_.extend(y.cpu().numpy())
    vl /= dataset_len
    va  = accuracy_score(vt_, vp)
    vpr, vrc, vf1, _ = precision_recall_fscore_support(
        vt_, vp, average="weighted", zero_division=0)
    return vl, va, vpr, vrc, vf1, vt_, vp


def train_phase5():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    # --- Load exact Phase-4 split ---
    if not os.path.exists(SPLIT_JSON):
        raise FileNotFoundError(
            f"[ERROR] Split file not found: {SPLIT_JSON}\n"
            "Run scripts/save_phase4_split.py first.")
    with open(SPLIT_JSON, encoding="utf-8") as f:
        split = json.load(f)
    classes = split["classes"]
    c2i = {c: i for i, c in enumerate(classes)}
    nc  = len(classes)
    print(f"[INFO] Loaded split: {nc} classes, "
          f"{sum(len(v) for v in split['train'].values())} train, "
          f"{sum(len(v) for v in split['val'].values())} val")

    tr_ds = TrainDS(split, c2i)
    va_ds = ValDS(split, c2i)
    tr_ld = DataLoader(tr_ds, BATCH_SIZE, shuffle=True, drop_last=True,
                       num_workers=0, collate_fn=collate_pad)
    va_ld = DataLoader(va_ds, BATCH_SIZE, shuffle=False,
                       num_workers=0, collate_fn=collate_pad)

    # --- Model ---
    model = build_model(nc, PHASE4_CKPT).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Phase5Model: {total_p:,} params | {nc} classes")
    print(f"[INFO] ALL layers unfrozen from epoch 1")

    # --- Optimizer (differential LRs, all unfrozen) ---
    opt = optim.AdamW([
        {"params": model.transformer_params(), "lr": LR_TRANS,   "name": "trans"},
        {"params": model.rnn_params(),         "lr": LR_RNN_CNN, "name": "rnn"},
        {"params": model.cnn_params(),         "lr": LR_RNN_CNN, "name": "cnn"},
        {"params": model.head_params(),        "lr": LR_HEAD,    "name": "head"},
    ], weight_decay=WEIGHT_DECAY)

    steps_per_epoch = len(tr_ld)

    # OneCycleLR covers epochs 1 to SWA_START-1
    sch = optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=[LR_TRANS * 8, LR_RNN_CNN * 8, LR_RNN_CNN * 8, LR_HEAD * 8],
        steps_per_epoch=steps_per_epoch,
        epochs=SWA_START - 1,
        pct_start=0.10,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )


    # SWA model + scheduler
    swa_model = AveragedModel(model)
    swa_sch   = SWALR(opt, swa_lr=SWA_LR, anneal_epochs=3)

    crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    hist = {k: [] for k in ["tl","ta","vl","va","vp","vr","vf1",
                             "lr_trans","lr_rnn","lr_head","swa_va"]}
    best_acc = 0.
    best_p   = os.path.join(SAVE_DIR, "best_phase5_model.pth")
    final_p  = os.path.join(SAVE_DIR, "final_phase5_model.pth")
    swa_p    = os.path.join(SAVE_DIR, "swa_phase5_model.pth")

    def _get_lr(name):
        for pg in opt.param_groups:
            if pg.get("name") == name: return pg["lr"]
        return opt.param_groups[0]["lr"]

    print(f"\n[START] Phase 5 -- {TOTAL_EPOCHS} epochs | SWA from Ep {SWA_START}\n")

    for ep in range(1, TOTAL_EPOCHS + 1):

        in_swa = ep >= SWA_START

        # Record LRs
        hist["lr_trans"].append(_get_lr("trans"))
        hist["lr_rnn"].append(_get_lr("rnn"))
        hist["lr_head"].append(_get_lr("head"))

        # --- Train ---
        model.train(); tl = tc = tn = 0
        bar = tqdm(tr_ld,
                   desc=f"E{ep:02d}/{TOTAL_EPOCHS} [{'SWA' if in_swa else 'OneCycle'}]",
                   leave=False, ncols=95)

        for X, y in bar:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()

            # --- Mixup ---
            if MIXUP_ALPHA > 0 and random.random() < 0.5:
                idx2  = torch.randperm(X.size(0), device=device)
                X2, y2 = X[idx2], y[idx2]
                lam   = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
                # Mixup on same-padded-length batch (already padded by collate)
                X_mix = lam * X + (1 - lam) * X2
                lo    = model(X_mix)
                ls    = mixup_criterion(crit, lo, y, y2, lam)
                # For accuracy logging use non-mixed predictions
                with torch.no_grad(): lo_raw = model(X)
                tc += (lo_raw.argmax(-1) == y).sum().item()
            else:
                lo = model(X); ls = crit(lo, y)
                tc += (lo.argmax(-1) == y).sum().item()

            ls.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            if not in_swa: sch.step()   # OneCycleLR steps per batch

            tl += ls.item() * X.size(0); tn += X.size(0)
            bar.set_postfix(loss=f"{ls.item():.3f}")

        tl /= tn; ta = tc / tn

        # SWA update (after warmup)
        if in_swa:
            swa_model.update_parameters(model)
            swa_sch.step()

        # --- Validate (standard model) ---
        vl, va, vpr, vrc, vf1, vt_, vp_ = evaluate(model, va_ld, crit, device, len(va_ds))
        for k, v in [("tl",tl),("ta",ta),("vl",vl),("va",va),
                     ("vp",vpr),("vr",vrc),("vf1",vf1)]:
            hist[k].append(v)

        # --- Validate SWA model (every epoch during SWA phase) ---
        swa_acc_str = ""
        if in_swa:
            update_bn(tr_ld, swa_model, device=device)
            _, swa_va, _, _, _, _, _ = evaluate(swa_model, va_ld, crit, device, len(va_ds))
            hist["swa_va"].append(swa_va)
            swa_acc_str = f" | SWA Acc={swa_va*100:.1f}%"

        phase_tag = "SWA     " if in_swa else "OneCycle"
        print(f"Ep {ep:02d}/{TOTAL_EPOCHS} [{phase_tag}] | "
              f"Train L={tl:.4f} Acc={ta*100:.1f}% | "
              f"Val L={vl:.4f} Acc={va*100:.1f}% F1={vf1:.4f}"
              f"{swa_acc_str} | "
              f"LR t={_get_lr('trans'):.2e} r={_get_lr('rnn'):.2e}",
              flush=True)

        if va > best_acc:
            best_acc = va
            torch.save(model.state_dict(), best_p)
            print(f"  * New best standard: {best_acc*100:.2f}%")

    # --- Final: save, update BN for SWA, evaluate both models ---
    torch.save(model.state_dict(), final_p)
    print(f"\n[DONE] Best Standard Val Acc: {best_acc*100:.2f}%")

    # Save SWA model (final BN update on full train set)
    print("[INFO] Final SWA BN update ...")
    update_bn(tr_ld, swa_model, device=device)
    torch.save(swa_model.state_dict(), swa_p)
    _, swa_final, _, _, _, ft_s, fp_s = evaluate(
        swa_model, va_ld, crit, device, len(va_ds))
    print(f"[DONE] SWA Final Val Acc: {swa_final*100:.2f}%")

    # Best model final eval
    model.load_state_dict(torch.load(best_p, weights_only=True))
    _, best_final, _, _, _, ft_b, fp_b = evaluate(
        model, va_ld, crit, device, len(va_ds))
    print(f"[DONE] Best Model Final Val Acc: {best_final*100:.2f}%")

    # Determine which to use as the canonical output
    final_classes = classes
    canonical_acc = max(swa_final, best_final)
    print(f"[DONE] Canonical Val Acc: {canonical_acc*100:.2f}% "
          f"({'SWA' if swa_final >= best_final else 'Best'})")

    with open(os.path.join(SAVE_DIR, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(final_classes, f, indent=2)
    with open(os.path.join(SAVE_DIR, "history.json"), "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)

    rep_b = classification_report(ft_b, fp_b, target_names=classes, zero_division=0)
    rep_s = classification_report(ft_s, fp_s, target_names=classes, zero_division=0)
    with open(os.path.join(SAVE_DIR, "classification_report.txt"),
              "w", encoding="utf-8") as f:
        f.write(f"Phase 5 — Best Model\n{'='*60}\n{rep_b}\n\n")
        f.write(f"Phase 5 — SWA Model\n{'='*60}\n{rep_s}\n")

    gen_plots(hist, ft_b, fp_b, classes, ft_s, fp_s)
    print(f"\n[SAVED] Checkpoints -> {SAVE_DIR}")
    print(f"[SAVED] Plots       -> {PLOTS_DIR}")


# =========================================================================
# 9. ENTRY
# =========================================================================

if __name__ == "__main__":
    train_phase5()
