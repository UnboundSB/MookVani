"""Phase 3 - Full word-level fine-tuning from Phase-2 checkpoint.

Strategy
--------
1. Load Phase-2 best checkpoint (Phase2Reconformer, d_model=128).
2. Replace classifier head for Phase-3 class count.
3. Freeze backbone (input_dense, bigru, transformer, TCNN) for FREEZE_EPOCHS,
   then unfreeze with differential LR.
4. Curriculum FPS: 1->25 fps linearly over TOTAL_EPOCHS.
5. All classes with >=1 sample kept; cap at MAX_SAMPLES_PER_CLASS (14).
6. Augmentation ONLY on train split (after train/val split).
7. Saves: models/phase3_word_model/{best,final}_phase3_model.pth,
          classes.json, history.json, classification_report.txt,
          plots/{acc_vs_loss,metrics,fps_schedule,confusion_matrix}.png
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
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BACKEND_DIR, "data", "tensors_include50_word_level")
PHASE2_CKPT = os.path.join(BACKEND_DIR, "models", "pretrained_phase_2_model",
                            "best_phase2_model.pth")
SAVE_DIR    = os.path.join(BACKEND_DIR, "models", "phase3_word_model")
PLOTS_DIR   = os.path.join(SAVE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from utils.dataloader_word import (  # noqa: E402
    discover_word_classes,
    load_split_tensors,
    load_word_split_tensors,
    split_word_classes,
)
from utils.pose_augmentation import (  # noqa: E402
    AugmentConfig,
    augment_pose_sequence as augment_pose_sequence_canonical,
)
from utils.checkpoint_utils import load_flexible  # noqa: E402
from utils.training_utils import (  # noqa: E402
    build_warm_restart_scheduler,
    escape_from_plateau,
    unfreeze_and_extend,
)

# =========================================================================
# HYPER-PARAMETERS
# =========================================================================
MIN_SAMPLES = 14          # a class below this tier is eliminated from the loader
MAX_SAMPLES    = 20          # hard budget of clips per class
SOURCE_FPS  = 25.0        # reference rate the corpus was measured at (NOT a cap)
START_FPS   = 1.0
END_FPS     = 50.0        # the curriculum may go ABOVE the source rate
MEDIAN_T    = 62
INPUT_DIM   = 163
D_MODEL     = 128
N_HEAD      = 4
N_TRANS     = 2
GRU_LAYERS  = 2
FREEZE_EP   = 5
TOTAL_EP    = 40
RESTART_EVERY  = 20        # cosine warm restart every N epochs (plateau recovery)
ESCAPE_PATIENCE = 10       # epochs without a val improvement -> escape the plateau
ESCAPE_SCALE    = 0.02     # sigma of the weight perturbation of the escape
BATCH_SZ    = 64
HEAD_LR     = 5e-4
BB_LR       = 5e-5
VAL_FRAC    = 0.20
WD          = 1e-4
LS          = 0.06
GRAD_CLIP   = 2.0

# =========================================================================
# 1. TEMPORAL SAMPLING
# =========================================================================

def center_out(T, n):
    n = min(n, T); mid = T // 2; picks = [mid]; r, l = mid+1, mid-1
    while len(picks) < n:
        if r < T and len(picks) < n: picks.append(r); r += 1
        if l >= 0 and len(picks) < n: picks.append(l); l -= 1
    return sorted(picks[:n])

def fps_for_epoch(ep, total):
    return START_FPS + (END_FPS - START_FPS) * (ep-1) / max(total-1, 1)

def target_T(fps):
    return max(1, round(MEDIAN_T * fps / SOURCE_FPS))

def sample_norm(tensor, fps):
    """Center-out sample (when downsampling) then interpolate to the target.

    The SOURCE frame rate is not a ceiling: above SOURCE_FPS every source frame
    is kept and the clip is interpolated UP to the denser target.
    """
    T = tensor.shape[0]
    n_pick = max(1, round(T * fps / SOURCE_FPS))
    s = tensor[center_out(T, n_pick)] if n_pick < T else tensor
    tgt = target_T(fps)
    if s.shape[0] == tgt: return s
    s = F.interpolate(s.T.unsqueeze(0), size=tgt, mode="linear",
                      align_corners=True)
    return s.squeeze(0).T

# =========================================================================
# 2. AUGMENTATION  (train only)
# =========================================================================

def augment(seq, **kwargs):
    """Geometric augmentation (train only).

    Delegates to the canonical implementation in
    ``Backend/utils/pose_augmentation.py``: the mirror swaps every L/R pair
    (mouth corners included) around the signer's own axis, the tilt rotates
    around the neck and the shift is bounded so nothing leaves the frame.
    """
    return augment_pose_sequence_canonical(seq, PHASE3_AUGMENT)


PHASE3_AUGMENT = AugmentConfig(
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

# =========================================================================
# 3. DATASETS
# =========================================================================

class TrainDS(Dataset):
    def __init__(self, data, fps=START_FPS):
        self.data = data; self.fps = fps
        self.items = [(c,i) for c,ts in data.items() for i in range(len(ts))]
    def set_fps(self, fps): self.fps = fps
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        c, i = self.items[idx]
        return augment(sample_norm(self.data[c][i], self.fps)), c

class ValDS(Dataset):
    def __init__(self, data, fps=END_FPS):
        self.data = data; self.fps = fps
        self.items = [(c,i) for c,ts in data.items() for i in range(len(ts))]
    def set_fps(self, fps): self.fps = fps
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        c, i = self.items[idx]
        return sample_norm(self.data[c][i], self.fps), c

# =========================================================================
# 4. MODEL  (identical to Phase2Reconformer)
# =========================================================================

class PosEnc(nn.Module):
    def __init__(self, d, maxlen=512):
        super().__init__()
        pe = torch.zeros(maxlen, d)
        pos = torch.arange(0, maxlen).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.)/d))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:,:x.size(1),:]

class TCNN(nn.Module):
    def __init__(self, ic, oc, k=3, drop=.1):
        super().__init__(); p=k//2
        self.c1=nn.Conv1d(ic,oc,k,padding=p); self.b1=nn.BatchNorm1d(oc)
        self.g1=nn.GELU(); self.d1=nn.Dropout(drop)
        self.c2=nn.Conv1d(oc,oc,k,padding=p); self.b2=nn.BatchNorm1d(oc)
        self.g2=nn.GELU(); self.d2=nn.Dropout(drop)
        self.res=nn.Conv1d(ic,oc,1) if ic!=oc else nn.Identity()
    def forward(self, x):
        r=self.res(x)
        o=self.d1(self.g1(self.b1(self.c1(x))))
        o=self.d2(self.g2(self.b2(self.c2(o))))
        return o+r

class Phase3Model(nn.Module):
    def __init__(self, nc, d=D_MODEL, gru=GRU_LAYERS,
                 nh=N_HEAD, nt=N_TRANS, drop=.15):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(INPUT_DIM,d), nn.LayerNorm(d), nn.Dropout(drop))
        self.gru  = nn.GRU(d, d//2, gru, batch_first=True, bidirectional=True,
                           dropout=drop if gru>1 else 0.)
        self.gnorm= nn.LayerNorm(d)
        self.pe   = PosEnc(d)
        el = nn.TransformerEncoderLayer(d, nh, d*2, drop, "gelu", batch_first=True)
        self.trans= nn.TransformerEncoder(el, nt, enable_nested_tensor=False)
        self.tnorm= nn.LayerNorm(d)
        self.t1   = TCNN(d, d, drop=drop)
        self.t2   = TCNN(d, d*2, drop=drop)
        self.ap   = nn.AdaptiveAvgPool1d(1)
        self.mp   = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(d*4, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Dropout(drop*2), nn.Linear(256, nc))

    def backbone_params(self):
        return (list(self.proj.parameters()) + list(self.gru.parameters()) +
                list(self.gnorm.parameters()) + list(self.pe.parameters()) +
                list(self.trans.parameters()) + list(self.tnorm.parameters()) +
                list(self.t1.parameters()) + list(self.t2.parameters()))

    def head_params(self): return list(self.head.parameters())

    def forward(self, x):
        h = self.proj(x)
        g, _ = self.gru(h); h = self.gnorm(h+g)
        h = self.pe(h); h = self.trans(h); h = self.tnorm(h)
        h = h.transpose(1,2); h = self.t1(h); h = self.t2(h)
        p = torch.cat([self.ap(h).squeeze(-1), self.mp(h).squeeze(-1)], -1)
        return self.head(p)

# =========================================================================
# 5. WEIGHT LOADING
# =========================================================================

def build_model(nc, ckpt):
    model = Phase3Model(nc)
    if not os.path.exists(ckpt):
        print(f"[WARN] Checkpoint not found: {ckpt} -- random init")
        return model
    raw = torch.load(ckpt, map_location="cpu", weights_only=True)
    remap = {}
    for k, v in raw.items():
        if k.startswith("classifier."): continue
        k = k.replace("input_dense.", "proj.").replace("input_proj.", "proj.")
        k = k.replace("bigru.", "gru.").replace("pos_encoder.", "pe.")
        k = k.replace("transformer_norm.", "tnorm.").replace("transformer.", "trans.")
        k = k.replace("gru_norm.", "gnorm.")
        k = k.replace("tcnn_block1.", "t1.").replace("tcnn_block2.", "t2.")
        k = k.replace("tcnn.0.", "t1.").replace("tcnn.1.", "t2.")
        # Phase-2 TCNN sub-keys use conv1/bn1/conv2/bn2/residual;
        # our compact TCNN uses c1/b1/c2/b2/res
        if k.startswith("t1.") or k.startswith("t2."):
            k = (k.replace(".conv1.", ".c1.").replace(".bn1.", ".b1.")
                  .replace(".conv2.", ".c2.").replace(".bn2.", ".b2.")
                  .replace(".residual.", ".res."))
        remap[k] = v
    miss, unex = load_flexible(model, remap, label="PHASE2->PHASE3")
    loaded = len(remap) - len(miss)
    print(f"[INFO] Loaded {loaded}/{len(remap)} Phase-2 keys.")
    if miss:  print(f"  Missing (new): {miss}")
    if unex:  print(f"  Unexpected:   {unex}")
    return model

# =========================================================================
# 6. DATA LOADING
# =========================================================================

def load_data(data_dir, max_per=MAX_SAMPLES):
    """Load the word corpus with the shared selection rule.

    A class that owns fewer than MIN_SAMPLES clips is eliminated from the
    loader (the files stay on disk) and every surviving class is capped at
    MAX_SAMPLES clips.  The classes are ordered by descending clip count.
    """
    print(f"[INFO] Scanning {data_dir} ...")
    infos = discover_word_classes(data_dir, min_samples_per_class=MIN_SAMPLES,
                                  max_samples_per_class=max_per)
    classes = [i.name for i in infos]
    print(f"[INFO] {len(classes)} classes (capped at {max_per})")

    # The canonical manifest decides which clip is training data and which one
    # is validation data - the SAME assignment every other phase uses.
    tr, va, classes = load_word_split_tensors(data_dir, MIN_SAMPLES, max_per)
    tot_tr = sum(len(v) for v in tr.values())
    tot_va = sum(len(v) for v in va.values())
    print(f"[INFO] Split: {tot_tr} train | {tot_va} val | {len(classes)} classes")
    return tr, va, classes

# =========================================================================
# 7. PLOTS
# =========================================================================
BG="#0f1117"; PAN="#1a1d27"; TXT="#e6edf3"; MUT="#8b949e"; BRD="#30363d"
LEG=dict(facecolor="#21262d",labelcolor="#c9d1d9",edgecolor="#30363d",fontsize=9)

def sax(ax):
    ax.set_facecolor(PAN); ax.tick_params(colors=MUT)
    for s in ax.spines.values(): s.set_edgecolor(BRD)
    ax.grid(True, ls="--", alpha=.2, color=MUT)

def plot_loss_acc(h):
    ep=range(1,len(h["tl"])+1)
    fig,(a1,a2)=plt.subplots(1,2,figsize=(14,5)); fig.patch.set_facecolor(BG)
    sax(a1); sax(a2)
    a1.plot(ep,h["tl"],color="#f85149",lw=2,label="Train Loss")
    a1.plot(ep,h["vl"],color="#ff9500",lw=2,ls="--",label="Val Loss")
    a1.set_title("Loss",color=TXT,fontsize=12,fontweight="bold")
    a1.set_xlabel("Epoch",color=MUT); a1.set_ylabel("CE Loss",color=MUT)
    a1.legend(**LEG)
    a2.plot(ep,[a*100 for a in h["ta"]],color="#58a6ff",lw=2,label="Train Acc%")
    a2.plot(ep,[a*100 for a in h["va"]],color="#3fb950",lw=2,ls="--",label="Val Acc%")
    a2.set_title("Accuracy (curriculum fps)",color=TXT,fontsize=12,fontweight="bold")
    a2.set_xlabel("Epoch",color=MUT); a2.set_ylabel("Acc %",color=MUT)
    a2.legend(**LEG)
    plt.suptitle("Phase 3 Word-level Fine-tuning",color=TXT,fontsize=13,fontweight="bold")
    plt.tight_layout()
    p=os.path.join(PLOTS_DIR,"acc_vs_loss.png")
    plt.savefig(p,dpi=200,bbox_inches="tight",facecolor=BG); plt.close()
    print(f"  [SAVED] {p}")

def plot_metrics(h):
    ep=range(1,len(h["vf1"])+1)
    fig,ax=plt.subplots(figsize=(10,5)); fig.patch.set_facecolor(BG); sax(ax)
    ax.plot(ep,h["vf1"],color="#bc8cff",lw=2.5,label="Weighted F1")
    ax.plot(ep,h["vp"],color="#58a6ff",lw=2,ls=":",label="Precision")
    ax.plot(ep,h["vr"],color="#3fb950",lw=2,ls="--",label="Recall")
    ax.set_title("Validation Metrics",color=TXT,fontsize=12,fontweight="bold")
    ax.set_xlabel("Epoch",color=MUT); ax.set_ylabel("Score",color=MUT)
    ax.legend(**LEG); plt.tight_layout()
    p=os.path.join(PLOTS_DIR,"metrics.png")
    plt.savefig(p,dpi=200,bbox_inches="tight",facecolor=BG); plt.close()
    print(f"  [SAVED] {p}")

def plot_fps(h):
    fl=h.get("fps",[]); 
    if not fl: return
    fig,ax=plt.subplots(figsize=(10,3)); fig.patch.set_facecolor(BG); sax(ax)
    ax.plot(range(1,len(fl)+1),fl,color="#f0c040",lw=2.5)
    ax.fill_between(range(1,len(fl)+1),fl,alpha=.2,color="#f0c040")
    ax.set_title("Curriculum FPS Schedule",color=TXT,fontsize=11,fontweight="bold")
    ax.set_xlabel("Epoch",color=MUT); ax.set_ylabel("FPS",color=MUT)
    plt.tight_layout()
    p=os.path.join(PLOTS_DIR,"fps_schedule.png")
    plt.savefig(p,dpi=150,bbox_inches="tight",facecolor=BG); plt.close()
    print(f"  [SAVED] {p}")

def plot_cm(yt,yp,classes):
    cm=confusion_matrix(yt,yp); n=len(classes); sz=max(12,n*.07)
    fig,ax=plt.subplots(figsize=(sz,sz*.9)); fig.patch.set_facecolor(BG)
    ax.set_facecolor(PAN)
    im=ax.imshow(cm,interpolation="nearest",cmap="Blues")
    fig.colorbar(im,ax=ax,fraction=.03,pad=.02)
    ax.set_title(f"Confusion Matrix ({n} classes)",color=TXT,fontsize=10,fontweight="bold")
    ax.set_xlabel("Predicted",color=MUT); ax.set_ylabel("True",color=MUT)
    fs=max(3.5,6.5-n*.012)
    ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n))
    ax.set_xticklabels(classes,rotation=90,fontsize=fs,color="#c9d1d9")
    ax.set_yticklabels(classes,fontsize=fs,color="#c9d1d9")
    plt.tight_layout()
    p=os.path.join(PLOTS_DIR,"confusion_matrix.png")
    plt.savefig(p,dpi=150,bbox_inches="tight",facecolor=BG); plt.close()
    print(f"  [SAVED] {p}")

def gen_plots(h, yt, yp, classes):
    print("[INFO] Generating plots ...")
    plot_loss_acc(h); plot_metrics(h); plot_fps(h); plot_cm(yt,yp,classes)

# =========================================================================
# 8. TRAINING
# =========================================================================

def train_phase3():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    train_data, val_data, classes = load_data(DATA_DIR)
    nc = len(classes)
    if nc == 0: raise RuntimeError(f"No classes in {DATA_DIR}")

    tr_ds = TrainDS(train_data, START_FPS)
    va_ds = ValDS(val_data, START_FPS)
    if len(tr_ds) == 0:
        raise RuntimeError("Training dataset is empty after split. "
                           "Check data or reduce VAL_FRAC.")

    tr_ld = DataLoader(tr_ds, BATCH_SZ, shuffle=True,  drop_last=True,  num_workers=0)
    va_ld = DataLoader(va_ds, BATCH_SZ, shuffle=False, num_workers=0)

    model = build_model(nc, PHASE2_CKPT).to(device)
    print(f"[INFO] Phase3Model: {sum(p.numel() for p in model.parameters()):,} params | {nc} classes")

    for p in model.backbone_params(): p.requires_grad = False
    opt = optim.AdamW(model.head_params(), lr=HEAD_LR, weight_decay=WD)
    for g in opt.param_groups: g["base_lr"] = g["lr"]
    print(f"[INFO] Epochs 1-{FREEZE_EP}: backbone FROZEN")

    crit = nn.CrossEntropyLoss(label_smoothing=LS)
    # Cosine WITH WARM RESTARTS so a plateau can be escaped instead of the LR
    # annealing to zero for the rest of the phase.
    sch, restart_every = build_warm_restart_scheduler(
        opt, total_epochs=TOTAL_EP, restart_every=RESTART_EVERY, eta_min=1e-6)
    hist = {k:[] for k in ["tl","ta","vl","va","vp","vr","vf1","fps"]}
    best_acc=0.; epochs_without_improvement=0; escapes=0
    best_p=os.path.join(SAVE_DIR,"best_phase3_model.pth")
    final_p=os.path.join(SAVE_DIR,"final_phase3_model.pth")

    print(f"\n[START] Phase 3 -- {TOTAL_EP} epochs, 1->50fps curriculum, "
          f"warm restart every {restart_every} epochs\n")

    for ep in range(1, TOTAL_EP+1):
        if ep == FREEZE_EP+1:
            print(f"\n[UNFREEZE] Epoch {ep}: backbone released @ lr={BB_LR}")
            # The optimiser is EXTENDED, not recreated: the AdamW moments of the
            # head are kept and the LR schedule is not thrown away.
            unfreeze_and_extend(opt, model.backbone_params(), lr=BB_LR, name="backbone")
            for g in opt.param_groups: g["lr"] = g.get("base_lr", g["lr"])

        fps = fps_for_epoch(ep, TOTAL_EP); T = target_T(fps)
        tr_ds.set_fps(fps); va_ds.set_fps(fps); hist["fps"].append(round(fps,2))

        # train
        model.train(); tl=tc=tn=0
        bar=tqdm(tr_ld, desc=f"E{ep:02d}/{TOTAL_EP} [{fps:.1f}fps T={T}]",
                 leave=False, ncols=90)
        for X,y in bar:
            X,y=X.to(device),y.to(device); opt.zero_grad()
            lo=model(X); ls=crit(lo,y); ls.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            tl+=ls.item()*X.size(0); tc+=(lo.argmax(-1)==y).sum().item(); tn+=X.size(0)
            bar.set_postfix(loss=f"{ls.item():.3f}")
        sch.step()
        tl/=tn; ta=tc/tn

        # val
        model.eval(); vl=0; vp=[]; vt_=[]
        with torch.no_grad():
            for X,y in va_ld:
                X,y=X.to(device),y.to(device); lo=model(X)
                vl+=crit(lo,y).item()*X.size(0)
                vp.extend(lo.argmax(-1).cpu().numpy())
                vt_.extend(y.cpu().numpy())
        vl/=len(va_ds); va=accuracy_score(vt_,vp)
        vprc,vrc,vf1,_=precision_recall_fscore_support(vt_,vp,average="weighted",zero_division=0)
        for k,v in [("tl",tl),("ta",ta),("vl",vl),("va",va),
                    ("vp",vprc),("vr",vrc),("vf1",vf1)]:
            hist[k].append(v)
        print(f"Ep {ep:02d}/{TOTAL_EP} | fps={fps:.1f} T={T:3d} | "
              f"Train L={tl:.4f} Acc={ta*100:.1f}% | "
              f"Val L={vl:.4f} Acc={va*100:.1f}% F1={vf1:.4f}", flush=True)
        if va > best_acc:
            best_acc=va; torch.save(model.state_dict(), best_p)
            print(f"  * New best: {best_acc*100:.2f}%")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # --- plateau escape -------------------------------------------------
        if ESCAPE_PATIENCE > 0 and epochs_without_improvement >= ESCAPE_PATIENCE:
            escapes += 1
            print(f"  [ESCAPE] No val improvement for {epochs_without_improvement} epochs "
                  f"(val acc {va*100:.2f}%) - kicking the model out of the plateau.")
            print(f"  [ESCAPE] {escape_from_plateau(model, opt, scale=ESCAPE_SCALE)}")
            epochs_without_improvement = 0

    torch.save(model.state_dict(), final_p)
    print(f"\n[DONE] Best Val Acc: {best_acc*100:.2f}% | escapes: {escapes}")
    with open(os.path.join(SAVE_DIR,"classes.json"),"w",encoding="utf-8") as f:
        json.dump(classes, f, indent=2)
    with open(os.path.join(SAVE_DIR,"history.json"),"w",encoding="utf-8") as f:
        json.dump(hist, f, indent=2)

    # Final eval at full fps
    print("\n[EVAL] Full 25fps evaluation ...")
    model.load_state_dict(torch.load(best_p, weights_only=True))
    model.eval(); va_ds.set_fps(END_FPS); fp2=[]; ft2=[]
    with torch.no_grad():
        for X,y in va_ld:
            X,y=X.to(device),y.to(device)
            fp2.extend(model(X).argmax(-1).cpu().numpy())
            ft2.extend(y.cpu().numpy())
    fa=accuracy_score(ft2,fp2)
    print(f"[EVAL] Full-fps Val Acc: {fa*100:.2f}%")
    rep=classification_report(ft2,fp2,target_names=classes,zero_division=0)
    with open(os.path.join(SAVE_DIR,"classification_report.txt"),"w",encoding="utf-8") as f:
        f.write(f"Phase 3 Full 25fps Eval\n{'='*60}\n{rep}")

    gen_plots(hist, ft2, fp2, classes)
    print(f"\n[SAVED] {SAVE_DIR}")
    print(f"[SAVED] {PLOTS_DIR}")

# =========================================================================
# 9. ENTRY
# =========================================================================

if __name__ == "__main__":
    train_phase3()
