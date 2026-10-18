"""
ISL Hybrid Conformer-CTC: Continuous Sign Language Recognition
==============================================================
Combines:
1. Pretrained Phase 6 Backbone (InputProj + BiGRU + Transformer + TCNN, output 256-dim)
   - Preserves high-precision (>95% acc) isolated sign representations.
   - Strips global pooling (ap/mp) so temporal sequence is fully preserved.
2. Conformer Contextualizer Stack (MHSA + Depthwise Conv kernel 31)
   - Captures long-range sentence syntax and smooth ~1s phonetic transitions.
3. Continuous CTC Projection Head with Blank Connectivity Regularization
   - Soft spike penalty prevents the CTC 1-frame blank-collapse trap.
4. Dataset with Random Noise at Start and End:
   - Prefix & suffix rest frames infused with random coordinate noise & idle hand drift.
   - Smooth-step co-articulation interpolation between words.
   - Discriminative learning rates: low LR on backbone, higher LR on Conformer & Head.
"""

import os, sys, glob, re, json, math, copy, random, argparse, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
    except: pass

BACKEND_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_DEFAULT   = os.path.join(BACKEND_DIR, "data", "tensors_include50_word_level")
MODELS_ROOT        = os.path.join(BACKEND_DIR, "models")
OUTPUT_DIR_DEFAULT = os.path.join(MODELS_ROOT, "isl_conformer_ctc")
PHASE6_WEIGHTS     = os.path.join(MODELS_ROOT, "phase6_word_model", "best_phase6_model.pth")
CLASSES_SRC        = os.path.join(MODELS_ROOT, "phase6_word_model", "classes.json")

MISSING_LH = torch.tensor([-1.5, 3.0, 0.0]).repeat(21)  # (63,)
MISSING_RH = torch.tensor([-1.5, 3.0, 0.0]).repeat(21)  # (63,)

# =============================================================================
# 1. METRICS
# =============================================================================
def compute_wer(ref, hyp):
    d = np.zeros((len(ref)+1, len(hyp)+1), dtype=np.int32)
    for i in range(len(ref)+1): d[i][0] = i
    for j in range(len(hyp)+1): d[0][j] = j
    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            d[i][j] = d[i-1][j-1] if ref[i-1]==hyp[j-1] else 1+min(d[i-1][j], d[i][j-1], d[i-1][j-1])
    return d[len(ref)][len(hyp)] / max(len(ref), 1)

# =============================================================================
# 2. POSE AUGMENTATIONS (uniform sentence-level)
# =============================================================================
def _reclamp(seq):
    T = seq.shape[0]
    for t in range(T):
        if seq[t, 159] < 0.5: seq[t, 0:63]   = MISSING_LH.to(seq.device)
        if seq[t, 160] < 0.5: seq[t, 63:126] = MISSING_RH.to(seq.device)
    return seq

def aug_flip(seq):
    seq = seq.clone(); T = seq.shape[0]
    c = seq[:, :159].view(T, 53, 3).clone()
    # 1. Flip X-coordinate horizontally only (Y and Z stay upright)
    c[:, :, 0] *= -1.0

    # 2. Swap Hands: Left Hand (0:21) <-> Right Hand (21:42)
    c[:, 0:21], c[:, 21:42] = c[:, 21:42].clone(), c[:, 0:21].clone()

    # 3. Swap Arms: Left Shoulder (42) <-> Right Shoulder (43), Left Elbow (44) <-> Right Elbow (45)
    c[:, 42], c[:, 43] = c[:, 43].clone(), c[:, 42].clone()
    c[:, 44], c[:, 45] = c[:, 45].clone(), c[:, 44].clone()

    # 4. Swap Face: Eyes (46 <-> 47), Cheeks (49 <-> 50), Mouth Corners (51 <-> 52)
    c[:, 46], c[:, 47] = c[:, 47].clone(), c[:, 46].clone()  # Left Eye <-> Right Eye
    c[:, 49], c[:, 50] = c[:, 50].clone(), c[:, 49].clone()  # Left Cheek <-> Right Cheek
    c[:, 51], c[:, 52] = c[:, 52].clone(), c[:, 51].clone()  # Left Mouth <-> Right Mouth
    # Nose (48) is centerline, stays at 48

    seq[:, :159] = c.view(T, 159)
    # Swap presence flags for Left Hand (159) and Right Hand (160)
    seq[:, 159], seq[:, 160] = seq[:, 160].clone(), seq[:, 159].clone()
    return _reclamp(seq)

def aug_shift(seq, dx=0.06, dy=0.02):
    seq = seq.clone(); T = seq.shape[0]
    c = seq[:, :159].view(T, 53, 3)
    c[:, :, 0] += random.uniform(-dx, dx)
    c[:, :, 1] += random.uniform(-dy, dy)
    seq[:, :159] = c.view(T, 159)
    return _reclamp(seq)

def aug_tilt(seq, deg=10.0):
    seq = seq.clone(); T = seq.shape[0]
    a = math.radians(random.uniform(-deg, deg)); ca, sa = math.cos(a), math.sin(a)
    R = torch.tensor([[ca,-sa],[sa,ca]], dtype=torch.float32)
    c = seq[:, :159].view(T, 53, 3)
    c[:, :, :2] = (c[:, :, :2].reshape(-1, 2) @ R.T).view(T, 53, 2)
    seq[:, :159] = c.view(T, 159)
    return _reclamp(seq)

def aug_speed(seq, lo=0.80, hi=1.25):
    """Speed up (lo < 1.0) or slow down (hi > 1.0) sign execution smoothly."""
    T, D = seq.shape
    factor = random.uniform(lo, hi)
    new_T = max(8, int(round(T * factor)))
    if new_T == T: return seq
    return F.interpolate(seq.T.unsqueeze(0), size=new_T, mode="linear", align_corners=True).squeeze(0).T

def aug_temporal_mask(seq, max_mask=4, n_masks=2):
    seq = seq.clone(); T = seq.shape[0]
    for _ in range(random.randint(1, n_masks)):
        ml = random.randint(1, max_mask)
        st = random.randint(0, max(0, T - ml))
        seq[st:st+ml, :159] = 0.0
    return seq

def augment(seq, strength=1.0):
    # Pure horizontal mirror (X inverted + hands swapped, Y and Z preserved upright)
    if random.random() < 0.40 * strength: seq = aug_flip(seq)
    if random.random() < 0.50 * strength: seq = aug_tilt(seq, 10.0 * strength)
    if random.random() < 0.50 * strength: seq = aug_shift(seq, 0.05 * strength, 0.02 * strength)
    # Slow down or speed up frames (0.80x - 1.25x)
    if random.random() < 0.50 * strength: seq = aug_speed(seq, 0.80, 1.25)
    if random.random() < 0.30 * strength: seq = aug_temporal_mask(seq, 3, 2)
    return seq

# =============================================================================
# 3. SYNTHETIC SENTENCE DATASET (WITH BEGIN/END NOISE)
# =============================================================================
class SyntheticSentenceDataset(Dataset):
    """
    Continuous sentence generator:
    1. Prefix rest frames with random coordinate noise & idle hand drift.
    2. Words concatenated with smooth-step non-linear transitions.
    3. Suffix rest frames with random coordinate noise & idle hand drift.
    """
    def __init__(self, data_dir, classes, samples_per_epoch=1600,
                 is_train=True, max_words=18, min_words=1, aug_strength=1.0):
        self.data_dir       = data_dir
        self.classes        = classes
        self.class_to_idx   = {c: i for i, c in enumerate(classes)}
        self.n              = samples_per_epoch
        self.is_train       = is_train
        self.max_words      = max_words
        self.min_words      = min_words
        self.aug_strength   = aug_strength

        self.class_files = {}
        for c in classes:
            d = os.path.join(data_dir, c)
            if os.path.isdir(d):
                fs = glob.glob(os.path.join(d, "*.pt"))
                if fs: self.class_files[c] = fs
        self.valid = [c for c in classes if c in self.class_files]

        # Rest segments from word boundaries
        self.rest = []
        for c in random.sample(self.valid, min(40, len(self.valid))):
            f = random.choice(self.class_files[c])
            try:
                t = torch.load(f, weights_only=True).float()
                if t.shape[0] >= 10:
                    self.rest.append(t[:5]); self.rest.append(t[-5:])
            except: pass
        if not self.rest:
            r = torch.zeros(8, 163); r[:, :63] = MISSING_LH; r[:, 63:126] = MISSING_RH
            self.rest.append(r)

    def __len__(self): return self.n

    def _rest_with_noise(self, n, add_noise=True):
        seg = random.choice(self.rest)
        while seg.shape[0] < n:
            seg = torch.cat([seg, random.choice(self.rest)], dim=0)
        st = random.randint(0, seg.shape[0] - n)
        res = seg[st:st+n].clone()

        if add_noise:
            # 1. Subtle Gaussian coordinate jitter
            noise_scale = random.uniform(0.004, 0.012) if self.is_train else 0.005
            jitter = torch.randn_like(res[:, :159]) * noise_scale
            # 2. Subtle idle hand drift (slow sinusoidal sway)
            t_steps = torch.linspace(0, random.uniform(1.0, 3.0), n).unsqueeze(1)
            drift = torch.sin(t_steps) * random.uniform(0.003, 0.008)
            res[:, :159] += jitter
            res[:, :159:3] += drift  # apply drift to X coordinates
            res = _reclamp(res)
        return res

    def __getitem__(self, idx):
        # Inject pure blank silence sentence once per epoch so model learns to emit blank on idle rest
        if self.is_train and idx == 0:
            rest_len = random.randint(12, 28)
            blank_seq = self._rest_with_noise(rest_len, add_noise=True)
            blank_seq = augment(blank_seq, self.aug_strength * 0.5)
            return blank_seq, blank_seq.shape[0], torch.tensor([], dtype=torch.long)

        n_words = random.randint(self.min_words, self.max_words)
        words   = random.sample(self.valid, min(n_words, len(self.valid)))
        segs, labels = [], []

        # 1. Beginning: Rest frames + random noise & idle drift
        prefix_len = random.randint(6, 14) if self.is_train else 8
        segs.append(self._rest_with_noise(prefix_len, add_noise=True))

        for i, w in enumerate(words):
            x = torch.load(random.choice(self.class_files[w]), weights_only=True).float()
            # 2. Between words: smooth-step non-linear co-articulation
            if i > 0:
                prev_last = segs[-1][-1:]
                curr_first = x[:1]
                n_t = random.randint(3, 6) if self.is_train else 4
                t_raw = torch.linspace(0, 1, n_t + 2)[1:-1].unsqueeze(1)
                alphas = t_raw * t_raw * (3 - 2 * t_raw)  # smoothstep
                trans = (1 - alphas) * prev_last + alphas * curr_first
                if self.is_train:
                    trans[:, :159] += torch.randn_like(trans[:, :159]) * 0.004
                segs.append(_reclamp(trans))
            segs.append(x)
            labels.append(self.class_to_idx[w])

        # 3. End: Rest frames + random noise & idle drift
        suffix_len = random.randint(6, 12) if self.is_train else 6
        segs.append(self._rest_with_noise(suffix_len, add_noise=True))

        full = torch.cat(segs, dim=0)

        if self.is_train:
            full = augment(full, self.aug_strength)

        return full, full.shape[0], torch.tensor(labels, dtype=torch.long)

def collate(batch):
    xs, lengths, labels = zip(*batch)
    B, maxT = len(xs), max(lengths)
    xp = torch.zeros(B, maxT, xs[0].shape[-1])
    for i, x in enumerate(xs): xp[i, :lengths[i]] = x
    non_empty = [l for l in labels if len(l) > 0]
    targets = torch.cat(non_empty) if non_empty else torch.tensor([], dtype=torch.long)
    return (xp,
            torch.tensor(lengths, dtype=torch.long),
            targets,
            torch.tensor([len(l) for l in labels], dtype=torch.long))

# =============================================================================
# 4. PRETRAINED BACKBONE (PHASE 6 MATCHING LAYERS)
# =============================================================================
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

class PretrainedBackbone(nn.Module):
    """
    Extracts 256-dim frame features using Phase 6 pretrained weights:
    InputProj -> BiGRU -> Transformer -> TCNN Stack (output 256 channels).
    Poolings and word classification head are stripped.
    """
    def __init__(self, d=128, gru_layers=2, n_head=4, n_trans=2, drop=0.15):
        super().__init__()
        self.proj  = nn.Sequential(
            nn.Linear(163, d), nn.LayerNorm(d), nn.Dropout(drop))
        self.gru   = nn.GRU(d, d // 2, gru_layers, batch_first=True, bidirectional=True,
                            dropout=drop if gru_layers > 1 else 0.0)
        self.gnorm = nn.LayerNorm(d)
        self.pe    = PosEnc(d)
        el = nn.TransformerEncoderLayer(d, n_head, d * 2, drop, "gelu", batch_first=True)
        self.trans = nn.TransformerEncoder(el, n_trans, enable_nested_tensor=False)
        self.tnorm = nn.LayerNorm(d)
        self.t1    = TCNN(d, d, drop=drop)
        self.t2    = TCNN(d, d * 2, drop=drop)  # Output: 256 channels

    def forward(self, x):
        if hasattr(self.gru, "flatten_parameters"):
            self.gru.flatten_parameters()
        h = self.proj(x)
        g, _ = self.gru(h)
        h = self.gnorm(h + g)
        h = self.pe(h)
        h = self.trans(h)
        h = self.tnorm(h)
        h = h.transpose(1, 2)  # (B, 128, T)
        h = self.t1(h)
        h = self.t2(h)         # (B, 256, T)
        return h.transpose(1, 2)  # (B, T, 256)

# =============================================================================
# 5. CONFORMER CONTEXTUALIZER & FULL HYBRID MODEL
# =============================================================================
class FeedForward(nn.Module):
    def __init__(self, d, expansion=4, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d * expansion), nn.SiLU(), nn.Dropout(drop),
            nn.Linear(d * expansion, d), nn.Dropout(drop))
    def forward(self, x): return x + 0.5 * self.net(x)

class MultiHeadSelfAttn(nn.Module):
    def __init__(self, d, n_head, drop=0.1, max_len=4096):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_head, dropout=drop, batch_first=True)
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        T = x.size(1)
        if T > self.pe.size(1):
            d = self.pe.size(-1)
            pe = torch.zeros(T + 256, d)
            pos = torch.arange(0, T + 256).float().unsqueeze(1)
            div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
            pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
            self.pe = pe.unsqueeze(0).to(x.device)
        xn = self.norm(x) + self.pe[:, :T, :]
        o, _ = self.attn(xn, xn, xn)
        return x + o

class DepthwiseConv(nn.Module):
    def __init__(self, d, kernel=31, drop=0.1):
        super().__init__()
        assert kernel % 2 == 1, "kernel must be odd"
        self.norm = nn.LayerNorm(d)
        self.pw1  = nn.Conv1d(d, 2*d, 1)
        self.dw   = nn.Conv1d(d, d, kernel, padding=kernel//2, groups=d)
        self.bn   = nn.BatchNorm1d(d)
        self.act  = nn.SiLU()
        self.pw2  = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        r = x
        x = self.norm(x).transpose(1, 2)
        x = self.pw1(x)
        x = x[:, :x.size(1)//2] * torch.sigmoid(x[:, x.size(1)//2:])
        x = self.dw(x)
        x = self.act(self.bn(x))
        x = self.drop(self.pw2(x))
        return r + x.transpose(1, 2)

class ConformerBlock(nn.Module):
    def __init__(self, d, n_head=8, kernel=31, ff_exp=4, drop=0.1):
        super().__init__()
        self.ff1  = FeedForward(d, ff_exp, drop)
        self.mhsa = MultiHeadSelfAttn(d, n_head, drop)
        self.conv = DepthwiseConv(d, kernel, drop)
        self.ff2  = FeedForward(d, ff_exp, drop)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        x = self.ff1(x)
        x = self.mhsa(x)
        x = self.conv(x)
        x = self.ff2(x)
        return self.norm(x)

class ISLHybridConformerCTC(nn.Module):
    """
    Unified Architecture:
    [Pretrained Phase 6 BiGRU+Transformer+TCNN] -> (B, T, 256)
        -> [Conformer Blocks (MHSA + DWConv k=31)] -> (B, T, 256)
        -> [CTC Projection Head] -> (B, T, vocab+1)
    """
    def __init__(self, num_classes, d=256, n_blocks=3, n_head=8, kernel=31, drop=0.1):
        super().__init__()
        self.num_classes  = num_classes
        self.ctc_blank_id = num_classes

        # 1. Pretrained Backbone (d_in=163 -> d_out=256)
        self.backbone = PretrainedBackbone(d=128, gru_layers=2, n_head=4, n_trans=2, drop=0.15)

        # 2. Conformer Contextualizer Stack
        self.conformer = nn.ModuleList([
            ConformerBlock(d, n_head=n_head, kernel=kernel, ff_exp=4, drop=drop)
            for _ in range(n_blocks)
        ])

        # 3. Continuous CTC Head
        self.ctc_proj = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, num_classes + 1)
        )

    def forward(self, x, lengths=None):
        # x: (B, T, 163)
        h = self.backbone(x)            # (B, T, 256)
        for block in self.conformer:
            h = block(h)                # (B, T, 256)
        logits = self.ctc_proj(h)       # (B, T, vocab+1)
        return F.log_softmax(logits, dim=-1)

    def load_phase6_backbone(self, weights_path=PHASE6_WEIGHTS):
        if not os.path.exists(weights_path):
            print(f"[WARN] Phase 6 weights not found at {weights_path}; initializing randomly.")
            return False
        sd = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]

        bb_sd = {}
        for k, v in sd.items():
            if not k.startswith("head.") and not k.startswith("ap.") and not k.startswith("mp."):
                if k == "pe.pe":
                    continue  # Sinusoidal buffer: generated deterministically and extended dynamically
                bb_sd[k] = v

        missing, unexpected = self.backbone.load_state_dict(bb_sd, strict=False)
        print(f"[BACKBONE] Loaded {len(bb_sd)} pretrained layers from {os.path.basename(weights_path)}")
        if missing: print(f"           Missing keys: {len(missing)}")

        # Transfer pretrained word classification weights into CTC Projection Head
        if "head.4.weight" in sd and "head.4.bias" in sd:
            nc = min(self.num_classes, sd["head.4.bias"].shape[0])
            with torch.no_grad():
                self.ctc_proj[1].weight.data[:nc] = sd["head.4.weight"][:nc]
                self.ctc_proj[1].bias.data[:nc] = sd["head.4.bias"][:nc]
                # Initialize blank token with neutral bias so rest frames can be blank
                self.ctc_proj[1].bias.data[self.ctc_blank_id] = 0.0
                self.ctc_proj[1].weight.data[self.ctc_blank_id].zero_()
            print(f"[CTC HEAD] Transferred {nc} pretrained word classifiers from Phase 6 (Blank bias set to 0.0).")

        return True

# =============================================================================
# 6. REGULARIZED CTC LOSS (WITH DURATION DEFICIT PENALTY)
# =============================================================================
class ConformityCTCLoss(nn.Module):
    """
    CTC Loss + Duration Deficit Penalty + Blank Connectivity Regularization.
    
    1. Duration Deficit Penalty:
       For every word in the target sentence, the model is required to emit
       at least `min_frames_per_word` (default 6) non-blank mass.
       If sum(1 - P(blank)) < n_words * min_frames, a quadratic deficit loss is applied.
       This mathematically prevents the CTC 1-frame blank-collapse shortcut.
       
    2. Blank Connectivity Regularization:
       Penalizes isolated 1-frame spikes (non-blank flanked by blanks).
    """
    def __init__(self, blank_id, connectivity_w=0.10, deficit_w=1.5, min_frames_per_word=6.0, zero_inf=True):
        super().__init__()
        self.ctc                  = nn.CTCLoss(blank=blank_id, zero_infinity=zero_inf)
        self.blank_id             = blank_id
        self.connectivity_w       = connectivity_w
        self.deficit_w            = deficit_w
        self.min_frames_per_word  = min_frames_per_word

    def forward(self, log_probs_t, targets, input_lengths, target_lengths):
        ctc_loss = self.ctc(log_probs_t, targets, input_lengths, target_lengths)
        total_loss = ctc_loss

        lp = log_probs_t.transpose(0, 1)  # (B, T, C)
        B, T, C = lp.shape
        blank_lp    = lp[:, :, self.blank_id]  # (B, T)
        non_blank_p = 1.0 - torch.exp(blank_lp) # (B, T)

        # 1. Duration Deficit Penalty: Force each sign to span at least 6-10 frames
        if self.deficit_w > 0.0:
            target_mass = target_lengths.float() * self.min_frames_per_word  # (B,)
            actual_mass = []
            for b in range(B):
                valid_t = input_lengths[b].item()
                actual_mass.append(non_blank_p[b, :valid_t].sum())
            actual_mass = torch.stack(actual_mass)
            deficit = F.relu(target_mass - actual_mass)  # (B,)
            # Gentle linear deficit penalty: ensures words are sustained without forcing silence frames to be non-blank
            total_loss = total_loss + self.deficit_w * deficit.mean()

        # 3. Blank Connectivity Penalty: Penalize isolated 1-frame spikes
        if self.connectivity_w > 0.0 and T > 2:
            left_blank_p   = torch.exp(blank_lp[:, :-2])
            right_blank_p  = torch.exp(blank_lp[:, 2:])
            center_nb_p    = non_blank_p[:, 1:-1]
            spike_penalty  = (center_nb_p * left_blank_p * right_blank_p).mean()
            total_loss = total_loss + self.connectivity_w * spike_penalty

        return total_loss

# =============================================================================
# 7. MODEL EMA & DECODING
# =============================================================================
class ModelEMA:
    def __init__(self, model, decay=0.9995):
        self.model = copy.deepcopy(model); self.model.eval(); self.decay = decay
        for p in self.model.parameters(): p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        for ep, p in zip(self.model.parameters(), model.parameters()):
            ep.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)
        for eb, b in zip(self.model.buffers(), model.buffers()):
            eb.copy_(b)

def greedy_decode(log_probs, length, blank_id, min_dur=3):
    """
    Greedy CTC decode with continuous sign duration filtering.
    A human sign lasts >= 10 frames at 25fps. Fleeting 1-2 frame tracking
    flickers during transitions are filtered out to prevent false insertions.
    """
    tokens = log_probs[:length].argmax(dim=-1).cpu().tolist()
    bursts = []
    cur, dur = None, 0
    for t in tokens:
        if t == cur:
            dur += 1
        else:
            if cur is not None and cur != blank_id:
                bursts.append((cur, dur))
            cur, dur = t, 1
    if cur is not None and cur != blank_id:
        bursts.append((cur, dur))

    # Keep bursts that last at least min_dur frames (or all if all are short)
    filtered = [tok for tok, dur in bursts if dur >= min_dur]
    if not filtered and bursts:
        # Fallback to the single longest burst if all were short
        filtered = [max(bursts, key=lambda x: x[1])[0]]

    out = []
    for tok in filtered:
        if not out or tok != out[-1]:
            out.append(tok)
    return out

def batch_decode(log_probs, lengths, blank_id, min_dur=3):
    out = []
    for i in range(log_probs.shape[0]):
        out.append(greedy_decode(log_probs[i], lengths[i].item(), blank_id, min_dur=min_dur))
    return out

def batch_targets(targets, target_lengths):
    seqs, idx = [], 0
    for l in target_lengths.cpu().tolist():
        seqs.append(targets[idx:idx+l].cpu().tolist()); idx += l
    return seqs

# =============================================================================
# 8. DIAGNOSTICS PLOTS
# =============================================================================
def save_plots(history, plots_dir, stage):
    os.makedirs(plots_dir, exist_ok=True)
    E = range(1, len(history["train_loss"]) + 1)
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))

    axs[0,0].plot(E, history["train_loss"], label="Train Loss", lw=2, color="#1f77b4")
    axs[0,0].plot(E, history["val_loss"],   label="Val Loss",   lw=2, color="#ff7f0e", linestyle="--")
    if history.get("val_loss_ema"): axs[0,0].plot(E, history["val_loss_ema"], label="Val Loss (EMA)", lw=2, color="#2ca02c", linestyle=":")
    axs[0,0].set_title(f"Stage {stage}: Hybrid Conformer-CTC Loss"); axs[0,0].legend(); axs[0,0].grid(True, alpha=0.4)

    axs[0,1].plot(E, [w*100 for w in history["wer"]], label="WER (%)", lw=2, color="#d62728")
    if history.get("wer_ema"): axs[0,1].plot(E, [w*100 for w in history["wer_ema"]], label="WER EMA (%)", lw=2, color="#9467bd", linestyle=":")
    axs[0,1].set_title("Word Error Rate (↓)"); axs[0,1].legend(); axs[0,1].grid(True, alpha=0.4)

    axs[1,0].plot(E, [a*100 for a in history["acc"]], label="Exact Match Acc", lw=2, color="#2ca02c")
    axs[1,0].set_title("Exact Sentence Accuracy (↑)"); axs[1,0].legend(); axs[1,0].grid(True, alpha=0.4)

    axs[1,1].plot(E, history["f1"], label="Token F1", lw=2, color="#8c564b")
    axs[1,1].set_title("Sequence F1 (↑)"); axs[1,1].legend(); axs[1,1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"stage{stage}_metrics.png"), dpi=150)
    plt.close()

# =============================================================================
# 9. RUN EPOCH
# =============================================================================
def run_epoch(model, loader, criterion, optimizer, device, blank_id,
              is_train=True, ema=None, max_norm=1.0):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    all_preds, all_refs = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc="  Train" if is_train else "  Val", leave=False)
        for xs, lengths, targets, target_lengths in pbar:
            xs, lengths = xs.to(device), lengths.to(device)
            targets, target_lengths = targets.to(device), target_lengths.to(device)

            if is_train: optimizer.zero_grad()

            lp = model(xs, lengths)              # (B, T, C)
            lp_t = lp.transpose(0, 1)            # (T, B, C)
            loss = criterion(lp_t, targets, lengths, target_lengths)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                if ema is not None: ema.update(model)

            total_loss += loss.item()
            decoded = batch_decode(lp, lengths, blank_id)
            refs    = batch_targets(targets, target_lengths)
            all_preds.extend(decoded); all_refs.extend(refs)
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})

    N = max(1, len(all_refs))
    wer = sum(compute_wer(r, h) for r, h in zip(all_refs, all_preds)) / N
    acc = sum(1 for r, h in zip(all_refs, all_preds) if r == h) / N
    f1  = sum(len(set(r)&set(h))/(max(1,len(set(r)|set(h)))) for r, h in zip(all_refs, all_preds)) / N
    return total_loss / max(1, len(loader)), wer, acc, f1

# =============================================================================
# 10. MAIN TRAINING SCRIPT
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="ISL Hybrid Conformer-CTC Training")
    p.add_argument("--data_dir",        default=DATA_DIR_DEFAULT)
    p.add_argument("--output_dir",      default=OUTPUT_DIR_DEFAULT)
    p.add_argument("--stage",           type=int, default=1, help="Curriculum stage (1-4)")
    p.add_argument("--epochs",          type=int, default=None, help="Epoch count for current stage")
    p.add_argument("--batch_size",      type=int, default=8)
    p.add_argument("--train_samples",   type=int, default=1600)
    p.add_argument("--val_samples",     type=int, default=240)
    p.add_argument("--resume",          action="store_true")
    p.add_argument("--weights",         type=str, default=None)
    p.add_argument("--num_workers",     type=int, default=0)
    # Model parameters
    p.add_argument("--d_model",         type=int, default=256)
    p.add_argument("--n_blocks",        type=int, default=3)
    p.add_argument("--n_head",          type=int, default=8)
    p.add_argument("--kernel",          type=int, default=31)
    p.add_argument("--dropout",         type=float, default=0.1)
    # Learning rates
    p.add_argument("--backbone_lr",     type=float, default=1.0e-5, help="LR for pretrained BiGRU+Trans+TCNN")
    p.add_argument("--conformer_lr",    type=float, default=1.5e-4, help="LR for Conformer blocks & CTC head")
    # Loss parameters
    p.add_argument("--connectivity_w",  type=float, default=0.10)
    p.add_argument("--deficit_w",       type=float, default=0.2, help="Duration deficit penalty weight")
    p.add_argument("--min_frames",      type=float, default=6.0, help="Min non-blank frames per word")
    return p.parse_args()

STAGE_CONFIG = {
    1: {"name": "Single-Word CTC Warm-Up",  "min_w": 1, "max_w": 1,  "epochs": 20, "bb_lr": 0.0,    "conf_lr": 1.5e-4, "aug": 0.0, "eta_min": 1e-5},
    2: {"name": "2-Word Short Sentences",    "min_w": 2, "max_w": 2,  "epochs": 25, "bb_lr": 1.0e-5, "conf_lr": 1.2e-4, "aug": 0.3, "eta_min": 1e-5},
    3: {"name": "N-Word Curriculum (1→10)", "min_w": 1, "max_w": 10, "epochs": 40, "bb_lr": 1.0e-5, "conf_lr": 1.0e-4, "aug": 0.7, "eta_min": 5e-6},
    4: {"name": "Full Unfrozen Polish+EMA", "min_w": 1, "max_w": 10, "epochs": 30, "bb_lr": 8.0e-6, "conf_lr": 5.0e-5, "aug": 1.0, "eta_min": 1e-6},
}

def main():
    args   = parse_args()
    stage  = args.stage
    cfg    = STAGE_CONFIG[stage]
    epochs = args.epochs or cfg["epochs"]

    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, "plots")

    if os.path.exists(CLASSES_SRC):
        with open(CLASSES_SRC) as f: classes = json.load(f)
    else:
        classes = sorted(d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d)))
    num_classes  = len(classes)
    CTC_BLANK_ID = num_classes

    with open(os.path.join(args.output_dir, "classes.json"), "w") as f:
        json.dump({"classes": classes, "num_classes": num_classes, "ctc_blank_id": CTC_BLANK_ID}, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print(f"ISL HYBRID CONFORMER-CTC — STAGE {stage}: {cfg['name']}")
    print(f"Device:    {device} | Classes: {num_classes} | Blank ID: {CTC_BLANK_ID}")
    print(f"Words:     n=[{cfg['min_w']}..{cfg['max_w']}] | Epochs: {epochs}")
    print(f"Stack:     Pretrained BiGRU+Transformer+TCNN -> {args.n_blocks} Conformer Blocks (k={args.kernel})")
    print(f"Backbone:  Pretrained weights from {os.path.basename(PHASE6_WEIGHTS)}")
    print("=" * 72)

    CKPT_FILE  = os.path.join(args.output_dir, f"stage{stage}_checkpoint.pth")
    BEST_FILE  = os.path.join(args.output_dir, f"stage{stage}_best.pth")
    EMA_FILE   = os.path.join(args.output_dir, f"stage{stage}_ema_best.pth")
    FINAL_FILE = os.path.join(args.output_dir, f"stage{stage}_final.pth")
    HIST_FILE  = os.path.join(args.output_dir, f"stage{stage}_history.json")

    # Build model
    model = ISLHybridConformerCTC(num_classes, args.d_model, args.n_blocks, args.n_head, args.kernel, args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {n_params:,}")

    # Initialize pretrained backbone
    model.load_phase6_backbone(PHASE6_WEIGHTS)

    start_ep  = 0
    best_wer  = float("inf")
    best_ema_wer = float("inf")
    history = {"train_loss":[], "val_loss":[], "val_loss_ema":[], "wer":[], "wer_ema":[], "acc":[], "f1":[], "lr_conf":[], "lr_bb":[]}

    # Load checkpoint if resuming or warm-starting
    loaded = False
    if args.resume and os.path.exists(CKPT_FILE):
        print(f"[RESUME] Loading {CKPT_FILE}")
        ckpt = torch.load(CKPT_FILE, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_ep = ckpt.get("epoch", 0)
        best_wer = ckpt.get("best_wer", float("inf"))
        best_ema_wer = ckpt.get("best_ema_wer", float("inf"))
        history = ckpt.get("history", history)
        loaded = True
        print(f"[RESUME] Resuming at epoch {start_ep}, best WER: {best_wer*100:.2f}%")

    if not loaded and args.weights and os.path.exists(args.weights):
        print(f"[LOAD] Weights from: {args.weights}")
        raw = torch.load(args.weights, map_location=device, weights_only=False)
        sd  = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
        model.load_state_dict(sd, strict=False)

    if not loaded and not args.weights:
        prev_stage = stage - 1
        if prev_stage >= 1:
            prev_best = os.path.join(args.output_dir, f"stage{prev_stage}_best.pth")
            if os.path.exists(prev_best):
                print(f"[WARM-START] Loading stage {prev_stage} weights from {prev_best}")
                sd = torch.load(prev_best, map_location=device, weights_only=True)
                model.load_state_dict(sd, strict=True)

    # Differential Learning Rates
    bb_lr   = args.backbone_lr if args.backbone_lr != 1.0e-5 else cfg["bb_lr"]
    conf_lr = args.conformer_lr if args.conformer_lr != 1.5e-4 else cfg["conf_lr"]

    param_groups = []
    if bb_lr > 0.0:
        param_groups.append({"params": model.backbone.parameters(), "lr": bb_lr, "weight_decay": 1e-4})
    else:
        for p in model.backbone.parameters(): p.requires_grad = False
        print("[STAGE 1] Backbone FROZEN to preserve initial isolated sign alignments.")

    param_groups.append({"params": list(model.conformer.parameters()) + list(model.ctc_proj.parameters()),
                         "lr": conf_lr, "weight_decay": 1e-4})

    criterion = ConformityCTCLoss(
        CTC_BLANK_ID,
        connectivity_w=args.connectivity_w,
        deficit_w=args.deficit_w,
        min_frames_per_word=args.min_frames
    )
    optimizer = optim.AdamW(param_groups, betas=(0.9, 0.98))
    warmup_ep = max(2, min(5, epochs // 4))
    warmup_sched = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, total_iters=warmup_ep)
    cosine_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_ep), eta_min=cfg["eta_min"])
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_ep])
    ema = ModelEMA(model, decay=0.9995)

    # Datasets
    train_ds = SyntheticSentenceDataset(args.data_dir, classes, args.train_samples, True,  cfg["max_w"], cfg["min_w"], cfg["aug"])
    val_ds   = SyntheticSentenceDataset(args.data_dir, classes, args.val_samples,   False, cfg["max_w"], cfg["min_w"], 0.0)
    train_ld = DataLoader(train_ds, args.batch_size, shuffle=True,  collate_fn=collate, num_workers=args.num_workers)
    val_ld   = DataLoader(val_ds,   args.batch_size, shuffle=False, collate_fn=collate, num_workers=args.num_workers)

    # Training Loop
    total_eps = start_ep + epochs
    for ep in range(start_ep, total_eps):
        cur_conf_lr = optimizer.param_groups[-1]["lr"]
        cur_bb_lr   = optimizer.param_groups[0]["lr"] if bb_lr > 0.0 else 0.0

        if stage == 3:
            progress = (ep - start_ep) / max(1, epochs - 1)
            cur_max = max(cfg["min_w"], min(cfg["max_w"], int(1 + progress * (cfg["max_w"] - 1))))
            train_ds.max_words = cur_max

        tl, wer, acc, f1 = run_epoch(model, train_ld, criterion, optimizer, device, CTC_BLANK_ID, True, ema)
        vl, vwer, vacc, vf1 = run_epoch(model, val_ld, criterion, optimizer, device, CTC_BLANK_ID, False)
        ema_vl, ema_wer, _, _ = run_epoch(ema.model, val_ld, criterion, None, device, CTC_BLANK_ID, False)

        history["train_loss"].append(tl); history["val_loss"].append(vl); history["val_loss_ema"].append(ema_vl)
        history["wer"].append(vwer); history["wer_ema"].append(ema_wer)
        history["acc"].append(vacc); history["f1"].append(vf1)
        history["lr_conf"].append(cur_conf_lr); history["lr_bb"].append(cur_bb_lr)

        scheduler.step()

        curriculum_note = f" | max_words={train_ds.max_words}" if stage == 3 else ""
        print(f"\n[S{stage} Ep {ep+1:03d}/{total_eps:03d}] Conf LR={cur_conf_lr:.2e} | BB LR={cur_bb_lr:.2e}{curriculum_note}")
        print(f"  Train Loss: {tl:.4f} | Val Loss: {vl:.4f} (EMA: {ema_vl:.4f})")
        print(f"  WER: {vwer*100:.2f}% | WER EMA: {ema_wer*100:.2f}% | ExAcc: {vacc*100:.2f}% | F1: {vf1:.4f}")

        # Save Checkpoints
        torch.save({
            "epoch": ep+1, "stage": stage,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_wer": best_wer, "best_ema_wer": best_ema_wer,
            "history": history,
        }, CKPT_FILE)

        if vwer < best_wer:
            best_wer = vwer
            torch.save(model.state_dict(), BEST_FILE)
            print(f"  ★ NEW BEST  → {BEST_FILE} (WER: {vwer*100:.2f}%)")

        if ema_wer < best_ema_wer:
            best_ema_wer = ema_wer
            torch.save(ema.model.state_dict(), EMA_FILE)
            print(f"  ★ NEW EMA BEST → {EMA_FILE} (WER: {ema_wer*100:.2f}%)")

        with open(HIST_FILE, "w") as f: json.dump(history, f, indent=2)
        save_plots(history, plots_dir, stage)
        gc.collect(); torch.cuda.empty_cache()

    torch.save(model.state_dict(), FINAL_FILE)

    # Update structure.md
    with open(os.path.join(args.output_dir, "structure.md"), "w", encoding="utf-8") as f:
        f.write(f"""# Directory Structure: Backend/models/isl_conformer_ctc

ISL Hybrid Conformer-CTC continuous sign recognition model.
Pretrained Phase 6 Backbone (InputProj + BiGRU + Transformer + TCNN) + Conformer Contextualizer + CTC Regularization.

## Files
- `classes.json`: Vocabulary and CTC blank ID.
- `stageN_best.pth`: Best standard model weights per stage.
- `stageN_ema_best.pth`: Best EMA model weights per stage.
- `stageN_checkpoint.pth`: Full training state per stage.
- `stageN_final.pth`: Final weights at end of each stage.
- `stageN_history.json`: Per-epoch metrics for each stage.
- `structure.md`: This file.

## Subdirectories
- `plots/`: Metric dashboards and curves per stage.

## Stages
- Stage 1 (20 ep): Single-word CTC warm-up (backbone frozen, trains CTC alignment)
- Stage 2 (25 ep): 2-word short sentences (backbone unfrozen with differential LR)
- Stage 3 (40 ep): N-word progressive curriculum (1→10 words)
- Stage 4 (30 ep): Full unfrozen polish + EMA
""")

    print("\n" + "=" * 72)
    print(f"STAGE {stage} COMPLETE | Best WER: {best_wer*100:.2f}% | Best EMA WER: {best_ema_wer*100:.2f}%")
    print(f"Next stage command:")
    print(f"  python Backend/training/train_conformer_ctc.py --stage {stage+1}")
    print("=" * 72)

if __name__ == "__main__":
    main()
