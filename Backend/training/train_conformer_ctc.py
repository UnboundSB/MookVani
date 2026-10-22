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

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from utils.dataloader_word import get_word_classes  # noqa: E402
from utils.checkpoint_utils import extract_state_dict, load_flexible  # noqa: E402
from utils.ctc_decoder import batch_decode_with_lm  # noqa: E402
from utils.word_lm import build_word_lm_from_index  # noqa: E402

# ---------------------------------------------------------------------------
# Canonical layout + the ONE implementation of the pose augmentations.
# ---------------------------------------------------------------------------
# The geometric augmentation (mirror / tilt / bounded left-right shift / speed),
# the missing-modality convention and the cross-corpus harmonisation used to
# live in this file and were copy-pasted into eight other files with three
# different index conventions (two of them buggy).  They now live in
# Backend/utils/pose_augmentation.py, which is the single source of truth.
# The names are re-exported here so every other trainer can keep importing
# them from this module.
from utils.pose_augmentation import (  # noqa: E402
    ARM_HI,
    ARM_LO,
    BODY_DIMS,
    FACE_HI,
    FACE_LO,
    FLAG_HI,
    FLAG_LO,
    LH_HI,
    LH_LO,
    MISSING_LH,
    MISSING_RH,
    NECK_ANCHOR_SENT,
    NECK_ANCHOR_WORD,
    NECK_OFFSET,
    NECK_OFFSET_WORD,
    RH_HI,
    RH_LO,
    augment,
    augment_pose_sequence,
    aug_flip,
    aug_shift,
    aug_speed,
    aug_temporal_mask,
    aug_tilt,
    harmonize_features,
    harmonize_frame,
    reclamp_missing,
    fill_missing_temporal,
)

# Historic alias: the convention is "an undetected modality owns a ZERO block".
_reclamp = reclamp_missing

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
    """Normalise the encoding of an UNDETECTED hand to a single convention: zeros.

    Why zeros and not the canonical resting pose?  The resting pose (-1.5, 3, 0) is
    defined RELATIVE TO THE CLIP'S OWN ANCHOR, and the three corpora use three
    different anchors (the image origin for the word corpus, the nose for the sentence
    corpus).  Writing the absolute (-1.5, 3, 0) into every corpus therefore moves the
    absent hand to a completely different place in each of them - up to four shoulder
    widths away from where the pretraining corpus had it.

    That broke the synthetic stage 1 completely: the frozen Phase 6 backbone was
    pretrained with the absent hand at the origin and suddenly saw it somewhere else,
    so the exact-sentence accuracy stayed at 0% (the original code reached 52% by
    epoch 8 and 74.6% by epoch 20).

    Zeros are the convention of the corpus the backbone was pretrained on, and the
    presence flag already tells the model that the hand is missing, so the geometry of
    an absent hand carries no information anyway.

    This also removes the right-hand sign bug: the old code put the missing RIGHT hand
    at -1.5, i.e. on the LEFT side of the body.
    """
    lh_missing = seq[:, 159] < 0.5
    rh_missing = seq[:, 160] < 0.5
    if lh_missing.any():
        seq[lh_missing, 0:63] = 0.0
    if rh_missing.any():
        seq[rh_missing, 63:126] = 0.0
    return seq

def harmonize_features(seq, zero_face=True, offset=None):
    """Move a landmark clip into the frame the backbone was pretrained in.

    The three tensor corpora in Backend/data were extracted at different times with
    different normalisation conventions, so the same physical point sits in three
    different places.  Measured on the finished tensors, using the neck (the shoulder
    midpoint) as the reference point:

      tensors_include50_word_level  (the backbone pretraining corpus)
          neck at (3.592, 2.769, -1.233)   - the anchor was the image origin
      tensors_isl_csltr_word_level   (current scripts/extraction.py)
          neck at (0.000, 0.000,  0.000)   - the anchor was the neck
      tensors_sentence_level
          neck at (0.011, 0.605, -0.523)   - the anchor was the nose, because the face
                                               was detected in 89% of the frames

    Handing a backbone that was pretrained in the first frame a clip expressed in the
    third one is a severe domain shift: the whole skeleton is displaced by more than
    three shoulder widths, and the 21 face dimensions that are identically zero in the
    pretraining corpus suddenly carry values.  That is enough to destroy the pretrained
    representation on the very first epoch of fine-tuning - the real-sentence run
    collapsed to 100% blank on epoch 1 with its validation loss rising from 5.8 to 10.1
    while the training loss was still falling.

    The correction is a single constant translation per corpus, because the crop
    geometry is not recoverable from the normalised landmarks.  The corpus means are
    matched exactly by translating the continuous corpus by NECK_OFFSET; the pretraining
    corpus needs no shift at all, so the Phase 6 weights stay valid (verified: 62.0% ->
    60.7% word accuracy, the small drop coming from the missing-modality harmonisation).

    The face block is cleared because it is identically zero in the pretraining corpus;
    leaving it populated would make those 21 dimensions mean something else during
    fine-tuning than during pretraining.

    Order matters: call this BEFORE _reclamp(), so that the canonical resting pose of an
    undetected hand is written in the final frame.
    """
    if seq.shape[-1] != 163:
        return seq
    if offset is None:
        offset = NECK_OFFSET
    seq = seq.contiguous().clone()
    if any(abs(float(v)) > 1e-9 for v in offset):
        lm = seq[:, :159].reshape(-1, 53, 3)
        off = seq.new_tensor(offset).view(1, 1, 3)
        seq[:, :159] = (lm + off).reshape(-1, 159)
    if zero_face:
        # The face block is identically zero in the pretraining corpus.
        seq[:, 138:159] = 0.0
        seq[:, 162] = 0.0
    return seq


def match_pretraining_domain(seq, target_wrist=2.96, target_hand=0.463):
    """Rewrite a clip so it looks like the ISOLATED-WORD corpus the backbone was
    pretrained on.

    Measured domain gap (in shoulder widths):

        include50_word  (pretraining)  hand extent 0.463   wrist-to-neck 2.961
        sentence_level  (fine-tuning)  hand extent 0.318   wrist-to-neck 0.935

    In the isolated-word videos the sign is performed with the arm EXTENDED (the wrist
    is three times further from the neck) and the hand is 1.46x bigger relative to the
    shoulders (a tighter crop).  In the continuous videos the arm is relaxed and the
    shot is wide.  The backbone therefore reads the two domains as completely different
    things.

    This rotates each hand about its elbow until the wrist-to-neck distance matches the
    pretraining corpus, and scales the hand about the wrist until its extent matches.
    The rotation is analytic (solve for the angle that puts the wrist on the circle of
    radius |wrist - target| around the neck), so the result is a clean rigid rotation of
    the hand - no landmark is distorted.
    """
    if seq.shape[-1] != 163:
        return seq
    seq = seq.contiguous().clone()
    T = seq.shape[0]
    lm = seq[:, :159].reshape(T, 53, 3)
    neck = (lm[:, 42] + lm[:, 43]) / 2.0

    for wrist_i, elbow_i, lo, hi in ((0, 44, 0, 21), (21, 45, 21, 42)):
        for t in range(T):
            w = lm[t, wrist_i]; e = lm[t, elbow_i]; nk = neck[t]
            v = w - e; u = nk - e
            lv, lu = float(np.linalg.norm(v)), float(np.linalg.norm(u))
            if lv < 1e-6 or lu < 1e-6:
                continue
            cos = (lv * lv + lu * lu - target_wrist ** 2) / (2 * lv * lu)
            if not (-1.0 <= cos <= 1.0):
                continue
            phi = math.atan2(float(u[1] * v[0] - u[0] * v[1]), float(u[0] * v[0] + u[1] * v[1]))
            theta = math.acos(cos) - phi
            c, s = math.cos(theta), math.sin(theta)
            d = lm[t, lo:hi, :2] - e[:2]
            rot = torch.stack([d[:, 0] * c - d[:, 1] * s, d[:, 0] * s + d[:, 1] * c], dim=1)
            lm[t, lo:hi, :2] = rot + e[:2]

    # scale the hand about its own wrist
    for lo, hi in ((0, 21), (21, 42)):
        blk = lm[:, lo:hi]
        w = blk[:, 0:1]
        ext = torch.linalg.norm(blk - w, dim=2).mean(1)
        f = torch.where(ext > 1e-6, torch.full_like(ext, target_hand) / ext.clamp(min=1e-6),
                        torch.ones_like(ext))
        lm[:, lo:hi] = w + (blk - w) * f[:, None, None]

    seq[:, :159] = lm.reshape(T, 159)
    return seq


def harmonize_frame(feat):
    """Align a SINGLE 163-dim frame to the reference frame of the backbone.

    Same operation as harmonize_features() + _reclamp(), but for the live inference
    path, which produces the features frame by frame.  Without it the live stream is
    expressed in a different geometric frame than the one the model was trained on and
    the recogniser returns nothing.
    """
    f = np.asarray(feat, dtype=np.float32).copy()
    if f.shape[-1] != 163:
        return f
    # The 159 landmark dims are 53 landmarks x 3 coordinates, so the offset is repeated.
    f[:159] += np.tile(np.asarray(NECK_OFFSET, dtype=np.float32), 53)
    f[138:159] = 0.0
    f[162] = 0.0
    if f[159] < 0.5:
        f[0:63] = 0.0
    if f[160] < 0.5:
        f[63:126] = 0.0
    return f


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
    c = seq[:, :BODY_DIMS].view(T, -1, 3)
    c[:, :, 0] += random.uniform(-dx, dx)
    c[:, :, 1] += random.uniform(-dy, dy)
    seq[:, :BODY_DIMS] = c.view(T, BODY_DIMS)
    return _reclamp(seq)

def aug_tilt(seq, deg=10.0):
    seq = seq.clone(); T = seq.shape[0]
    a = math.radians(random.uniform(-deg, deg)); ca, sa = math.cos(a), math.sin(a)
    R = torch.tensor([[ca,-sa],[sa,ca]], dtype=torch.float32)
    c = seq[:, :BODY_DIMS].view(T, -1, 3)
    c[:, :, :2] = (c[:, :, :2].reshape(-1, 2) @ R.T).view(T, -1, 2)
    seq[:, :BODY_DIMS] = c.view(T, BODY_DIMS)
    return _reclamp(seq)

def aug_speed(seq, lo=0.80, hi=1.25):
    """Speed up (lo < 1.0) or slow down (hi > 1.0) sign execution smoothly.

    Only the 159 landmark channels are resampled.  The 4 presence flags are discrete
    and are resampled with nearest-neighbour and re-thresholded: the previous version
    linearly interpolated all 163 channels, which produced fractional flags (0.5, 0.3,
    ...) and silently corrupted the _reclamp() bookkeeping of every augmented clip.
    """
    T, D = seq.shape
    factor = random.uniform(lo, hi)
    new_T = max(8, int(round(T * factor)))
    if new_T == T: return seq
    lm = F.interpolate(seq[:, :159].T.unsqueeze(0), size=new_T,
                       mode="linear", align_corners=True).squeeze(0).T
    fl = F.interpolate(seq[:, 159:163].T.unsqueeze(0), size=new_T,
                       mode="nearest").squeeze(0).T
    return torch.cat([lm, (fl > 0.5).to(seq.dtype)], dim=1)

def aug_temporal_mask(seq, max_mask=4, n_masks=2):
    """SpecAugment-style temporal cutout over the landmark channels.

    The presence flags are cleared for the masked frames so the model never sees the
    inconsistent state "modality present but geometry exactly zero".  That state does
    not occur in the extracted data (verified: 0 such frames across both corpora), so
    creating it in training only adds label noise.
    """
    seq = seq.clone(); T = seq.shape[0]
    for _ in range(random.randint(1, n_masks)):
        ml = random.randint(1, max_mask)
        st = random.randint(0, max(0, T - ml))
        seq[st:st+ml, :159] = 0.0
        seq[st:st+ml, 159:163] = 0.0
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
                 is_train=True, max_words=18, min_words=1, aug_strength=1.0,
                 max_per_class=None, fill_missing="none"):
        self.data_dir       = data_dir
        self.classes        = classes
        self.class_to_idx   = {c: i for i, c in enumerate(classes)}
        self.n              = samples_per_epoch
        self.is_train       = is_train
        self.max_words      = max_words
        self.min_words      = min_words
        self.aug_strength   = aug_strength
        self.fill_missing   = fill_missing

        self.class_files = {}
        for c in classes:
            d = os.path.join(data_dir, c)
            if os.path.isdir(d):
                fs = sorted(glob.glob(os.path.join(d, "*.pt")))
                if max_per_class:
                    fs = fs[:max_per_class]
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
        res = harmonize_features(res, offset=NECK_OFFSET_WORD)
        res = _reclamp(res)

        if add_noise:
            # 1. Subtle Gaussian coordinate jitter (body landmarks only - the face block
            #    stays exactly zero, see BODY_DIMS)
            noise_scale = random.uniform(0.004, 0.012) if self.is_train else 0.005
            jitter = torch.randn_like(res[:, :BODY_DIMS]) * noise_scale
            # 2. Subtle idle hand drift (slow sinusoidal sway)
            t_steps = torch.linspace(0, random.uniform(1.0, 3.0), n).unsqueeze(1)
            drift = torch.sin(t_steps) * random.uniform(0.003, 0.008)
            res[:, :BODY_DIMS] += jitter
            res[:, 0:BODY_DIMS:3] += drift  # X coordinates of the body landmarks
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
            # The word corpus is already in the reference frame the backbone was
            # pretrained in, so only the face block and the missing-modality encoding
            # have to be harmonised.
            x = harmonize_features(x, offset=NECK_OFFSET_WORD)
            x = _reclamp(x.contiguous())
            if self.fill_missing and self.fill_missing != "none":
                x = fill_missing_temporal(x, mode=self.fill_missing)
            # 2. Between words: smooth-step non-linear co-articulation
            if i > 0:
                prev_last = segs[-1][-1:]
                curr_first = x[:1]
                n_t = random.randint(3, 6) if self.is_train else 4
                t_raw = torch.linspace(0, 1, n_t + 2)[1:-1].unsqueeze(1)
                alphas = t_raw * t_raw * (3 - 2 * t_raw)  # smoothstep
                trans = (1 - alphas) * prev_last + alphas * curr_first
                if self.is_train:
                    # Body landmarks only - the face block must stay exactly zero.
                    trans[:, :BODY_DIMS] += torch.randn_like(trans[:, :BODY_DIMS]) * 0.004
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

class MixedContinuousDataset(Dataset):
    """
    Mixes three sources of continuous-recognition training data so that the sign
    inventory, the CTC alignment and the real domain are learned together:

      0. WORD      - one isolated sign clip (262 classes, the backbone's own domain)
                     target = [class]
      1. SYNTHETIC - N word clips concatenated with context frames
                     target = [c1 ... cN]
      2. REAL      - a continuous human sentence
                     target = the in-scope words in signing order

    The three sources have complementary weaknesses:
      - the word clips have the best class coverage (4,273 clips over 262 classes) but
        no temporal context;
      - the synthetic sentences have context but are built from the SAME tight-crop,
        extended-arm domain as the word clips;
      - the real sentences have the right domain but only ~1.4 labelled words each and a
        class distribution dominated by pronouns (56% are i/you/he/she).

    Mixing them lets the strong signal (the word inventory, which the backbone already
    classifies at 68%) support the weak one (the real continuous domain).  The source is
    drawn at random per sample from `weights`.

    `class_cap` trims the classes that have more clips than the median in the synthetic
    source, so the synthetic part contributes an approximately balanced class
    distribution and does not reinforce the pronoun bias of the real part.
    """

    def __init__(self, word_dir, classes, samples_per_epoch=1600, is_train=True,
                 weights=(0.40, 0.35, 0.25), max_words=4, min_words=1, aug_strength=1.0,
                 real_items=None, real_dir=None, harmonize=True, domain_match=False,
                 class_cap=None, max_len=300, rest_from_real=True, real_word_cap=0):
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.n = samples_per_epoch
        self.is_train = is_train
        self.weights = [w / sum(weights) for w in weights]
        self.max_words = max_words
        self.min_words = min_words
        self.aug_strength = aug_strength
        self.harmonize = harmonize
        self.domain_match = domain_match
        self.max_len = max_len
        self._cache = {}

        # ---- source 0/1: the isolated-word corpus --------------------------
        self.class_files = {}
        for c in classes:
            d = os.path.join(word_dir, c)
            if os.path.isdir(d):
                fs = sorted(glob.glob(os.path.join(d, "*.pt")))
                if fs:
                    self.class_files[c] = fs
        self.valid = [c for c in classes if c in self.class_files]

        # class cap: trim the classes that have more clips than the median
        counts = sorted(len(v) for v in self.class_files.values())
        median = counts[len(counts) // 2] if counts else 1
        self.class_cap = median if not class_cap else int(class_cap)
        self.syn_files = {}
        for c, fs in self.class_files.items():
            if len(fs) > self.class_cap:
                fs = random.sample(fs, self.class_cap)
            self.syn_files[c] = fs
        self.syn_valid = [c for c in self.valid if self.syn_files[c]]
        if not self.syn_valid:            # the cap must never empty the class list
            self.syn_valid = list(self.valid)
            self.syn_files = {c: self.class_files[c] for c in self.valid}

        # ---- source 2: the continuous corpus -------------------------------
        self.real_items = list(real_items or [])
        self.real_dir = real_dir
        self.real_word_cap = int(real_word_cap or 0)
        self.real_pool = self._build_real_pool(self.real_items, self.real_word_cap)
        self.real_counts = self._word_counts(self.real_items)

        # ---- context ("rest") frames ----------------------------------------
        # Prefer REAL continuous frames as the context: the model then learns to find a
        # short gesture inside a real continuous clip, which is the actual task.
        self.rest = []
        if rest_from_real and self.real_items:
            for it in random.sample(self.real_items, min(120, len(self.real_items))):
                try:
                    t = torch.load(it["tensor_path"], weights_only=True).float()
                    if t.shape[0] >= 10:
                        self.rest.append(("real", t[:6]))
                        self.rest.append(("real", t[-6:]))
                except Exception:
                    pass
        for c in random.sample(self.valid, min(40, len(self.valid))):
            f = random.choice(self.class_files[c])
            try:
                t = torch.load(f, weights_only=True).float()
                if t.shape[0] >= 10:
                    self.rest.append(("word", t[:5]))
                    self.rest.append(("word", t[-5:]))
            except Exception:
                pass
        if not self.rest:
            r = torch.zeros(8, 163)
            self.rest.append(("word", r))

    def __len__(self):
        return self.n

    # -- frequency capping of the continuous corpus --------------------------
    @staticmethod
    def _word_counts(items):
        cnt = {}
        for it in items:
            for w in set(it.get('in_scope_indices', [])):
                cnt[w] = cnt.get(w, 0) + 1
        return cnt

    def _build_real_pool(self, items, cap):
        """Re-sample the continuous clips so the over-represented words are seen less.

        The continuous corpus is dominated by a handful of pronouns: in the 1007 training
        clips 'i' appears in 283, 'you' in 271, 'he' in 136 and 'she' in 93 - the top six
        words hold 61% of all 1442 word instances.  A model trained on that learns the
        label prior instead of the signs (measured: it predicted only i/he/you and never
        once a content word - 28.7% word accuracy against a 20.8% prior baseline).

        The clips are RE-SAMPLED rather than dropped: a clip is kept with probability
        min over its in-scope words of cap / count(word).  Dropping them instead would
        delete ~40% of the corpus, because most pronoun clips contain nothing else
        (175 of the 283 'i' clips have 'i' as their ONLY in-scope word).
        """
        pool = []
        for it in items:
            ws = list(it.get('in_scope_indices', []))
            if not ws:
                continue
            p = 1.0
            if cap:
                cnt = self._word_counts(items)
                for w in set(ws):
                    p = min(p, min(1.0, cap / max(1, cnt.get(w, 1))))
            pool.append((it, p))
        return pool

    # -- feature pipeline ---------------------------------------------------
    def _prep(self, seq, is_real):
        if self.harmonize:
            seq = harmonize_features(seq, offset=(NECK_OFFSET if is_real else NECK_OFFSET_WORD))
        if self.domain_match and is_real:
            seq = match_pretraining_domain(seq)
        return _reclamp(seq.contiguous())

    def _load_word(self, cls):
        key = ("w", cls)
        if key not in self._cache:
            f = random.choice(self.class_files[cls])
            self._cache[key] = self._prep(torch.load(f, weights_only=True).float(), False)
        return self._cache[key]

    def _load_real(self, it):
        key = ("r", it["tensor_path"])
        if key not in self._cache:
            s = torch.load(it["tensor_path"], weights_only=True).float()
            if s.shape[0] > self.max_len:
                s = F.interpolate(s.T[None], size=self.max_len, mode="linear",
                                  align_corners=True)[0].T
            self._cache[key] = self._prep(s, True)
        return self._cache[key]

    def _rest_seg(self, n):
        seg = None
        while seg is None or seg.shape[0] < n:
            kind, t = random.choice(self.rest)
            if kind == "real" and self.harmonize:
                t = self._prep(t, True)
            seg = t if seg is None else torch.cat([seg, t], dim=0)
        st = random.randint(0, seg.shape[0] - n)
        res = seg[st:st + n].clone()
        return _reclamp(harmonize_features(res, offset=NECK_OFFSET_WORD).contiguous()) \
            if not self.harmonize else _reclamp(res.contiguous())

    def _with_noise(self, res, n):
        if not self.is_train:
            return res
        noise_scale = random.uniform(0.004, 0.012)
        res = res.clone()
        res[:, :BODY_DIMS] += torch.randn_like(res[:, :BODY_DIMS]) * noise_scale
        ts = torch.linspace(0, random.uniform(1.0, 3.0), n).unsqueeze(1)
        res[:, 0:BODY_DIMS:3] += torch.sin(ts) * random.uniform(0.003, 0.008)
        return _reclamp(res)

    # -- the three sources --------------------------------------------------
    def _word_sample(self):
        cls = random.choice(self.valid)
        x = self._load_word(cls)
        pre = random.randint(4, 10) if self.is_train else 6
        suf = random.randint(4, 10) if self.is_train else 6
        segs = [self._with_noise(self._rest_seg(pre), pre), x,
                self._with_noise(self._rest_seg(suf), suf)]
        full = torch.cat(segs, dim=0)
        if self.is_train:
            full = augment(full, self.aug_strength * 0.6)
        return full, full.shape[0], torch.tensor([self.class_to_idx[cls]], dtype=torch.long)

    def _synthetic_sample(self):
        n_words = random.randint(self.min_words, self.max_words)
        pool = self.syn_valid or self.valid
        words = random.sample(pool, min(n_words, len(pool)))
        segs, labels = [], []
        pre = random.randint(6, 14) if self.is_train else 8
        segs.append(self._with_noise(self._rest_seg(pre), pre))
        for i, w in enumerate(words):
            x = self._load_word(w)
            if i > 0:
                prev_last = segs[-1][-1:]
                curr_first = x[:1]
                n_t = random.randint(3, 6) if self.is_train else 4
                t_raw = torch.linspace(0, 1, n_t + 2)[1:-1].unsqueeze(1)
                alphas = t_raw * t_raw * (3 - 2 * t_raw)
                trans = (1 - alphas) * prev_last + alphas * curr_first
                if self.is_train:
                    trans[:, :BODY_DIMS] += torch.randn_like(trans[:, :BODY_DIMS]) * 0.004
                segs.append(_reclamp(trans.contiguous()))
            segs.append(x)
            labels.append(self.class_to_idx[w])
        suf = random.randint(6, 12) if self.is_train else 6
        segs.append(self._with_noise(self._rest_seg(suf), suf))
        full = torch.cat(segs, dim=0)
        if self.is_train:
            full = augment(full, self.aug_strength)
        return full, full.shape[0], torch.tensor(labels, dtype=torch.long)

    def _real_sample(self):
        # Sample a continuous clip, honouring the per-word frequency cap.
        it = self.real_pool[0][0]
        for _ in range(8):
            cand, p = random.choice(self.real_pool)
            if random.random() <= p:
                it = cand
                break
        s = self._load_real(it)
        labels = it["in_scope_indices"][:18]
        if self.is_train:
            s = augment(s, self.aug_strength)
        return s, s.shape[0], torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        r = random.random()
        if r < self.weights[0]:
            return self._word_sample()
        if r < self.weights[0] + self.weights[1]:
            return self._synthetic_sample()
        return self._real_sample()


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

class MaskedBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d that ignores padded frames.

    Subclassing nn.BatchNorm1d keeps the parameter names identical (`weight`, `bias`,
    `running_mean`, `running_var`, `num_batches_tracked`), so EVERY existing checkpoint
    still loads unchanged and `isinstance(m, _BatchNorm)` checks keep working.

    Why this matters: the sentences in a batch have very different lengths (25 .. 300
    frames for the continuous corpus) and `collate` right-pads them with zeros.  A plain
    BatchNorm1d normalises over the full padded width, so the padded region - which the
    depthwise conv (kernel 31) fills with real activations - poisons the batch statistics
    and the running estimates.  The symptom is a validation loss an order of magnitude
    above the training loss, and a blank-collapsed CTC head, because the evaluation pass
    uses the polluted running statistics.
    """

    def forward(self, x, mask=None):
        # mask: (B, T) float, 1.0 = real frame, 0.0 = padding
        if mask is None or not self.training:
            return super().forward(x)
        B, C, T = x.shape
        m = mask.to(x.dtype).unsqueeze(1)              # (B, 1, T)
        cnt = m.sum(dim=2, keepdim=True).clamp(min=1.0)  # (B, 1, 1)
        mean = (x * m).sum(dim=2, keepdim=True) / cnt
        var = (((x - mean) ** 2) * m).sum(dim=2, keepdim=True) / cnt
        xhat = (x - mean) * torch.rsqrt(var + self.eps)
        out = xhat * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        with torch.no_grad():
            n = cnt.reshape(B)                                   # (B,)
            unbiased = var.squeeze(-1) * (n / (n - 1.0).clamp(min=1.0)).unsqueeze(1)  # (B, C)
            self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean.squeeze(-1).mean(0))
            self.running_var.mul_(1 - self.momentum).add_(self.momentum * unbiased.mean(0))
            self.num_batches_tracked.add_(1)
        return out


class TCNN(nn.Module):
    def __init__(self, ic, oc, k=3, drop=0.1, causal=False):
        super().__init__()
        self.k = k
        self.causal = causal
        self.c1 = nn.Conv1d(ic, oc, k, padding=0)
        self.b1 = MaskedBatchNorm1d(oc)
        self.g1 = nn.GELU()
        self.d1 = nn.Dropout(drop)
        self.c2 = nn.Conv1d(oc, oc, k, padding=0)
        self.b2 = MaskedBatchNorm1d(oc)
        self.g2 = nn.GELU()
        self.d2 = nn.Dropout(drop)
        self.res = nn.Conv1d(ic, oc, 1) if ic != oc else nn.Identity()

    def forward(self, x, mask=None, causal=False):
        causal = causal or self.causal
        r = self.res(x)
        o = self.d1(self.g1(self.b1(self._conv(self.c1, x, causal), mask)))
        o = self.d2(self.g2(self.b2(self._conv(self.c2, o, causal), mask)))
        return o + r

    def _conv(self, conv, x, causal):
        # A left padding of k-1 makes the convolution causal without changing
        # the output length.
        pad = (self.k - 1, 0) if causal else (self.k // 2, self.k // 2)
        return conv(F.pad(x, pad))

class GRUExpander(nn.Module):
    """A recurrent DECODER that expands a reduced sequence back to the frame rate.

    The idea: the strided front-end compresses the timeline (T -> T/s) so the
    self-attention becomes cheap, but a plain interpolation back to T frames
    produces piecewise-linear artefacts that the CTC head reads as
    half-frame motions.  A learned expansion produces a coherent trajectory
    instead.

    The decoder consumes the compressed sequence repeated at the TARGET rate
    (nearest-neighbour alignment on the original timeline) and integrates it
    recurrently, so every output frame carries the temporal context of the
    compressed frames around it.

    It is a decoder, not an encoder: it is unidirectional (the information
    flows forward only) and it is the only recurrent part of the model, which
    is exactly the price of the expansion.
    """

    def __init__(self, d, layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(d, d, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.layers = layers

    def forward(self, x, target_len=None):
        """x: (B, T', d) -> (B, T, d) with T = target_len (or T' when None)."""
        B, Tp, d = x.shape
        if target_len is None or target_len <= Tp:
            # Nothing to expand: (optionally) compress to the target length.
            if target_len is not None and target_len < Tp:
                idx = torch.linspace(0, Tp - 1, target_len, device=x.device).round().long()
                x = x[:, idx]
            out, _ = self.gru(x)
            return out

        idx = torch.linspace(0, Tp - 1, target_len, device=x.device).round().long()
        ctx = x[:, idx]                      # (B, T, d) aligned on the original timeline
        out, _ = self.gru(ctx)
        return out


class PretrainedBackbone(nn.Module):
    """
    Extracts 256-dim frame features using Phase 6 pretrained weights:
    InputProj -> BiGRU -> Transformer -> TCNN Stack (output 256 channels).
    Poolings and word classification head are stripped.

    ``causal=True`` makes the whole stack causal (a unidirectional GRU and a
    causal attention mask), which is what a live, streaming recogniser needs:
    a frame can then be classified without waiting for the end of the clip.
    The offline model stays bidirectional (the default), because a bidirectional
    context is worth a lot of accuracy on a recorded clip.

    ``gru_mode``
        ``"residual"`` (the default, the historical behaviour) - the GRU
        smooths the sequence and its output is ADDED to the input.
        ``"expand"`` - the GRU is a DECODER that expands the (compressed)
        sequence back to the frame rate, and the Transformer receives its
        output directly instead of the projected features.
    """

    def __init__(self, d=128, gru_layers=2, n_head=4, n_trans=2, drop=0.15, causal=False,
                 gru_mode="residual"):
        super().__init__()
        self.causal = causal
        self.gru_mode = gru_mode
        self.proj  = nn.Sequential(
            nn.Linear(163, d), nn.LayerNorm(d), nn.Dropout(drop))
        # "expand"        : the GRU IS the decoder (the pretrained BiGRU is NOT used)
        # "expand_keep"   : the pretrained BiGRU smooths AND a decoder expands
        keep_gru = gru_mode != "expand"
        if keep_gru:
            # A bidirectional GRU with hidden = d/2 emits d channels, a causal
            # (unidirectional) one has to use hidden = d to emit the same width.
            gru_hidden = d if causal else d // 2
            self.gru = nn.GRU(d, gru_hidden, gru_layers, batch_first=True,
                              bidirectional=not causal,
                              dropout=drop if gru_layers > 1 else 0.0)
        else:
            self.gru = None
        if gru_mode != "residual":
            self.expander = GRUExpander(d, layers=1, dropout=drop)
        else:
            self.expander = None
        self.gnorm = nn.LayerNorm(d)
        self.pe    = PosEnc(d)
        el = nn.TransformerEncoderLayer(d, n_head, d * 2, drop, "gelu", batch_first=True)
        self.trans = nn.TransformerEncoder(el, n_trans, enable_nested_tensor=False)
        self.tnorm = nn.LayerNorm(d)
        self.t1    = TCNN(d, d, drop=drop, causal=causal)
        self.t2    = TCNN(d, d * 2, drop=drop, causal=causal)  # Output: 256 channels

    def forward(self, x, mask=None, attn_mask=None, target_len=None, lengths=None):
        if hasattr(self.gru, "flatten_parameters"):
            self.gru.flatten_parameters()
        h = self.proj(x)
        if self.gru_mode == "expand":
            # The GRU is a decoder: it expands the compressed sequence and the
            # Transformer receives its output DIRECTLY.
            h = self.expander(h, target_len)
            h = self.gnorm(h)
            if mask is None and lengths is not None:
                # The mask has to be built at the EXPANDED resolution.
                T = h.size(1)
                ar = torch.arange(T, device=h.device).unsqueeze(0)
                lens = lengths.to(h.device).clamp(min=1, max=T).unsqueeze(1)
                mask = (ar < lens).to(h.dtype)
        else:
            g, _ = self.gru(h)
            h = self.gnorm(h + g)
            if self.gru_mode == "expand_keep":
                # The pretrained BiGRU smooths, and a decoder expands the
                # result: the Transformer receives the expanded tensors and the
                # pretrained temporal model is KEPT.
                h = self.expander(h, target_len)
                if mask is None and lengths is not None:
                    T = h.size(1)
                    ar = torch.arange(T, device=h.device).unsqueeze(0)
                    lens = lengths.to(h.device).clamp(min=1, max=T).unsqueeze(1)
                    mask = (ar < lens).to(h.dtype)
        h = self.pe(h)
        # Padded frames must not be attended to by the valid frames.
        kpm = None if mask is None else (mask < 0.5)
        h = self.trans(h, src_key_padding_mask=kpm, mask=attn_mask)
        h = self.tnorm(h)
        h = h.transpose(1, 2)  # (B, 128, T)
        h = self.t1(h, mask)
        h = self.t2(h, mask)   # (B, 256, T)
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

    def forward(self, x, mask=None, attn_mask=None):
        T = x.size(1)
        if T > self.pe.size(1):
            d = self.pe.size(-1)
            pe = torch.zeros(T + 256, d)
            pos = torch.arange(0, T + 256).float().unsqueeze(1)
            div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
            pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
            self.pe = pe.unsqueeze(0).to(x.device)
        xn = self.norm(x) + self.pe[:, :T, :]
        kpm = None if mask is None else (mask < 0.5)
        o, _ = self.attn(xn, xn, xn, key_padding_mask=kpm, attn_mask=attn_mask)
        return x + o

class DepthwiseConv(nn.Module):
    def __init__(self, d, kernel=31, drop=0.1):
        super().__init__()
        assert kernel % 2 == 1, "kernel must be odd"
        self.kernel = kernel
        self.norm = nn.LayerNorm(d)
        self.pw1  = nn.Conv1d(d, 2*d, 1)
        self.dw   = nn.Conv1d(d, d, kernel, padding=0, groups=d)
        self.bn   = MaskedBatchNorm1d(d)
        self.act  = nn.SiLU()
        self.pw2  = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask=None, causal=False):
        r = x
        x = self.norm(x).transpose(1, 2)
        x = self.pw1(x)
        x = x[:, :x.size(1)//2] * torch.sigmoid(x[:, x.size(1)//2:])
        # The padding decides whether the convolution is causal: with a left
        # padding of kernel-1 an output frame only ever sees the PAST.
        if causal:
            x = F.pad(x, (self.kernel - 1, 0))
        else:
            x = F.pad(x, (self.kernel // 2, self.kernel // 2))
        x = self.dw(x)
        x = self.act(self.bn(x, mask))
        x = self.drop(self.pw2(x))
        return r + x.transpose(1, 2)

class ConformerBlock(nn.Module):
    def __init__(self, d, n_head=8, kernel=31, ff_exp=4, drop=0.1, causal=False):
        super().__init__()
        self.ff1  = FeedForward(d, ff_exp, drop)
        self.mhsa = MultiHeadSelfAttn(d, n_head, drop)
        self.conv = DepthwiseConv(d, kernel, drop)
        self.ff2  = FeedForward(d, ff_exp, drop)
        self.norm = nn.LayerNorm(d)
        self.causal = causal

    def forward(self, x, mask=None, attn_mask=None):
        x = self.ff1(x)
        x = self.mhsa(x, mask, attn_mask)
        x = self.conv(x, mask, causal=self.causal)
        x = self.ff2(x)
        return self.norm(x)


class StridedFrontEnd(nn.Module):
    """The strided convolution stem of a Conformer ASR front-end.

    The backbone used to run its Transformer over EVERY frame of a clip (up to
    300), so the attention was O(T^2) and the model had to learn a
    representation at the full frame rate.  A stack of stride-2 convolutions
    halves (or quarters) the temporal resolution before the self-attention:
    the attention cost drops by 4x (16x) and the sequence becomes 4x (16x)
    cheaper to train.

    The convolution is applied over the raw 163-dim landmark channels, so the
    first layer keeps the geometry of the skeleton intact; only the temporal
    resolution is reduced.
    """

    def __init__(self, in_dim=163, channels=256, stride=2, kernel=3, drop=0.1, causal=False):
        super().__init__()
        assert stride in (1, 2, 4, 8), "the stride must be a power of two"
        layers = []
        c_in = in_dim
        n_conv = {2: 1, 4: 2, 8: 3}[stride]
        for _ in range(n_conv):
            if causal:
                # A LEFT padding of kernel-1 keeps the output length identical
                # while making every output frame depend on the past only.
                layers.append(nn.ConstantPad1d((kernel - 1, 0), 0.0))
                layers.append(nn.Conv1d(c_in, channels, kernel_size=kernel, stride=2, padding=0))
            else:
                layers.append(nn.Conv1d(c_in, channels, kernel_size=kernel, stride=2,
                                        padding=kernel // 2))
            layers += [
                nn.BatchNorm1d(channels),
                nn.GELU(),
                nn.Dropout(drop),
            ]
            c_in = channels
        self.net = nn.Sequential(*layers)
        self.out_dim = c_in
        self.stride = stride

    def forward(self, x):
        # x: (B, T, 163) -> (B, T_out, out_dim)
        h = self.net(x.transpose(1, 2))
        return h.transpose(1, 2)

class ISLHybridConformerCTC(nn.Module):
    """
    Unified Architecture:
    [StridedFrontEnd (optional)] ->
    [Pretrained Phase 6 BiGRU+Transformer+TCNN] -> (B, T', 256)
        -> [Conformer Blocks (MHSA + DWConv k=31)] -> (B, T', 256)
        -> [CTC Projection Head] -> (B, T', vocab+1)

    ``stride``
        The temporal resolution of the front-end.  ``stride=1`` (the default)
        keeps the historical behaviour: one output frame per input frame.
        ``stride=2`` halves the resolution before the self-attention, which
        cuts the attention cost by 4x and the sequence cost by 2x.  The CTC
        head then works on T/2 frames, so the number of frames a sign must
        occupy (``min_frames_per_sign``) and the minimum duration of a decoded
        burst have to be divided by the stride as well.

    ``causal``
        A fully causal model (unidirectional GRU + causal attention mask).
        Required by a streaming recogniser, at the cost of the bidirectional
        context.

    ``gru_expand``
        Where the GRU (as a recurrent decoder) expands the compressed sequence
        back to the frame rate:

        ``"none"``   - the historical behaviour: the GRU smooths the sequence
                      and its output is added to the input.  With ``stride=1``
                      nothing is compressed, so nothing has to be expanded.
        ``"before"`` - the GRU expands BEFORE the self-attention, so the
                      Transformer receives the expanded tensors directly (the
                      attention then runs at the FULL frame rate, i.e. no
                      attention saving - the gain is a coherent trajectory).
        ``"after"``  - the sequence is compressed, the attention runs on the
                      compressed sequence (4x cheaper with stride 2) and the
                      GRU expands the result back to the frame rate before the
                      CTC head.  This is the Conformer-CTC design and it keeps
                      the frame-level CTC alignment.
    """
    def __init__(self, num_classes, d=256, n_blocks=3, n_head=8, kernel=31, drop=0.1,
                 stride=1, causal=False, gru_expand="none", aux_head=False):
        super().__init__()
        self.num_classes  = num_classes
        self.ctc_blank_id = num_classes
        self.stride       = int(stride)
        self.causal       = bool(causal)
        self.gru_expand   = str(gru_expand)

        # 0. Optional strided convolution front-end
        self.front = None
        if self.stride > 1:
            self.front = StridedFrontEnd(163, channels=256, stride=self.stride, drop=drop,
                                         causal=causal)
            in_dim = self.front.out_dim
        else:
            in_dim = 163

        # 1. Pretrained Backbone (d_in -> d_out=256)
        self.backbone = PretrainedBackbone(
            d=128, gru_layers=2, n_head=4, n_trans=2, drop=0.15, causal=causal,
            gru_mode=("expand" if self.gru_expand == "before" else
                      "expand_keep" if self.gru_expand == "before_keep_gru" else
                      "residual"))
        if self.front is not None:
            # The front-end replaces the first linear projection of the
            # backbone: the backbone then consumes the 256 channels it emits.
            self.backbone.proj = nn.Sequential(
                nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.Dropout(0.15))

        # 2. Conformer Contextualizer Stack
        self.conformer = nn.ModuleList([
            ConformerBlock(d, n_head=n_head, kernel=kernel, ff_exp=4, drop=drop,
                           causal=causal)
            for _ in range(n_blocks)
        ])

        # 3. The expansion AFTER the contextualizer (the Conformer-CTC design)
        self.expander = None
        if self.gru_expand == "after":
            self.expander = GRUExpander(d, layers=1, dropout=drop)

        # 4. Continuous CTC Head
        self.ctc_proj = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, num_classes + 1)
        )

        # 5. Optional auxiliary (deep supervision) head on the contextualizer
        #    output.  It gives the intermediate layers a gradient of their own,
        #    so the whole stack is trained instead of only the top of it.
        self.aux_proj = None
        if aux_head:
            self.aux_proj = nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, num_classes + 1)
            )

    # ------------------------------------------------------------------ #
    def output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        """The number of OUTPUT frames of every clip (after the strided front-end).

        The CTC alignment is defined on the output frames, so the CTC loss and
        the decoder have to use these lengths, not the input lengths.
        """
        if self.stride <= 1 or lengths is None:
            return lengths
        return torch.div(lengths + self.stride - 1, self.stride, rounding_mode="floor")

    @staticmethod
    def _causal_mask(T: int, device) -> torch.Tensor:
        """Upper-triangular mask: a frame may only attend to the past."""
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, x, lengths=None, return_aux=False):
        # x: (B, T, 163)
        # `lengths` is the true (unpadded) length of every sequence in the batch.  When it
        # is given, the padded frames are excluded from the BatchNorm statistics and from the
        # attention, so a batch of very different sentence lengths no longer poisons the
        # running estimates used at evaluation time.
        B, T_in, _ = x.shape
        in_lengths = lengths  # the lengths BEFORE the front-end

        # 0. Strided front-end (reduces the temporal resolution)
        if self.front is not None:
            x = self.front(x)
            lengths = self.output_lengths(lengths)

        mask = None
        if lengths is not None:
            T = x.size(1)
            ar = torch.arange(T, device=x.device).unsqueeze(0)
            lens = lengths.to(x.device).clamp(min=1, max=T).unsqueeze(1)
            mask = (ar < lens).to(x.dtype)          # 1.0 = real frame, 0.0 = padding

        attn_mask = self._causal_mask(x.size(1), x.device) if self.causal else None

        if self.gru_expand in ("before", "before_keep_gru"):
            # The GRU expands the compressed sequence and the Transformer
            # receives the EXPANDED tensors directly.  The attention then runs
            # at the original frame rate.
            h = self.backbone(x, None, attn_mask, target_len=T_in, lengths=in_lengths)
            T = h.size(1)
            if lengths is not None:
                ar = torch.arange(T, device=h.device).unsqueeze(0)
                lens = in_lengths.to(h.device).clamp(min=1, max=T).unsqueeze(1)
                mask = (ar < lens).to(h.dtype)
            for block in self.conformer:
                h = block(h, mask, attn_mask)
        else:
            h = self.backbone(x, mask, attn_mask)
            for block in self.conformer:
                h = block(h, mask, attn_mask)      # (B, T', 256)

            if self.gru_expand == "after":
                # Expand back to the frame rate before the CTC head, so the
                # frame-level alignment of the CTC is preserved.
                h_aux = h
                h = self.expander(h, T_in)
                if mask is not None:
                    ar = torch.arange(h.size(1), device=h.device).unsqueeze(0)
                    lens = in_lengths.to(h.device).clamp(min=1, max=h.size(1)).unsqueeze(1)
                    mask = (ar < lens).to(h.dtype)
            else:
                h_aux = h

        logits = self.ctc_proj(h)              # (B, T'', vocab+1)
        log_probs = F.log_softmax(logits, dim=-1)
        if self.aux_proj is not None and return_aux:
            aux = self.aux_proj(h_aux)         # the contextualizer output
            return log_probs, F.log_softmax(aux, dim=-1)
        return log_probs

    def reseed_ctc_head(self, weights_path=PHASE6_WEIGHTS):
        """Re-initialise the CTC projection from the pretrained word classifier.

        NOTE: the Phase 6 classifier was trained on POOLED features (the average and the
        max over time), while the CTC head works on PER-FRAME features, so the transfer
        is not semantically exact.  Importing it into a converged head is therefore
        destructive - it was measured to push the loss from 14 to 276 in one epoch.  Use
        rebalance_ctc_head() instead for a local-minimum escape.
        """
        if not os.path.exists(weights_path):
            return False
        sd = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        if "head.4.weight" not in sd or "head.4.bias" not in sd:
            return False
        nc = min(self.num_classes, sd["head.4.bias"].shape[0])
        with torch.no_grad():
            self.ctc_proj[1].weight.data[:nc] = sd["head.4.weight"][:nc]
            self.ctc_proj[1].bias.data[:nc] = sd["head.4.bias"][:nc]
            self.ctc_proj[1].bias.data[self.ctc_blank_id] = 0.0
            self.ctc_proj[1].weight.data[self.ctc_blank_id].zero_()
        return True

    def rebalance_ctc_head(self, blank_bias=-1.5, weight_scale=1.0):
        """Stop the blank token from winning every frame.

        A CTC head that has collapsed onto the blank token has a strongly positive bias
        on the blank row.  Lowering that bias (and optionally shrinking the whole
        weight matrix so the logits stay moderate) gives the non-blank classes a chance
        to win the argmax again, without importing a classifier that was trained on a
        different feature space.
        """
        with torch.no_grad():
            if weight_scale != 1.0:
                self.ctc_proj[1].weight.data.mul_(weight_scale)
            if blank_bias is not None:
                self.ctc_proj[1].bias.data[self.ctc_blank_id] = float(blank_bias)
        return True

    def load_phase6_backbone(self, weights_path=PHASE6_WEIGHTS, class_map=None):
        """Load the pretrained word backbone and (optionally) the word classifier.

        ``class_map`` maps OUR class index -> the row index of the pretrained
        classifier.  It is built from the class NAMES, because the vocabulary
        changed (the word loader now trains on the frequency-filtered class
        list), so the row order of the old checkpoint does not match anymore.
        When the map is not given the transfer is only performed if the class
        count did not change at all.
        """
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

        # `load_flexible` drops the entries whose shape changed: with a strided
        # front-end the first projection is replaced (163 -> 256 channels), so
        # the pretrained `proj.*` no longer fits.
        missing, unexpected = load_flexible(self.backbone, bb_sd, label="BACKBONE")
        print(f"[BACKBONE] Loaded {len(bb_sd) - len(missing)} pretrained layers from "
              f"{os.path.basename(weights_path)}")
        if missing: print(f"           Missing keys: {len(missing)}")

        # Transfer pretrained word classification weights into the CTC head.
        if "head.4.weight" in sd and "head.4.bias" in sd:
            src_w, src_b = sd["head.4.weight"], sd["head.4.bias"]
            if class_map is not None:
                # Row-by-row transfer using the class names.
                with torch.no_grad():
                    for our_idx, src_idx in class_map.items():
                        if our_idx >= self.num_classes or src_idx >= src_w.shape[0]:
                            continue
                        self.ctc_proj[1].weight.data[our_idx] = src_w[src_idx]
                        self.ctc_proj[1].bias.data[our_idx] = src_b[src_idx]
                self.ctc_proj[1].bias.data[self.ctc_blank_id] = 0.0
                self.ctc_proj[1].weight.data[self.ctc_blank_id].zero_()
                print(f"[CTC HEAD] Transferred {len(class_map)} pretrained word classifiers "
                      f"by class name (Blank bias set to 0.0).")
            elif src_w.shape[0] == self.num_classes + 1:
                with torch.no_grad():
                    self.ctc_proj[1].weight.data[:self.num_classes] = src_w[:self.num_classes]
                    self.ctc_proj[1].bias.data[:self.num_classes] = src_b[:self.num_classes]
                self.ctc_proj[1].bias.data[self.ctc_blank_id] = 0.0
                self.ctc_proj[1].weight.data[self.ctc_blank_id].zero_()
                print(f"[CTC HEAD] Transferred {self.num_classes} pretrained word classifiers "
                      f"(Blank bias set to 0.0).")
            else:
                print(f"[CTC HEAD] SKIPPED the classifier transfer: the vocabulary changed "
                      f"({src_w.shape[0]} pretrained rows vs {self.num_classes} classes) and no "
                      f"class-name map was given.")

        return True

# =============================================================================
# 6. ANTI-COLLAPSE CTC LOSS
# =============================================================================
class AntiCollapseCTCLoss(nn.Module):
    """
    CTC loss hardened against the all-blank local minimum.

    Plain CTC has a trivially reachable optimum: emit blank everywhere plus one short
    spike per word.  Measured on the collapsed real-sentence checkpoint
    (models/isl_real_sentences/real_sentence_best.pth):

        mean non-blank mass   : 0.026     (the model should spend ~0.15 - 0.30)
        blank-dominant frames : 97.8%     (the blank wins the argmax)
        mean emission entropy : 0.040     (out of ln(263) = 5.57 -> already one-hot)

    The last number is the important one: the collapsed model is NOT diffuse, it is
    *confidently* blank.  That is why the old duration-deficit term could not help - it
    only constrains the summed non-blank mass, which the model satisfies by spreading a
    little probability over many frames while the blank still wins the argmax and the
    greedy decoder still returns nothing.

    So the penalty below is built on the *margin* between the best non-blank class and
    the blank, i.e. exactly the quantity the decoder uses:

        share_t = p_best_nonblank_t / (p_best_nonblank_t + p_blank_t)

    `share_t` is 1 when a non-blank token wins, 0 when the blank wins, and it is smooth
    (no saturating sigmoid), so the gradient survives in the collapsed regime.

    Terms
    -----
    1. ctc          : standard CTC, mean reduction (divides by the target length).
    2. emission     : LOWER bound on the emission count.  Each sample must produce at
                      least `n_words * min_frames_per_sign` "winning" frames.  The
                      penalty is relative (relu(1 - achieved / required)) so it does not
                      vanish as the deficit grows.
    3. blank cap    : UPPER bound on the mean blank probability, so the model cannot
                      simply spam non-blank tokens everywhere to satisfy term 2.
    4. spike penalty: penalises isolated 1-frame non-blank islands (a real sign lasts
                      several frames).
    5. peakedness   : optional per-frame entropy penalty.  Left at 0 by default: the
                      collapsed model already has a near-zero entropy, so this term adds
                      nothing to the blank-collapse fix.
    The emission barrier is safe only because the CTC term is honest: the barrier wants
    many frames where a non-blank token wins, which is exactly the direction that makes
    the alignment infeasible, and an infeasible alignment has an infinite loss that
    zero_infinity would replace with 0.  Together they let the model make the CTC term
    vanish (measured: stage-1 training loss 7.0 with a 0.6% exact-sentence accuracy,
    while the old code showed the other symptom of the same exploit - a corpus WER of
    112-270% with a 74% exact accuracy, because a few samples exploded into hundreds
    of insertions).  With the per-sample filtering above, an infeasible sample is
    dropped instead of being rewarded.
    """

    def __init__(self, blank_id, blank_target=0.97, blank_w=0.5,
                 emission_w=1.0, min_frames_per_sign=10.0,
                 spike_w=0.10, entropy_w=0.0, focal_gamma=0.0,
                 deficit_w=0.0, min_frames_per_word=0.0,
                 class_weights=None,
                 logprob_floor=-20.0, zero_inf=False):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank_id, zero_infinity=zero_inf)
        # Per-sample variant: infeasible samples come back as +inf and are filtered out
        # by the caller instead of being silently replaced by 0.
        self.ctc_none = nn.CTCLoss(blank=blank_id, reduction="none", zero_infinity=zero_inf)
        self.blank_id            = blank_id
        self.blank_target        = float(blank_target)
        self.blank_w             = float(blank_w)
        self.emission_w          = float(emission_w)
        self.min_frames_per_sign = float(min_frames_per_sign)
        self.spike_w             = float(spike_w)
        self.entropy_w           = float(entropy_w)
        self.focal_gamma         = float(focal_gamma)
        # Legacy duration-deficit term (the original formula).  Kept so the old recipe
        # can be reproduced for comparison; it is satisfiable by spreading a little
        # probability over many frames, so it is off by default.
        self.deficit_w            = float(deficit_w)
        self.min_frames_per_word = float(min_frames_per_word)
        # Floor on the log-probabilities: keeps the CTC loss finite (and differentiable)
        # when the alignment is impossible, so "emit nothing" is never free.
        self.logprob_floor       = None if logprob_floor is None else float(logprob_floor)
        # Optional per-class loss weights, used to correct a label distribution that is
        # dominated by a few frequent words.  They DOWN-WEIGHT the frequent classes - the
        # data is kept - so the correction cannot starve the CTC head of supervision.
        if class_weights is None:
            self.class_weights = None
        else:
            self.register_buffer("class_weights",
                                 torch.as_tensor(class_weights, dtype=torch.float32))

    def _sample_weights(self, targets, target_lengths):
        """Per-sample loss weight from the class weights of its target words."""
        B = target_lengths.shape[0]
        dev = targets.device
        cw = self.class_weights.to(dev)
        offs = torch.cumsum(target_lengths.to(dev), 0)
        out = []
        for i in range(B):
            s = int(offs[i] - target_lengths[i])
            e = int(offs[i])
            if e > s:
                w = cw[targets[s:e].long()]
                out.append(w.mean() if w.numel() else torch.ones((), device=dev))
            else:
                out.append(torch.ones((), device=dev))
        return torch.stack(out)

    def forward(self, log_probs_t, targets, input_lengths, target_lengths):
        # --- HONEST CTC TERM -----------------------------------------------
        # nn.CTCLoss(reduction='mean', zero_infinity=True) has a serious exploit: a
        # model that never emits a blank makes the alignment of ANY target impossible,
        # the per-sample loss becomes +inf and is silently replaced by 0.  The model then
        # learns to spam non-blank tokens, the CTC term vanishes and the whole loss
        # collapses to the regularisers.  Measured on the synthetic stage 1: the loss
        # fell to 7.0 while the exact-sentence accuracy was 0.6% (the old code showed
        # the other symptom of the same exploit - a corpus WER of 112-270% with a 74%
        # exact accuracy, because a few samples exploded into hundreds of insertions).
        #
        # The fix is to floor the log-probabilities: the impossible alignment then has
        # a large but FINITE cost with a usable gradient, so the model is pushed to
        # emit and the reported loss is a meaningful progress metric.
        lp = log_probs_t
        if self.logprob_floor is not None:
            lp = lp.clamp(min=self.logprob_floor)
        per_sample = self.ctc_none(lp, targets, input_lengths, target_lengths)
        finite = torch.isfinite(per_sample)
        if finite.any():
            vals = per_sample[finite]
            if self.focal_gamma > 0.0:
                p_seq = torch.exp(-vals.clamp(max=12.0))
                vals = torch.pow(1.0 - p_seq, self.focal_gamma) * vals
            if self.class_weights is not None:
                # Per-sample weight = mean of the weights of the sample's target words.
                # This corrects a label distribution dominated by a few frequent words
                # WITHOUT removing any data (measured: re-sampling the frequent classes
                # away instead removes the supervision and collapses the CTC head - the
                # model then emitted 0.16 words per sentence instead of 1.08).
                w = self._sample_weights(targets, target_lengths)
                vals = vals * w[finite]
                ctc_loss = vals.sum() / w[finite].sum().clamp(min=1e-6)
            else:
                ctc_loss = vals.mean()
        else:
            # Every sample in the batch is infeasible - keep the graph alive.
            ctc_loss = per_sample.sum() * 0.0
        total_loss = ctc_loss

        lp = log_probs_t.transpose(0, 1)  # (B, T, C)
        B, T, C = lp.shape
        blank_lp    = lp[:, :, self.blank_id]
        p_blank     = torch.exp(blank_lp)                 # (B, T)
        non_blank_p = 1.0 - p_blank                       # (B, T)

        # Valid-frame mask (the collate pads the batch to the longest sequence).
        lens = input_lengths.to(lp.device).clamp(min=1, max=T).unsqueeze(1)
        ar = torch.arange(T, device=lp.device).unsqueeze(0)
        valid = (ar < lens).to(lp.dtype)                  # (B, T)
        n_valid = valid.sum(1).clamp(min=1.0)             # (B,)

        # 1. EMISSION DEFICIT (the lower bound that actually prevents the all-blank
        #    collapse).  Counts the frames where a non-blank token is at least as likely
        #    as the blank - the same comparison the greedy decoder makes.
        #
        #    The penalty is a log barrier, log(required / achieved).  A linear
        #    relu(1 - achieved/required) term has a gradient of 1/required per frame,
        #    which is far too weak in the collapsed regime; the barrier's gradient is
        #    1/achieved, so it keeps pushing while the model emits nothing at all.
        if self.emission_w > 0.0 and self.min_frames_per_sign > 0.0 and self.blank_id > 0:
            p_best_nb = lp[:, :, :self.blank_id].exp().max(dim=-1).values   # (B, T)
            share = p_best_nb / (p_best_nb + p_blank + 1e-8)                # (B, T)
            achieved = (share * valid).sum(1)                              # (B,)
            required = target_lengths.to(lp.device).float() * self.min_frames_per_sign
            # Never demand more than half of the clip: the remaining frames are pauses
            # and out-of-scope signs that legitimately have to stay blank.
            required = torch.minimum(required, n_valid * 0.5).clamp(min=1.0)
            # A sample with an EMPTY target must not be pushed to emit anything.
            has_target = (target_lengths.to(lp.device) > 0).float()
            barrier = F.relu(torch.log(required) - torch.log(achieved.clamp(min=1e-4)))
            total_loss = total_loss + self.emission_w * (barrier * has_target).mean()

        # 2. BLANK CAP (the upper bound, so the model cannot spam non-blank tokens).
        if self.blank_w > 0.0:
            mean_blank = (p_blank * valid).sum(1) / n_valid
            excess = F.relu(mean_blank - self.blank_target)
            total_loss = total_loss + self.blank_w * excess.mean()

        # 2. Legacy duration deficit (the original formula, off by default).
        if self.deficit_w > 0.0 and self.min_frames_per_word > 0.0:
            target_mass = target_lengths.to(lp.device).float() * self.min_frames_per_word
            actual_mass = (non_blank_p * valid).sum(1)
            deficit = F.relu(target_mass - actual_mass)
            total_loss = total_loss + self.deficit_w * deficit.mean()

        # 3. Spike penalty: isolated 1-frame non-blank islands.
        if self.spike_w > 0.0 and T > 2:
            left_blank_p   = p_blank[:, :-2]
            right_blank_p  = p_blank[:, 2:]
            center_nb_p    = non_blank_p[:, 1:-1]
            spike_penalty  = (center_nb_p * left_blank_p * right_blank_p).sum() / valid[:, 1:-1].sum().clamp(min=1.0)
            total_loss = total_loss + self.spike_w * spike_penalty

        # 4. Peakedness (optional; no effect on the blank collapse, see the docstring).
        if self.entropy_w > 0.0:
            ent = -(lp.exp() * lp).sum(-1)            # (B, T)
            mean_ent = (ent * valid).sum(1) / n_valid
            total_loss = total_loss + self.entropy_w * mean_ent.mean()

        return total_loss


# Backward-compatible alias: the old name is used by the other trainers.
ConformityCTCLoss = AntiCollapseCTCLoss

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
# 7b. LOCAL-MINIMUM ESCAPE
# =============================================================================
def escape_local_minimum(model, optimizer, blank_collapsed=False, scale=0.02,
                         reseed_head=False, blank_bias=-1.5, weight_scale=0.95,
                         phase6_weights=PHASE6_WEIGHTS):
    """Kick the optimiser out of a flat CTC local minimum.

    Four mechanisms are applied together, because any single one is usually not enough
    once the CTC head has settled on the blank token:

      1. Gaussian perturbation of the trainable Conformer / CTC-head weights
         (SGD-noise style).  The pretrained backbone is left untouched so the isolated
         sign representation survives.
      2. Reset of the BatchNorm running statistics - but ONLY for the layers that are
         actually in training mode.  The runners keep the BatchNorm frozen (`set_bn_eval`),
         so their running statistics are the ones used at evaluation time; resetting them
         would silently disable the normalisation and blow the loss up.
      3. Rebalancing of the CTC head: the blank bias is lowered and the weight matrix is
         slightly shrunk, so the blank token cannot win every frame any more.  Importing
         the Phase 6 word classifier instead was measured to push the loss from 14 to
         276 in a single epoch, because that classifier was trained on POOLED features
         while the CTC head works on PER-FRAME features.
      4. The caller is expected to restart the LR schedule (cosine warm restart) so the
         optimiser takes a fresh, larger step.

    Returns a short human-readable description of what was done.
    """
    import torch.nn as nn

    touched = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not p.requires_grad or p.dim() < 1:
                continue
            if name.startswith("backbone."):
                continue
            sigma = p.std().clamp(min=1e-3)
            p.add_(torch.randn_like(p) * scale * sigma)
            touched += 1

    bn_reset = 0
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.training:
            m.reset_running_stats()
            bn_reset += 1

    rebalanced = False
    reseeded = False
    if hasattr(model, "rebalance_ctc_head"):
        rebalanced = bool(model.rebalance_ctc_head(blank_bias=blank_bias,
                                                    weight_scale=weight_scale))
    if reseed_head and hasattr(model, "reseed_ctc_head"):
        reseeded = bool(model.reseed_ctc_head(phase6_weights))

    for g in optimizer.param_groups:
        if "base_lr" in g:
            g["lr"] = g["base_lr"]

    return (f"perturbed {touched} tensors (sigma={scale}), reset {bn_reset} trainable BN stats, "
            f"rebalanced CTC head={rebalanced} (blank_bias={blank_bias}), "
            f"re-seeded={reseeded}, LR reset to base")


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
              is_train=True, ema=None, max_norm=1.0, mask_padding=True, min_dur=3,
              beam_width=0, beam_lm=None, beam_alpha=0.0, beam_beta=0.0,
              beam_topk=5, id_to_word=None):
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

            # mask_padding=False reproduces the legacy behaviour (the padded frames are
            # attended to and pollute the BatchNorm statistics).
            lp = model(xs, lengths) if mask_padding else model(xs)
            lp_t = lp.transpose(0, 1)            # (T, B, C)

            # The CTC alignment lives on the OUTPUT frames: with a strided
            # front-end there are fewer of them than there are input frames.
            out_lengths = (model.output_lengths(lengths)
                           if hasattr(model, "output_lengths") else lengths)

            loss = criterion(lp_t, targets, out_lengths, target_lengths)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                if ema is not None: ema.update(model)

            total_loss += loss.item()

            if beam_width and beam_width > 0:
                # Prefix beam search (optionally rescored by the word-level LM).
                decoded = batch_decode_with_lm(
                    lp, out_lengths, blank_id, id_to_word=id_to_word, lm=beam_lm,
                    alpha=beam_alpha, beta=beam_beta, beam_width=beam_width,
                    topk_labels=beam_topk,
                    min_dur=max(1, min_dur // max(1, getattr(model, "stride", 1))))
            else:
                decoded = batch_decode(lp, out_lengths, blank_id,
                                       min_dur=max(1, min_dur // max(1, getattr(model, "stride", 1))))
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
    # Word-corpus selection (shared with utils/dataloader_word.py)
    p.add_argument("--min_samples_per_class", type=int, default=14,
                   help="A class with fewer clips than this is eliminated from the vocabulary")
    p.add_argument("--max_samples_per_class", type=int, default=15,
                   help="Hard budget of clips loaded per class")
    p.add_argument("--fill_missing",    type=str, default="none",
                   choices=["none", "hold", "interpolate"],
                   help="Fill the drop-out holes of a clip with the last valid value")
    # Model parameters
    p.add_argument("--d_model",         type=int, default=256)
    p.add_argument("--n_blocks",        type=int, default=3)
    p.add_argument("--n_head",          type=int, default=8)
    p.add_argument("--kernel",          type=int, default=31)
    p.add_argument("--dropout",         type=float, default=0.1)
    p.add_argument("--stride",          type=int, default=1, choices=[1, 2, 4, 8],
                   help="Temporal stride of the convolution front-end (1 = the historical "
                        "behaviour, one output frame per input frame).  stride=2 halves "
                        "the resolution before the self-attention: 4x cheaper attention.")
    p.add_argument("--causal",          type=int, default=0,
                   help="Causal model (unidirectional GRU + causal attention mask) for "
                        "streaming inference.  0 = the bidirectional offline model.")
    # Learning rates
    p.add_argument("--backbone_lr",     type=float, default=1.0e-5, help="LR for pretrained BiGRU+Trans+TCNN")
    p.add_argument("--conformer_lr",    type=float, default=1.5e-4, help="LR for Conformer blocks & CTC head")
    # Loss parameters
    p.add_argument("--blank_target",     type=float, default=0.97,
                   help="Upper bound on the mean blank probability (0.97 = at most 97%% blank)")
    p.add_argument("--blank_w",          type=float, default=0.5,
                   help="Weight of the blank cap")
    p.add_argument("--emission_w",       type=float, default=1.0,
                   help="Weight of the emission deficit (the anti all-blank-collapse term)")
    p.add_argument("--min_frames_per_sign", type=float, default=10.0,
                   help="Minimum frames a sign should occupy; drives the required emission count")
    p.add_argument("--connectivity_w",   type=float, default=0.10, help="Spike penalty weight")
    p.add_argument("--entropy_w",        type=float, default=0.0,
                   help="Per-frame emission-entropy penalty (0 disables)")
    p.add_argument("--deficit_w",        type=float, default=0.0,
                   help="(legacy) duration deficit penalty weight - 0 disables it")
    p.add_argument("--min_frames",       type=float, default=0.0,
                   help="(legacy) min non-blank frames per word - only used with --deficit_w")
    # Local-minimum escape
    p.add_argument("--escape_patience",  type=int, default=6,
                   help="Epochs without val-WER improvement before an escape is triggered (0 disables)")
    p.add_argument("--escape_scale",     type=float, default=0.02,
                   help="Weight perturbation sigma used by the escape")
    # Prefix beam search + word-level language model
    p.add_argument("--beam_width",       type=int, default=0,
                   help="Width of the CTC prefix beam search (0 = the greedy decoder)")
    p.add_argument("--beam_topk",        type=int, default=5,
                   help="How many labels per frame the beam search expands")
    p.add_argument("--beam_alpha",       type=float, default=0.0,
                   help="Weight of the word-level language model in the beam rescoring "
                        "(0 = pure acoustic)")
    p.add_argument("--beam_beta",        type=float, default=0.0,
                   help="Insertion penalty of the beam rescoring")
    p.add_argument("--lm_index",         type=str,
                   default=os.path.join(BACKEND_DIR, "data", "tensors_sentence_level", "index.json"),
                   help="Sentence index the word-level language model is estimated from")
    p.add_argument("--lm_file",          type=str, default=None,
                   help="Save / load the language model at this path")
    p.add_argument("--mask_padding", type=int, default=1,
                   help="Exclude the padded frames from the attention and from the "
                        "BatchNorm statistics (0 = legacy behaviour, the padded frames "
                        "are attended to and pollute the statistics)")
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

    # ---- vocabulary ------------------------------------------------------
    # The CTC vocabulary is the vocabulary of the WORD loader: the top-most
    # frequent classes of the word corpus (a class below the 14-clip tier is
    # eliminated) with a 15-clip budget per class.
    classes = get_word_classes(
        args.data_dir,
        min_samples_per_class=args.min_samples_per_class,
        max_samples_per_class=args.max_samples_per_class,
    )
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
    model = ISLHybridConformerCTC(num_classes, args.d_model, args.n_blocks, args.n_head,
                                   args.kernel, args.dropout,
                                   stride=args.stride, causal=bool(args.causal)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {n_params:,}")
    if args.stride > 1:
        print(f"[FRONT-END] Stride {args.stride}: the CTC head works on T/{args.stride} frames.")
    if args.causal:
        print("[FRONT-END] Causal model: streaming capable (no future context).")

    # Initialize pretrained backbone (the word classifier is transferred BY CLASS
    # NAME, because the vocabulary is not the one the checkpoint was trained on).
    phase6_classes = None
    if os.path.exists(CLASSES_SRC):
        try:
            with open(CLASSES_SRC) as f:
                phase6_classes = json.load(f)
            if isinstance(phase6_classes, dict):
                phase6_classes = phase6_classes.get("classes", [])
        except Exception:
            phase6_classes = None

    class_map = None
    if phase6_classes:
        src_idx = {c: i for i, c in enumerate(phase6_classes)}
        class_map = {i: src_idx[c] for i, c in enumerate(classes) if c in src_idx}

    model.load_phase6_backbone(PHASE6_WEIGHTS, class_map=class_map)

    start_ep  = 0
    best_wer  = float("inf")
    best_ema_wer = float("inf")
    history = {"train_loss":[], "val_loss":[], "val_loss_ema":[], "wer":[], "wer_ema":[], "acc":[], "f1":[], "lr_conf":[], "lr_bb":[]}

    # Load checkpoint if resuming or warm-starting
    loaded = False
    if args.resume and os.path.exists(CKPT_FILE):
        print(f"[RESUME] Loading {CKPT_FILE}")
        ckpt = torch.load(CKPT_FILE, map_location=device, weights_only=False)
        load_flexible(model, extract_state_dict(ckpt), label="RESUME")
        start_ep = ckpt.get("epoch", 0)
        best_wer = ckpt.get("best_wer", float("inf"))
        best_ema_wer = ckpt.get("best_ema_wer", float("inf"))
        history = ckpt.get("history", history)
        loaded = True
        print(f"[RESUME] Resuming at epoch {start_ep}, best WER: {best_wer*100:.2f}%")

    if not loaded and args.weights and os.path.exists(args.weights):
        print(f"[LOAD] Weights from: {args.weights}")
        raw = torch.load(args.weights, map_location=device, weights_only=False)
        load_flexible(model, extract_state_dict(raw), label="WEIGHTS")

    if not loaded and not args.weights:
        prev_stage = stage - 1
        if prev_stage >= 1:
            prev_best = os.path.join(args.output_dir, f"stage{prev_stage}_best.pth")
            if os.path.exists(prev_best):
                print(f"[WARM-START] Loading stage {prev_stage} weights from {prev_best}")
                sd = torch.load(prev_best, map_location=device, weights_only=True)
                load_flexible(model, sd, label=f"STAGE{prev_stage}")

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
    # Remember the base LR so the escape can restart the schedule from it.
    for g in param_groups:
        g["base_lr"] = g["lr"]

    criterion = AntiCollapseCTCLoss(
        CTC_BLANK_ID,
        blank_target=args.blank_target,
        blank_w=args.blank_w,
        emission_w=args.emission_w,
        # The emission barrier counts OUTPUT frames, so with a strided
        # front-end the requirement has to be divided by the stride.
        min_frames_per_sign=max(1.0, args.min_frames_per_sign / max(1, args.stride)),
        spike_w=args.connectivity_w,
        entropy_w=args.entropy_w,
        deficit_w=args.deficit_w,
        min_frames_per_word=args.min_frames,
    )
    optimizer = optim.AdamW(param_groups, betas=(0.9, 0.98))
    warmup_ep = max(2, min(5, epochs // 4))
    warmup_sched = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, total_iters=warmup_ep)
    cosine_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_ep), eta_min=cfg["eta_min"])
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_ep])
    ema = ModelEMA(model, decay=0.9995)

    # Datasets
    train_ds = SyntheticSentenceDataset(args.data_dir, classes, args.train_samples, True,  cfg["max_w"], cfg["min_w"], cfg["aug"],
                                        max_per_class=args.max_samples_per_class, fill_missing=args.fill_missing)
    val_ds   = SyntheticSentenceDataset(args.data_dir, classes, args.val_samples,   False, cfg["max_w"], cfg["min_w"], 0.0,
                                        max_per_class=args.max_samples_per_class, fill_missing=args.fill_missing)
    train_ld = DataLoader(train_ds, args.batch_size, shuffle=True,  collate_fn=collate, num_workers=args.num_workers)
    val_ld   = DataLoader(val_ds,   args.batch_size, shuffle=False, collate_fn=collate, num_workers=args.num_workers)

    # ---- word-level language model (optional) -----------------------------
    # The LM is estimated on the TRAINING split of the continuous corpus only:
    # the validation sentences must stay unseen, otherwise the tuning of
    # --beam_alpha would be meaningless.
    beam_lm = None
    if args.beam_width and args.beam_width > 0:
        if args.lm_file and os.path.exists(args.lm_file):
            from utils.word_lm import WordLM
            beam_lm = WordLM.load(args.lm_file)
            print(f"[LM] Loaded the language model from {args.lm_file}")
        elif os.path.exists(args.lm_index):
            beam_lm = build_word_lm_from_index(args.lm_index, classes=classes)
            print(f"[LM] Word-level LM built from {args.lm_index}: "
                  f"{len(beam_lm.unigram)} word types, {sum(beam_lm.unigram.values())} tokens")
            if args.lm_file:
                beam_lm.save(args.lm_file)
        else:
            print(f"[LM] {args.lm_index} not found - the beam search will be purely acoustic.")

    # Training Loop
    total_eps = start_ep + epochs
    epochs_without_improvement = 0
    escapes = 0
    for ep in range(start_ep, total_eps):
        cur_conf_lr = optimizer.param_groups[-1]["lr"]
        cur_bb_lr   = optimizer.param_groups[0]["lr"] if bb_lr > 0.0 else 0.0

        if stage == 3:
            progress = (ep - start_ep) / max(1, epochs - 1)
            cur_max = max(cfg["min_w"], min(cfg["max_w"], int(1 + progress * (cfg["max_w"] - 1))))
            train_ds.max_words = cur_max

        tl, wer, acc, f1 = run_epoch(model, train_ld, criterion, optimizer, device, CTC_BLANK_ID, True, ema,
                                     mask_padding=args.mask_padding,
                                     beam_width=args.beam_width, beam_lm=beam_lm,
                                     beam_alpha=args.beam_alpha, beam_beta=args.beam_beta,
                                     beam_topk=args.beam_topk, id_to_word=classes)
        vl, vwer, vacc, vf1 = run_epoch(model, val_ld, criterion, optimizer, device, CTC_BLANK_ID, False,
                                        mask_padding=args.mask_padding,
                                        beam_width=args.beam_width, beam_lm=beam_lm,
                                        beam_alpha=args.beam_alpha, beam_beta=args.beam_beta,
                                        beam_topk=args.beam_topk, id_to_word=classes)
        ema_vl, ema_wer, _, _ = run_epoch(ema.model, val_ld, criterion, None, device, CTC_BLANK_ID, False,
                                          mask_padding=args.mask_padding,
                                          beam_width=args.beam_width, beam_lm=beam_lm,
                                          beam_alpha=args.beam_alpha, beam_beta=args.beam_beta,
                                          beam_topk=args.beam_topk, id_to_word=classes)

        history["train_loss"].append(tl); history["val_loss"].append(vl); history["val_loss_ema"].append(ema_vl)
        history["wer"].append(vwer); history["wer_ema"].append(ema_wer)
        history["acc"].append(vacc); history["f1"].append(vf1)
        history["lr_conf"].append(cur_conf_lr); history["lr_bb"].append(cur_bb_lr)

        scheduler.step()

        curriculum_note = f" | max_words={train_ds.max_words}" if stage == 3 else ""
        print(f"\n[S{stage} Ep {ep+1:03d}/{total_eps:03d}] Conf LR={cur_conf_lr:.2e} | BB LR={cur_bb_lr:.2e}{curriculum_note}")
        print(f"  Train Loss: {tl:.4f} | Val Loss: {vl:.4f} (EMA: {ema_vl:.4f})")
        print(f"  WER: {vwer*100:.2f}% | WER EMA: {ema_wer*100:.2f}% | ExAcc: {vacc*100:.2f}% | F1: {vf1:.4f}")

        # --- Local-minimum escape -------------------------------------------
        if vwer < best_wer - 1e-4:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if args.escape_patience > 0 and epochs_without_improvement >= args.escape_patience:
            escapes += 1
            print(f"  [ESCAPE] No val-WER improvement for {epochs_without_improvement} epochs "
                  f"(WER {vwer*100:.2f}%). Kicking the model out of the local minimum.")
            info = escape_local_minimum(model, optimizer, scale=args.escape_scale)
            print(f"  [ESCAPE] {info}")
            ema = ModelEMA(model, decay=0.9995)
            # Restart the LR schedule from the base LR (cosine warm restart).
            for g in optimizer.param_groups:
                g["lr"] = g.get("base_lr", g["lr"])
            warmup_sched = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=1)
            cosine_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - ep), eta_min=cfg["eta_min"])
            scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[1])
            epochs_without_improvement = 0

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
