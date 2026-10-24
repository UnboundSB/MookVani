"""
train_noise_filter.py
=====================
Stage 1 of the segmentor curriculum: NOISE FILTRATION.

Two copies of the same YAP model (teacher=clean, student=noisy) are loaded
into memory simultaneously and fed the same word-level clip. The student must
produce presence/end/entity outputs consistent with the teacher despite seeing
a MediaPipe-corrupted version of the clip.

Loss = consistency + word-label (classification)

    L = L_consist + lambda_cls * L_cls

Consistency forces the student to agree with the teacher on:
  * per-pose presence probability         (KL divergence, frame-level)
  * per-pose end probability              (KL divergence, frame-level)
  * entity vectors (after HiPPO readout)  (MSE, after soft alignment)

Word-label keeps both from collapse: every clip has exactly 1 word label;
the entity from the clean path is classified by a small head.

Dataset
-------
Reads the word-level tensor dataset the backbone was pretrained on
(Backend/data/tensors_include50_word_level or tensors_isl_csltr_word_level).
Each .npy file: (T, J*3) float32 (MediaPipe coords, flattened).
The class is inferred from the filename (class_index_<label>.npy pattern or
via the index.json used by the CTC trainers).

Run
---
    python Backend/segmentor/train_noise_filter.py \\
        --out_dir Backend/models/yap_stage1 --epochs 30

After training, the student weights are saved as yap_noise_robust.pth.
The teacher weights are discarded (they are just the EMA / clean reference).
"""
from __future__ import annotations

import argparse, json, os, random, sys, gc
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# -- local imports (run from the repo root or add segmentor to sys.path) -----
# The Windows console is cp1252: a stray box-drawing character would
# crash the run at the LAST print.
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))  # Backend/
from utils.checkpoint_utils import load_flexible
from utils.pose_augmentation import AugmentConfig, augment_hands_batch

from yap_conformer import (
    YAPConfig, YAPCore, YAP, YAPpingConformer, ConformerConfig,
    focal_presence_loss, layer_conf_loss, end_targets, presence_targets,
    make_segments, frame_dts,
)
from noise_aug import add_noise
from tqdm import tqdm
from utils.plotting import plot_training_curves, write_report

# -- defaults -----------------------------------------------------------------
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIRS = [
    os.path.join(BACKEND_DIR, "data", "tensors_include50_word_level"),
    os.path.join(BACKEND_DIR, "data", "tensors_isl_csltr_word_level"),
]
DEFAULT_OUT = os.path.join(BACKEND_DIR, "segmentor", "models", "yap_stage1")
# The published vocabulary of the project (the top-most frequent classes of the
# word corpus) - the same list every other trainer uses.
CLASSES_PATH = os.path.join(BACKEND_DIR, "models", "word_vocabulary", "classes.json")

# The canonical augmentation set (the same one the word and sentence models
# use).  It is applied to the CLEAN clip, so the teacher and the student see
# the same augmented clip and the only difference between them is the noise.
AUG_CFG = AugmentConfig(
    flip_prob=0.5,
    tilt_prob=0.5,
    tilt_deg=12.0,
    shift_prob=0.5,
    shift_dx=0.05,
    shift_dy=0.02,
    scale_prob=0.4,
    scale_lo=0.90,
    scale_hi=1.10,
    speed_prob=0.4,
    speed_lo=0.85,
    speed_hi=1.15,
    time_shift_prob=0.2,
    time_shift_max=3,
    mask_prob=0.0,
    noise_std=0.0,
)


# ==============================================================================
# Dataset
# ==============================================================================
class WordClipDataset(Dataset):
    """
    Each item is ONE word-level clip. Returns:
      pose (T,J,3) float  clean MediaPipe landmarks (harmonized)
      conf (T,J)   float  confidence (1.0 if unavailable)
      valid (T,J)  bool   all True for stored tensors (dropout added in aug)
      label        int    word class index
      t            (T,)   synthetic uniform timestamps at 30 fps
    """

    def __init__(self, clips: list[dict], num_joints: int, max_len: int = 400):
        self.clips = clips
        self.J = num_joints
        self.max_len = max_len

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        item = self.clips[idx]
        raw = np.load(item["path"]).astype(np.float32)          # (T, J*3)
        T = min(len(raw), self.max_len)
        raw = raw[:T]
        pose = torch.from_numpy(raw).view(T, self.J, 3)
        conf = torch.ones(T, self.J)
        valid = torch.ones(T, self.J, dtype=torch.bool)
        t = torch.arange(T, dtype=torch.float32) / 30.0
        return pose, conf, valid, item["label"], t


def _macro_f1(preds, labels, num_classes: int) -> float:
    """Macro F1 without sklearn dependency."""
    f1s = []
    for c in range(num_classes):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


def _collate(batch):
    """Pad to the longest clip in the batch."""
    poses, confs, valids, labels, ts = zip(*batch)
    B = len(poses)
    T = max(p.shape[0] for p in poses)
    J = poses[0].shape[1]
    pose_b = torch.zeros(B, T, J, 3)
    conf_b = torch.zeros(B, T, J)
    valid_b = torch.zeros(B, T, J, dtype=torch.bool)
    t_b = torch.zeros(B, T)
    lengths = torch.zeros(B, dtype=torch.long)
    for i, (p, c, v, _, t_) in enumerate(zip(poses, confs, valids, labels, ts)):
        n = p.shape[0]
        pose_b[i, :n] = p
        conf_b[i, :n] = c
        valid_b[i, :n] = v
        t_b[i, :n] = t_
        t_b[i, n:] = t_[-1]        # pad timestamps (harmless, valid=False)
        lengths[i] = n
    return pose_b, conf_b, valid_b, torch.tensor(labels, dtype=torch.long), t_b, lengths


def load_clips(data_dirs, classes_path, max_per_class=None):
    """Scan data directories for .npy word clips. Returns list of dicts."""
    # Load class list for label mapping
    with open(classes_path) as f:
        classes = json.load(f)
    label_map = {c: i for i, c in enumerate(classes)}

    clips = []
    for d in data_dirs:
        d = Path(d)
        if not d.exists():
            continue
        # Try index.json first (same format as CTC trainers)
        idx_path = d / "index.json"
        if idx_path.exists():
            with open(idx_path) as f:
                entries = json.load(f)
            for e in entries:
                lbl = e.get("label") or e.get("word")
                if lbl not in label_map:
                    continue
                p = d / e["path"] if not os.path.isabs(e["path"]) else Path(e["path"])
                if p.exists():
                    clips.append({"path": str(p), "label": label_map[lbl]})
        else:
            # Fallback: scan .npy files named <anything>/<label>/<...>.npy
            for npy in d.rglob("*.npy"):
                lbl = npy.parent.name
                if lbl in label_map:
                    clips.append({"path": str(npy), "label": label_map[lbl]})

    if max_per_class:
        from collections import defaultdict
        by_cls = defaultdict(list)
        for c in clips:
            by_cls[c["label"]].append(c)
        clips = []
        for items in by_cls.values():
            random.shuffle(items)
            clips.extend(items[:max_per_class])

    random.shuffle(clips)
    return clips


# ==============================================================================
# Consistency losses
# ==============================================================================
def presence_kl(clean_logit, noisy_logit, lengths):
    """KL(clean || noisy) on per-pose presence probability, masked to valid frames."""
    B, T = clean_logit.shape
    mask = torch.zeros(B, T, dtype=torch.bool, device=clean_logit.device)
    for b, ln in enumerate(lengths):
        mask[b, :ln] = True
    p_c = torch.sigmoid(clean_logit).clamp(1e-6, 1 - 1e-6)
    p_n = torch.sigmoid(noisy_logit).clamp(1e-6, 1 - 1e-6)
    kl = p_c * (p_c.log() - p_n.log()) + (1 - p_c) * ((1 - p_c).log() - (1 - p_n).log())
    return (kl * mask.float()).sum() / mask.float().sum().clamp_min(1)


def entity_consistency(ents_clean, ents_noisy):
    """
    MSE between the entity vectors of clean and noisy paths.
    Both are shape (S, entity_dim). S may differ if the noisy segmenter
    finds a different number of segments; we align by min(S_c, S_n).
    """
    S = min(ents_clean.shape[0], ents_noisy.shape[0])
    if S == 0:
        return ents_clean.sum() * 0.0
    return F.mse_loss(ents_clean[:S], ents_noisy[:S].detach())


# ==============================================================================
# Training
# ==============================================================================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # -- model config ----------------------------------------------------------
    # The body layout must match the corpus: our tensors store the 2 hands, the
    # 2 shoulders + the 2 elbows and 7 face points (J = 53).  The hands-only
    # preset (J=42) throws the shoulders away (and with them the body frame),
    # and the holistic preset (J=79) asks for points the corpus does not have.
    _layout = getattr(args, "layout", "corpus")
    _cfg_kwargs = dict(d_model=args.d_model, n_mem_layers=args.n_mem_layers,
                       summary_channels=args.summary_channels,
                       hippo_order=args.hippo_order, entity_dim=args.entity_dim)
    if _layout == "corpus":
        ycfg = YAPConfig.corpus(**_cfg_kwargs)
    elif _layout == "holistic":
        ycfg = YAPConfig.holistic(**_cfg_kwargs)
    else:
        ycfg = YAPConfig(**_cfg_kwargs)
    print(f"[LAYOUT] {_layout}: J = {ycfg.total_joints} "
          f"({', '.join(f'{p.name}={p.num_joints}' for p in ycfg.parts)})")
    J = ycfg.total_joints

    # -- data ------------------------------------------------------------------
    with open(args.classes) as f:
        classes = json.load(f)
    num_classes = len(classes)

    all_clips = load_clips(args.data_dirs, args.classes, max_per_class=args.max_per_class)
    if not all_clips:
        raise RuntimeError(f"No clips found in {args.data_dirs}. Check paths.")
    split = max(1, int(len(all_clips) * 0.9))
    train_clips, val_clips = all_clips[:split], all_clips[split:]
    print(f"[DATA] {len(train_clips)} train clips, {len(val_clips)} val clips, {num_classes} classes")

    train_ds = WordClipDataset(train_clips, J, args.max_len)
    val_ds   = WordClipDataset(val_clips,   J, args.max_len)
    train_ld = DataLoader(train_ds, args.batch_size, shuffle=True,  collate_fn=_collate, num_workers=args.num_workers)
    val_ld   = DataLoader(val_ds,   args.batch_size, shuffle=False, collate_fn=_collate, num_workers=args.num_workers)

    # -- two model instances in memory -----------------------------------------
    # teacher: receives CLEAN clips, no grad (reference signal)
    # student: receives NOISY clips, gradients flow through it
    # They share NO weights. The student is what we train and save.
    teacher = YAPpingConformer(ycfg, ConformerConfig(num_words=num_classes, d_model=args.d_model)).to(device)
    student = YAPpingConformer(ycfg, ConformerConfig(num_words=num_classes, d_model=args.d_model)).to(device)

    # Classification head (word label from the clean entity)
    cls_head = nn.Linear(args.entity_dim, num_classes).to(device)

    # If resuming or initialising from a pretrained YAP
    if args.init and os.path.exists(args.init):
        sd = torch.load(args.init, map_location=device, weights_only=True)
        load_flexible(student, sd, label="INIT")
        load_flexible(teacher, sd, label="INIT", verbose=False)
        print(f"[INIT] loaded weights from {args.init}")
    if args.resume and os.path.exists(os.path.join(args.out_dir, "yap_noise_ckpt.pth")):
        ck = torch.load(os.path.join(args.out_dir, "yap_noise_ckpt.pth"), map_location=device, weights_only=False)
        load_flexible(student, ck["student"], label="RESUME")
        load_flexible(teacher, ck["teacher"], label="RESUME", verbose=False)
        load_flexible(cls_head, ck["cls_head"], label="RESUME")
        print(f"[RESUME] epoch {ck['epoch']}")

    optimizer = torch.optim.AdamW(list(student.parameters()) + list(cls_head.parameters()),
                                  lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.t0, T_mult=args.t_mult)

    os.makedirs(args.out_dir, exist_ok=True)
    if not os.path.isdir(args.out_dir):
        raise RuntimeError(f"[FATAL] could not create out_dir: {args.out_dir}")
    print(f"[OUT] out_dir verified: {os.path.abspath(args.out_dir)}")
    for dd in args.data_dirs:
        if not os.path.isdir(dd):
            raise RuntimeError(f"[FATAL] data_dir not found: {dd}")
        print(f"[DATA] data_dir verified: {os.path.abspath(dd)}")
    best_loss = float("inf")
    rng = random.Random(args.seed)
    history = {"train_loss": [], "true_loss": [], "train_acc": [], "train_f1": [], "lr": [], "blank": [], "div": []}

    print("=" * 72)
    print("YAP Stage 1 - NOISE FILTRATION  (kicked off)")
    print(f"  device        : {device}")
    print(f"  data_dirs     : {args.data_dirs}")
    print(f"  classes       : {args.classes} ({num_classes} classes)")
    print(f"  out_dir       : {args.out_dir}")
    print(f"  layout        : {args.layout}  (J={J})")
    print(f"  epochs        : {args.epochs}  |  batch_size={args.batch_size}  grad_accum={args.grad_accum}  (eff. batch={args.batch_size*args.grad_accum})")
    print(f"  d_model       : {args.d_model}  n_mem_layers={args.n_mem_layers}  summary_channels={args.summary_channels}  hippo_order={args.hippo_order}  entity_dim={args.entity_dim}")
    print(f"  lr            : {args.lr}  ema_decay={args.ema_decay}  cls_w={args.cls_w}  ent_w={args.ent_w}")
    print(f"  aug           : {args.aug}  aug_strength={args.aug_strength}  max_len={args.max_len}  seed={args.seed}")
    print(f"  noise         : scale[{args.noise_scale_min},{args.noise_scale_max}]  prob={args.noise_prob}  jitter_std={args.jitter_std}")
    print(f"  anti-blank    : blank_w={args.blank_w}  blank_floor={args.blank_floor}")
    print(f"  anti-collapse : div_w={args.div_w}")
    print(f"  scheduler     : CosineAnnealingWarmRestarts T0={args.t0} T_mult={args.t_mult} (simulated annealing)")
    print(f"  num_workers   : {args.num_workers}")
    print("=" * 72)
    print()

    for ep in tqdm(range(args.epochs), desc="Stage 1 epochs", leave=True):
        student.train(); teacher.eval(); cls_head.train()
        ep_loss = 0.0; n_ok = 0
        all_preds = []
        all_labels = []

        micro = 0
        steps_done = 0
        pbar = tqdm(train_ld, desc=f"Ep {ep+1:02d}/{args.epochs}", leave=False)
        for pose, conf, valid, labels, t, lengths in pbar:
            pose, conf, valid, labels, t, lengths = (
                pose.to(device), conf.to(device), valid.to(device),
                labels.to(device), t.to(device), lengths.to(device))

            # -- build noisy batch ---------------------------------------------
            # -- the canonical augmentation (mirror / tilt / shift / speed) --
            # It is applied ONCE, to the clean clip, and the teacher and the
            # student then see the SAME augmented clip: the only difference
            # between them is the noise.  (If the two saw different
            # augmentations the consistency loss would punish the student for
            # the augmentation instead of for the noise.)
            if args.aug:
                pose = augment_hands_batch(pose, lengths, AUG_CFG.scaled(args.aug_strength),
                                           rng=rng)

            # -- build noisy batch --
            # The noise INTENSITY is drawn at random for every batch, so the
            # student learns to be robust across the whole range of corruption
            # levels instead of only at the nominal one.
            scale = 0.0
            if rng.random() < args.noise_prob:
                scale = rng.uniform(args.noise_scale_min, args.noise_scale_max)
            pose_n_list, conf_n_list, valid_n_list, len_n = [], [], [], []
            for b in range(pose.shape[0]):
                ln = int(lengths[b])
                pn, cn, vn = add_noise(pose[b, :ln], conf[b, :ln], valid[b, :ln], ycfg,
                                        jitter_std=args.jitter_std, scale=scale, rng=rng)
                pose_n_list.append(pn); conf_n_list.append(cn); valid_n_list.append(vn)
                len_n.append(pn.shape[0])
            T_n = max(len_n)
            B, J3 = pose.shape[0], J
            pose_n  = pose.new_zeros(B, T_n, J3, 3)
            conf_n  = conf.new_zeros(B, T_n, J3)
            valid_n = torch.zeros(B, T_n, J3, dtype=torch.bool, device=device)
            t_n     = t.new_zeros(B, T_n)
            lengths_n = torch.tensor(len_n, device=device)
            for b in range(B):
                ln = len_n[b]
                pose_n[b, :ln]  = pose_n_list[b]
                conf_n[b, :ln]  = conf_n_list[b]
                valid_n[b, :ln] = valid_n_list[b]
                t_n[b, :ln]     = torch.arange(ln, dtype=t.dtype, device=device) / 30.0
                t_n[b, ln:]     = t_n[b, ln - 1]

            # -- teacher (clean, no grad) ---------------------------------------
            with torch.no_grad():
                out_t = teacher(pose, conf, valid, t, lengths)

            # -- student (noisy, gradients) ------------------------------------
            out_s = student(pose_n, conf_n, valid_n, t_n, lengths_n)

            # -- losses --------------------------------------------------------
            yap_t, yap_s = out_t["yap"], out_s["yap"]
            # presence KL (student must agree with teacher on when gesture happens)
            # Both have shape (B, T_clean) and (B, T_noisy); use min length
            T_min = min(yap_t["presence"].shape[1], yap_s["presence"].shape[1])
            l_pres = presence_kl(
                yap_t["presence"][:, :T_min],
                yap_s["presence"][:, :T_min],
                lengths.clamp_max(T_min))
            # end-head KL
            l_end = presence_kl(
                yap_t["end"][:, :T_min],
                yap_s["end"][:, :T_min],
                lengths.clamp_max(T_min))
            # entity consistency (MSE after soft alignment)
            ent_t = out_t.get("entity_raw")
            ent_s = out_s.get("entity_raw")
            l_ent = (entity_consistency(out_t["word_logits"].view(-1, num_classes + 1),
                                        out_s["word_logits"].view(-1, num_classes + 1))
                     if out_t["token_mask"].sum() > 0 and out_s["token_mask"].sum() > 0
                     else torch.tensor(0.0, device=device))
            # word classification on clean path (anti-collapse + keeps both correct)
            if out_t["token_mask"].sum() > 0:
                wl = out_t["word_logits"]                        # (B, K, V+1)
                # pick the first valid entity per batch row
                first_k = out_t["token_mask"].float().argmax(dim=1)  # (B,)
                ent_vec_clean = wl[torch.arange(B, device=device), first_k]  # (B, V+1)
                l_cls = F.cross_entropy(ent_vec_clean[:, 1:], labels)         # skip blank col
                # collect predictions for epoch accuracy / F1
                preds = ent_vec_clean[:, 1:].argmax(-1)
                all_preds.append(preds.detach().cpu())
                all_labels.append(labels.detach().cpu())
            else:
                l_cls = torch.tensor(0.0, device=device)

            loss = l_pres + l_end + args.ent_w * l_ent + args.cls_w * l_cls

            # ---- anti-blank: every clip has exactly 1 word, so presence
            #        mass shouldn't collapse to ~0 (all blank / no gesture) ----
            presence_mean = yap_t["presence"].mean()
            l_blank = torch.relu(args.blank_floor - presence_mean)

            # ---- anti-collapse: word predictions across the batch should
            #        not all be the same class (local-minimum collapse) ----
            if out_t["token_mask"].sum() > 0:
                wl = out_t["word_logits"]                      # (B, K, V+1)
                probs = wl.softmax(-1)[:, :, 1:]               # drop blank col -> (B, K, V)
                # take the max-prob non-blank class per entity slot, then per batch row
                pred_class = probs.argmax(-1)                  # (B, K)
                # variance of predicted classes across batch (per slot), averaged
                l_div = -(pred_class.float().std(dim=0).mean())  # negative = want high std
            else:
                l_div = torch.tensor(0.0, device=device)

            loss = loss + args.blank_w * l_blank + args.div_w * l_div
            if torch.isfinite(loss):
                # The gradient is ACCUMULATED over --grad_accum micro-batches:
                # the effective batch stays the same while the memory stays low.
                (loss / args.grad_accum).backward()
                micro += 1
                if micro % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    steps_done += 1
                    # EMA teacher update (exponential moving average of student)
                    with torch.no_grad():
                        for tp, sp in zip(teacher.parameters(), student.parameters()):
                            tp.data.mul_(args.ema_decay).add_(sp.data, alpha=1 - args.ema_decay)
                ep_loss += loss.item(); n_ok += 1
                pbar.set_postfix(loss=f"{ep_loss/max(1,n_ok):.4f}",
                                 lr=f"{scheduler.get_last_lr()[0]:.2e}")

        pbar.close()

        if micro % args.grad_accum != 0 and steps_done > 0:
            # flush the last partial accumulation
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps_done += 1
            with torch.no_grad():
                for tp, sp in zip(teacher.parameters(), student.parameters()):
                    tp.data.mul_(args.ema_decay).add_(sp.data, alpha=1 - args.ema_decay)

        if device.type == "cuda":
            mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"    [MEM] peak GPU memory {mem:.2f} GB (batch {args.batch_size} x accum {args.grad_accum})")
        scheduler.step()
        # ---- epoch metrics: true (non-negative) task loss, acc, f1 ----
        pure_task_loss = float(l_pres + l_end + args.ent_w * l_ent + args.cls_w * l_cls)
        if all_preds:
            all_preds_t = torch.cat(all_preds)
            all_labels_t = torch.cat(all_labels)
            epoch_acc = float((all_preds_t == all_labels_t).float().mean().item())
            epoch_f1 = _macro_f1(all_preds_t.numpy(), all_labels_t.numpy(), num_classes)
        else:
            epoch_acc = 0.0
            epoch_f1 = 0.0

        composite_loss = ep_loss / max(1, n_ok)
        # strip out the negative regularizers for the "true" non-negative task loss
        true_loss = composite_loss - args.blank_w * float(l_blank) - args.div_w * float(l_div)

        history["train_loss"].append(composite_loss)
        history["true_loss"].append(true_loss)
        history["train_acc"].append(epoch_acc)
        history["train_f1"].append(epoch_f1)
        history["lr"].append(scheduler.get_last_lr()[0])
        history["blank"].append(float(l_blank))
        history["div"].append(float(l_div))

        print(f"[Ep {ep+1:02d}/{args.epochs}] "
              f"LOSS {true_loss:.4f} | ACC {epoch_acc*100:.2f}% | F1 {epoch_f1:.4f} | "
              f"lr {scheduler.get_last_lr()[0]:.2e} | {steps_done} steps "
              f"(batch {args.batch_size})")
        if composite_loss != true_loss:
            print(f"     (composite={composite_loss:.4f} | blank_pen={args.blank_w*float(l_blank):.4f} | div_pen={args.div_w*float(l_div):.4f})")

        # -- live plots after every epoch --
        plot_training_curves(
            history, args.out_dir,
            title="Stage 1 - noise filtration (YAPpingConformer)",
            tag="stage1_",
        )
        write_report(
            history, args.out_dir,
            title="Stage 1 - noise filtration (YAPpingConformer)",
            prefix="stage1_",
        )
        # -- refresh GPU memory each epoch: clear unusable / stale tensors --
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if (ep + 1) % 5 == 0 or ep + 1 == args.epochs:
            # Save checkpoint
            torch.save({"epoch": ep + 1, "student": student.state_dict(),
                        "teacher": teacher.state_dict(), "cls_head": cls_head.state_dict()},
                       os.path.join(args.out_dir, "yap_noise_ckpt.pth"))
            if ep_loss < best_loss:
                best_loss = ep_loss
                torch.save(student.state_dict(), os.path.join(args.out_dir, "yap_noise_robust.pth"))
                print(f"  * new best -> yap_noise_robust.pth")

    # ---- training history ----------------------------------------
    import json as _json
    with open(os.path.join(args.out_dir, "history.json"), "w", encoding="utf-8") as _f:
        _json.dump(history, _f, indent=2)
    print(f"\nDone. Robust YAP saved to {os.path.join(args.out_dir, 'yap_noise_robust.pth')}")


def parse_args():
    p = argparse.ArgumentParser(description="YAP Stage 1: Noise Filtration Training")
    p.add_argument("--data_dirs", nargs="+", default=DEFAULT_DATA_DIRS)
    p.add_argument("--classes",   default=CLASSES_PATH)
    p.add_argument("--out_dir",   default=DEFAULT_OUT)
    p.add_argument("--init",      default=None, help="Pretrained YAPpingConformer weights")
    p.add_argument("--resume",    action="store_true")
    p.add_argument("--epochs",    type=int,   default=30)
    p.add_argument("--batch_size", type=int,   default=4,
                   help="Micro-batch size.  The YAP stage 1 runs the teacher AND "
                        "the student at the same time, so the memory is about twice "
                        "a normal forward pass: keep it low (4) and raise "
                        "--grad_accum to keep the effective batch.")
    p.add_argument("--grad_accum",  type=int,   default=2,
                   help="Accumulate N micro-batches before the optimiser steps "
                        "(the effective batch = batch_size * grad_accum)")
    p.add_argument("--layout",         type=str,   default="corpus",
                   choices=["corpus", "holistic", "hands"],
                   help="The body layout: corpus (J=53: the hands + the 2 "
                        "shoulders + the 2 elbows + the 7 face points), holistic "
                        "(J=79) or hands only (J=42)")
    p.add_argument("--num_workers",  type=int,   default=2, help="DataLoader workers: they load and augment the clips in parallel")
    # --- the noise: its intensity is drawn at RANDOM for every batch -------- #
    p.add_argument("--jitter_std",       type=float, default=0.015,
                   help="Base Gaussian jitter of the coordinates (in the normalised units)")
    p.add_argument("--noise_scale_min",  type=float, default=0.0,
                   help="Lowest noise intensity drawn per batch (0 = a clean clip)")
    p.add_argument("--noise_scale_max",  type=float, default=1.0,
                   help="Highest noise intensity drawn per batch")
    p.add_argument("--noise_prob",       type=float, default=0.9,
                   help="Probability that a batch gets any noise at all")
    # --- the canonical augmentation ---------------------------------------- #
    p.add_argument("--aug",              type=int, default=1,
                   help="Apply the canonical augmentation (mirror / tilt / shift / speed) "
                        "to the clean clip before the teacher/student split")
    p.add_argument("--aug_strength",     type=float, default=1.0,
                   help="Scales the magnitude of the augmentation")
    p.add_argument("--lr",        type=float, default=3e-4)
    p.add_argument("--ema_decay", type=float, default=0.99, help="Teacher EMA decay")
    p.add_argument("--cls_w",     type=float, default=1.0,  help="Word-label loss weight")
    p.add_argument("--ent_w",     type=float, default=0.5,  help="Entity consistency weight")
    # ---- anti-blank / anti-collapse knobs ----
    p.add_argument("--blank_w",   type=float, default=1.0,  help="Weight on presence-mass anti-blank penalty")
    p.add_argument("--blank_floor",type=float, default=0.15, help="Minimum mean presence per clip (anti-blank floor)")
    p.add_argument("--div_w",     type=float, default=0.5,  help="Weight on prediction-diversity anti-collapse term")
    # ---- simulated cosine annealing (SGDR warm restarts) ----
    p.add_argument("--t0",        type=int,   default=5,    help="First restart period (epochs) for cosine warm restarts")
    p.add_argument("--t_mult",    type=int,   default=2,    help="Multiply restart period by this after each restart")
    p.add_argument("--max_len",   type=int,   default=400)
    p.add_argument("--max_per_class", type=int, default=None)
    p.add_argument("--seed",      type=int,   default=42)
    # YAP architecture
    p.add_argument("--d_model",         type=int, default=128)
    p.add_argument("--n_mem_layers",    type=int, default=3)
    p.add_argument("--summary_channels",type=int, default=32)
    p.add_argument("--hippo_order",     type=int, default=16)
    p.add_argument("--entity_dim",      type=int, default=256)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
