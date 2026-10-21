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

import argparse, json, os, random, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ── local imports (run from the repo root or add segmentor to sys.path) ─────
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))  # Backend/

from yap_conformer import (
    YAPConfig, YAPCore, YAP, YAPpingConformer, ConformerConfig,
    focal_presence_loss, layer_conf_loss, end_targets, presence_targets,
    make_segments, frame_dts,
)
from noise_aug import add_noise

# ── defaults ─────────────────────────────────────────────────────────────────
DEFAULT_DATA_DIRS = [
    "Backend/data/tensors_include50_word_level",
    "Backend/data/tensors_isl_csltr_word_level",
]
DEFAULT_OUT = "Backend/segmentor/models/yap_stage1"
CLASSES_PATH = "Backend/data/classes.json"


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

    # ── model config ──────────────────────────────────────────────────────────
    ycfg = YAPConfig(d_model=args.d_model, n_mem_layers=args.n_mem_layers,
                     summary_channels=args.summary_channels, hippo_order=args.hippo_order,
                     entity_dim=args.entity_dim)
    J = ycfg.total_joints

    # ── data ──────────────────────────────────────────────────────────────────
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
    train_ld = DataLoader(train_ds, args.batch_size, shuffle=True,  collate_fn=_collate, num_workers=0)
    val_ld   = DataLoader(val_ds,   args.batch_size, shuffle=False, collate_fn=_collate, num_workers=0)

    # ── two model instances in memory ─────────────────────────────────────────
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
        student.load_state_dict(sd, strict=False)
        teacher.load_state_dict(sd, strict=False)
        print(f"[INIT] loaded weights from {args.init}")
    if args.resume and os.path.exists(os.path.join(args.out_dir, "yap_noise_ckpt.pth")):
        ck = torch.load(os.path.join(args.out_dir, "yap_noise_ckpt.pth"), map_location=device, weights_only=False)
        student.load_state_dict(ck["student"])
        teacher.load_state_dict(ck["teacher"])
        cls_head.load_state_dict(ck["cls_head"])
        print(f"[RESUME] epoch {ck['epoch']}")

    optimizer = torch.optim.AdamW(list(student.parameters()) + list(cls_head.parameters()),
                                  lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    os.makedirs(args.out_dir, exist_ok=True)
    best_loss = float("inf")
    rng = random.Random(args.seed)

    for ep in range(args.epochs):
        student.train(); teacher.eval(); cls_head.train()
        ep_loss = 0.0; n_ok = 0

        for pose, conf, valid, labels, t, lengths in train_ld:
            pose, conf, valid, labels, t, lengths = (
                pose.to(device), conf.to(device), valid.to(device),
                labels.to(device), t.to(device), lengths.to(device))

            # ── build noisy batch ─────────────────────────────────────────────
            pose_n_list, conf_n_list, valid_n_list, len_n = [], [], [], []
            for b in range(pose.shape[0]):
                ln = int(lengths[b])
                pn, cn, vn = add_noise(pose[b, :ln], conf[b, :ln], valid[b, :ln], ycfg, rng=rng)
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

            # ── teacher (clean, no grad) ───────────────────────────────────────
            with torch.no_grad():
                out_t = teacher(pose, conf, valid, t, lengths)

            # ── student (noisy, gradients) ────────────────────────────────────
            out_s = student(pose_n, conf_n, valid_n, t_n, lengths_n)

            # ── losses ────────────────────────────────────────────────────────
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
            else:
                l_cls = torch.tensor(0.0, device=device)

            loss = l_pres + l_end + args.ent_w * l_ent + args.cls_w * l_cls
            if torch.isfinite(loss):
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                # EMA teacher update (exponential moving average of student)
                with torch.no_grad():
                    for tp, sp in zip(teacher.parameters(), student.parameters()):
                        tp.data.mul_(args.ema_decay).add_(sp.data, alpha=1 - args.ema_decay)
                ep_loss += loss.item(); n_ok += 1

        scheduler.step()
        print(f"[Ep {ep+1:02d}/{args.epochs}] loss {ep_loss/max(1,n_ok):.4f} "
              f"| lr {scheduler.get_last_lr()[0]:.2e}")

        if (ep + 1) % 5 == 0 or ep + 1 == args.epochs:
            # Save checkpoint
            torch.save({"epoch": ep + 1, "student": student.state_dict(),
                        "teacher": teacher.state_dict(), "cls_head": cls_head.state_dict()},
                       os.path.join(args.out_dir, "yap_noise_ckpt.pth"))
            if ep_loss < best_loss:
                best_loss = ep_loss
                torch.save(student.state_dict(), os.path.join(args.out_dir, "yap_noise_robust.pth"))
                print(f"  ★ new best -> yap_noise_robust.pth")

    print(f"\nDone. Robust YAP saved to {os.path.join(args.out_dir, 'yap_noise_robust.pth')}")


def parse_args():
    p = argparse.ArgumentParser(description="YAP Stage 1: Noise Filtration Training")
    p.add_argument("--data_dirs", nargs="+", default=DEFAULT_DATA_DIRS)
    p.add_argument("--classes",   default=CLASSES_PATH)
    p.add_argument("--out_dir",   default=DEFAULT_OUT)
    p.add_argument("--init",      default=None, help="Pretrained YAPpingConformer weights")
    p.add_argument("--resume",    action="store_true")
    p.add_argument("--epochs",    type=int,   default=30)
    p.add_argument("--batch_size",type=int,   default=8)
    p.add_argument("--lr",        type=float, default=3e-4)
    p.add_argument("--ema_decay", type=float, default=0.99, help="Teacher EMA decay")
    p.add_argument("--cls_w",     type=float, default=1.0,  help="Word-label loss weight")
    p.add_argument("--ent_w",     type=float, default=0.5,  help="Entity consistency weight")
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
