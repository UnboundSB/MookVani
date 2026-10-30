"""
YAP Boundary/End Trainer — end head only.

Trains YAP's frame-level end head to answer: "does a gesture end right here?"
— with NOTHING else trainable.

Data mix (every batch):
  • Word clips from tensors_include50_word_level / tensors_isl_csltr_word_level
    (the 228 classes from classes.json, max 20 per class).  End target = Gaussian
    bump centred on the last frame of the clip — the sign ends there.
  • Random junk generated ON THE FLY: 60–180 frames of resting hands with tremor.
    End target = 0 everywhere — no gesture ends in junk.

Everything in yap_conformer.py is FROZEN except yap.core.end_head.
"""
from __future__ import annotations

import argparse, json, os, random, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))  # Backend/

from yap_conformer import (
    YAPConfig, YAPpingConformer, ConformerConfig,
    end_targets, focal_presence_loss,
)
from utils.segmentor_utils import create_yap_model, SegmentorModelCfg, load_yap_weights

BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CLASSES = BACKEND_DIR / "models" / "word_vocabulary" / "classes.json"
DEFAULT_WORD_DIRS = [
    BACKEND_DIR / "data" / "tensors_include50_word_level",
    BACKEND_DIR / "data" / "tensors_isl_csltr_word_level",
]
DEFAULT_OUT = BACKEND_DIR / "segmentor" / "models" / "yap_boundary_only"

# 163-dim layout offsets (53 joints x 3 = 159 coords + 4 flags)
HAND_DIM = 63
BODY_START = 126
BODY_DIM = 12
FACE_START = 138
FACE_DIM = 21
FLAGS_START = 159
FLAGS_DIM = 4
TOTAL_DIM = 163

rng_np = np.random.default_rng(42)


def make_junk_clip(t: int, rng: np.random.Generator = None) -> np.ndarray:
    """One junk clip: (T, 163) — realistic non-sign poses.

    Hands resting near torso with subtle tremor, body/face still, flags=0.
    No gesture ends here — end target = 0 everywhere.
    """
    if rng is None:
        rng = rng_np
    clip = np.zeros((t, TOTAL_DIM), dtype=np.float32)

    # --- Hands: resting near waist with subtle tremor -----------------------
    for h in range(2):
        base = h * HAND_DIM
        hand = np.zeros((t, 21, 3), dtype=np.float32)

        rest_x = -0.30 if h == 0 else 0.30
        rest_y = 0.55
        rest_z = -0.40

        wrist = np.zeros((t, 3), dtype=np.float32)
        wrist[:, 0] = rest_x + rng.normal(0, 0.02, size=t)
        wrist[:, 1] = rest_y + rng.normal(0, 0.015, size=t)
        wrist[:, 2] = rest_z + rng.normal(0, 0.01, size=t)

        for j in range(1, 21):
            offset = rng.normal(0, 0.03, size=(t, 3))
            hand[:, j, :] = wrist + offset

        hand[:, 0, :] = wrist
        clip[:, base:base + 63] = hand.reshape(t, 63)

    # --- Body: still --------------------------------------------------------
    body = np.zeros((t, 4, 3), dtype=np.float32)
    body[:, 0, :] = [-0.25, 0.30, -0.35]
    body[:, 1, :] = [0.25, 0.30, -0.35]
    body[:, 2, :] = [-0.20, 0.65, -0.30]
    body[:, 3, :] = [0.20, 0.65, -0.30]
    body += rng.normal(0, 0.005, size=(t, 4, 3)).astype(np.float32)
    clip[:, BODY_START:BODY_START + BODY_DIM] = body.reshape(t, 12)

    # --- Face: still --------------------------------------------------------
    face = np.zeros((t, 7, 3), dtype=np.float32)
    face[:, 0, :] = [0.0, 0.55, -0.25]
    face[:, 1, :] = [-0.05, 0.50, -0.20]
    face[:, 2, :] = [0.05, 0.50, -0.20]
    face[:, 3, :] = [-0.08, 0.45, -0.15]
    face[:, 4, :] = [0.08, 0.45, -0.15]
    face[:, 5, :] = [0.0, 0.40, -0.10]
    face[:, 6, :] = [0.0, 0.62, -0.30]
    face += rng.normal(0, 0.005, size=(t, 7, 3)).astype(np.float32)
    clip[:, FACE_START:FACE_START + FACE_DIM] = face.reshape(t, 21)

    clip[:, FLAGS_START:FLAGS_START + FLAGS_DIM] = 0.0

    return clip


class BoundaryDataset(Dataset):
    """
    Mixes word clips (end=1 at last frame) with on-the-fly junk (end=0 everywhere).
    Returns (pose, conf, valid, label, t, end_target, is_word).
    """

    def __init__(self, word_clips: list[dict], num_joints: int,
                 max_len: int = 400, junk_ratio: float = 0.5, seed: int = 42,
                 junk_rng: np.random.Generator = None):
        self.word_clips = word_clips
        self.J = num_joints
        self.max_len = max_len
        self.junk_ratio = junk_ratio
        self.rng = random.Random(seed)
        self.junk_rng = junk_rng if junk_rng is not None else np.random.default_rng(seed)

        self.items = [("w", i) for i in range(len(word_clips))]
        self.rng.shuffle(self.items)

    def __len__(self):
        return len(self.items) * 2

    def __getitem__(self, idx):
        if idx < len(self.items):
            kind, i = self.items[idx]
            return self._get_word(i)
        else:
            return self._get_junk()

    def _get_word(self, idx):
        item = self.word_clips[idx]
        try:
            raw = torch.load(item["path"], weights_only=True).float().numpy()
        except Exception:
            data = np.load(item["path"])
            if isinstance(data, np.lib.npyio.NpzFile):
                raw = data[data.files[0]].astype(np.float32)
            else:
                raw = data.astype(np.float32)
        T = min(len(raw), self.max_len)
        raw = raw[:T, : self.J * 3]
        pose = torch.from_numpy(raw).view(T, self.J, 3)
        conf = torch.ones(T, self.J)
        valid = torch.ones(T, self.J, dtype=torch.bool)
        t = torch.arange(T, dtype=torch.float32) / 30.0

        # End target: Gaussian bump at the last frame — the sign ends there
        spans = [(0.0, (T - 1) / 30.0)]
        end_tgt = end_targets(t, spans, sigma_s=0.05)
        return pose, conf, valid, item["label"], t, end_tgt, True

    def _get_junk(self):
        T = int(self.rng.randint(60, 180))
        raw = make_junk_clip(T, self.junk_rng)
        raw = raw[:, : self.J * 3]
        pose = torch.from_numpy(raw).view(T, self.J, 3)
        conf = torch.ones(T, self.J)
        valid = torch.ones(T, self.J, dtype=torch.bool)
        t = torch.arange(T, dtype=torch.float32) / 30.0
        end_tgt = torch.zeros(T)  # no gesture ends in junk
        return pose, conf, valid, -1, t, end_tgt, False


def collate(batch):
    poses, confs, valids, labels, ts, end_tgts, is_words = zip(*batch)
    B = len(poses)
    T = max(p.shape[0] for p in poses)
    J = poses[0].shape[1]
    pose_b = torch.zeros(B, T, J, 3)
    conf_b = torch.zeros(B, T, J)
    valid_b = torch.zeros(B, T, J, dtype=torch.bool)
    t_b = torch.zeros(B, T)
    end_b = torch.zeros(B, T)
    mask_b = torch.zeros(B, T, dtype=bool)
    label_b = torch.full((B,), -1, dtype=torch.long)
    lengths = torch.zeros(B, dtype=torch.long)
    for i, (p, c, v, l, t_, et, iw, length) in enumerate(zip(
            poses, confs, valids, labels, ts, end_tgts, is_words,
            [p.shape[0] for p in poses])):
        n = p.shape[0]
        pose_b[i, :n] = p
        conf_b[i, :n] = c
        valid_b[i, :n] = v
        t_b[i, :n] = t_
        t_b[i, n:] = t_[-1]
        end_b[i, :n] = et
        mask_b[i, :n] = True
        label_b[i] = l
        lengths[i] = n
    return pose_b, conf_b, valid_b, label_b, t_b, end_b, mask_b, lengths


def load_word_clips(data_dirs, classes_path, max_per_class=20):
    """Scan word tensor dirs for .pt clips, capped at max_per_class per class."""
    with open(classes_path) as f:
        classes = json.load(f)
    label_map = {c: i for i, c in enumerate(classes)}

    clips = []
    for d in data_dirs:
        d = Path(d)
        if not d.exists():
            continue
        for pt in d.rglob("*.pt"):
            lbl = pt.parent.name
            if lbl in label_map:
                clips.append({"path": str(pt), "label": label_map[lbl]})

    if max_per_class:
        by_cls = defaultdict(list)
        for c in clips:
            by_cls[c["label"]].append(c)
        clips = []
        for items in by_cls.values():
            random.shuffle(items)
            clips.extend(items[:max_per_class])

    random.shuffle(clips)
    return clips


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.classes, encoding="utf-8") as f:
        classes = json.load(f)
    num_classes = len(classes)
    print(f"[INFO] {num_classes} classes from classes.json")

    # ---- word clips -------------------------------------------------------
    word_clips = load_word_clips(args.data_dirs, args.classes, args.max_per_class)
    print(f"[INFO] {len(word_clips)} word clips loaded (max {args.max_per_class} per class)")
    if not word_clips:
        raise RuntimeError("No word clips found! Check data_dirs.")

    # ---- stratified split: 20% of each class to validation ----------------
    by_cls = defaultdict(list)
    for c in word_clips:
        by_cls[c["label"]].append(c)

    train_clips = []
    val_clips = []
    for label, items in by_cls.items():
        random.shuffle(items)
        n_val = max(1, int(len(items) * 0.2))
        val_clips.extend(items[:n_val])
        train_clips.extend(items[n_val:])

    random.shuffle(train_clips)
    random.shuffle(val_clips)
    print(f"[DATA] Train: {len(train_clips)} clips | Val: {len(val_clips)} clips "
          f"(20% per class held out)")

    # ---- datasets / loaders ----------------------------------------------
    train_ds = BoundaryDataset(train_clips, args.joints,
                               max_len=args.max_len,
                               junk_ratio=args.junk_ratio,
                               seed=args.seed,
                               junk_rng=np.random.default_rng(args.seed))
    val_ds = BoundaryDataset(val_clips, args.joints,
                             max_len=args.max_len,
                             junk_ratio=args.junk_ratio,
                             seed=args.seed + 1,
                             junk_rng=np.random.default_rng(args.seed + 1000))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=args.num_workers)

    # ---- model ------------------------------------------------------------
    sm_cfg = SegmentorModelCfg(
        layout=args.layout, d_model=args.d_model, n_mem_layers=args.n_mem_layers,
        summary_channels=args.summary_channels, hippo_order=args.hippo_order,
        entity_dim=args.entity_dim, num_words=num_classes,
        conformer_d_model=args.d_model, conformer_heads=4, conformer_layers=4,
    )
    model = create_yap_model(sm_cfg, device)

    if args.init and os.path.exists(args.init):
        load_yap_weights(model, args.init, label="INIT")
        print(f"[INIT] loaded weights from {args.init}")

    # ---- FREEZE everything except end head --------------------------------
    for p in model.parameters():
        p.requires_grad = False
    for p in model.yap.core.end_head.parameters():
        p.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[FREEZE] Trainable: {n_trainable:,} / {n_total:,} "
          f"({100.0 * n_trainable / n_total:.2f}%) — end head only")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.out_dir, exist_ok=True)
    best_loss = float("inf")
    history = {"train_loss": [], "true_loss": [], "lr": []}

    print("=" * 72)
    print("YAP Boundary Trainer — end head ONLY, everything else frozen")
    print("=" * 72)
    print(f"  device      : {device}")
    print(f"  word clips  : {len(word_clips)}")
    print(f"  epochs      : {args.epochs}  batch={args.batch_size}")
    print(f"  junk ratio  : {args.junk_ratio} (junk frames generated on the fly)")
    print(f"  out_dir     : {args.out_dir}")
    print("=" * 72)

    for ep in range(args.epochs):
        # ---- re-seed junk generator each epoch --------------------------
        train_ds.junk_rng = np.random.default_rng(args.seed + ep)
        val_ds.junk_rng = np.random.default_rng(args.seed + ep + 5000)

        # ===== TRAIN =====
        model.train()
        ep_loss = 0.0
        n_ok = 0

        pbar = tqdm(train_loader, desc=f"Ep {ep+1:02d}/{args.epochs} [train]", leave=False)
        for pose, conf, valid, labels, t, end_target, mask, lengths in pbar:
            pose, conf, valid, labels, t, end_target, mask, lengths = (
                x.to(device) for x in (pose, conf, valid, labels, t, end_target, mask, lengths))

            optimizer.zero_grad(set_to_none=True)
            out = model.yap.forward_frames(pose, conf, valid, t)
            end_logit = out["end"]  # (B, T)

            pmask = mask.float()
            T_min = min(end_logit.shape[1], end_target.shape[1])
            l_end = focal_presence_loss(
                end_logit[:, :T_min], end_target[:, :T_min], pmask[:, :T_min])

            loss = l_end
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ep_loss += loss.item()
                n_ok += 1
                pbar.set_postfix(end=f"{l_end.item():.4f}")

        pbar.close()
        train_avg = ep_loss / max(1, n_ok)

        # ===== VALIDATION =====
        model.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for pose, conf, valid, labels, t, end_target, mask, lengths in val_loader:
                pose, conf, valid, labels, t, end_target, mask, lengths = (
                    x.to(device) for x in (pose, conf, valid, labels, t, end_target, mask, lengths))

                out = model.yap.forward_frames(pose, conf, valid, t)
                end_logit = out["end"]

                pmask = mask.float()
                T_min = min(end_logit.shape[1], end_target.shape[1])
                v_loss = focal_presence_loss(
                    end_logit[:, :T_min], end_target[:, :T_min], pmask[:, :T_min])

                if torch.isfinite(v_loss):
                    val_loss += v_loss.item()
                    val_n += 1

        val_avg = val_loss / max(1, val_n)

        scheduler.step()
        print(f"[Ep {ep+1:02d}/{args.epochs}] "
              f"train_end={train_avg:.4f} val_end={val_avg:.4f} | "
              f"lr {scheduler.get_last_lr()[0]:.2e} | {n_ok} steps "
              f"(batch {args.batch_size})")

        history["train_loss"].append(train_avg)
        history["true_loss"].append(train_avg)
        history["lr"].append(scheduler.get_last_lr()[0])

        if val_avg < best_loss:
            best_loss = val_avg
            torch.save(model.state_dict(), os.path.join(args.out_dir, "yap_end_best.pth"))
            print(f"  * new best (val_end={val_avg:.4f}) -> yap_end_best.pth")

    print(f"\nDone. Best end model saved to {args.out_dir}/yap_end_best.pth")


def parse_args():
    p = argparse.ArgumentParser(description="YAP Boundary Trainer — end head only")
    p.add_argument("--data_dirs", nargs="+",
                   default=[str(d) for d in DEFAULT_WORD_DIRS])
    p.add_argument("--classes", default=str(DEFAULT_CLASSES))
    p.add_argument("--out_dir", default=str(DEFAULT_OUT))
    p.add_argument("--init", default=None, help="Pretrained YAPpingConformer weights")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--junk_ratio", type=float, default=0.5,
                   help="Fraction of junk items in the dataset (0..1)")
    p.add_argument("--max_per_class", type=int, default=20,
                   help="Max clips per word class (matches old pipeline)")
    p.add_argument("--max_len", type=int, default=400)
    p.add_argument("--joints", type=int, default=53,
                   help="53 = corpus layout (hands+body+face)")
    p.add_argument("--layout", type=str, default="corpus",
                   choices=["corpus", "holistic", "hands"])
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_mem_layers", type=int, default=3)
    p.add_argument("--summary_channels", type=int, default=32)
    p.add_argument("--hippo_order", type=int, default=16)
    p.add_argument("--entity_dim", type=int, default=256)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
