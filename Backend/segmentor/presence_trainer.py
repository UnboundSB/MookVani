"""
YAP Presence Trainer — standalone, presence head only.

Trains YAP's frame-level presence head to answer: "is this a real ISL word sign
or just junk?" — with NOTHING else trainable.

Data mix (every batch):
  • Word clips from tensors_include50_word_level / tensors_isl_csltr_word_level
    (the 228 classes from classes.json, max 20 per class).  Presence = 1 for
    every frame — the whole clip is one sign.
  • Random junk generated ON THE FLY: 60–180 frames of pure garbage coordinates
    (random hand jitter, still body/face, all presence flags = 0).  Presence =
    0 everywhere — no sign at all.

Everything in yap_conformer.py is FROZEN except yap.core.presence_head.
"""
from __future__ import annotations

import argparse, json, os, random, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))  # Backend/

from yap_conformer import (
    YAPConfig, YAPpingConformer, ConformerConfig,
    presence_targets, focal_presence_loss,
)
from utils.segmentor_utils import create_yap_model, SegmentorModelCfg, load_yap_weights

BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CLASSES = BACKEND_DIR / "models" / "word_vocabulary" / "classes.json"
DEFAULT_WORD_DIRS = [
    BACKEND_DIR / "data" / "tensors_include50_word_level",
    BACKEND_DIR / "data" / "tensors_isl_csltr_word_level",
]
DEFAULT_OUT = BACKEND_DIR / "segmentor" / "models" / "yap_presence_only"

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


def make_junk_clip(t: int) -> np.ndarray:
    """One junk clip: (T, 163) — random hand jitter, still body/face, flags=0."""
    clip = np.zeros((t, TOTAL_DIM), dtype=np.float32)

    # Hands: random walk / jitter (simulates tracker noise / non-sign movement)
    for h in range(2):
        base = h * HAND_DIM
        hand = np.zeros((t, 21, 3), dtype=np.float32)
        hand[:, 0, :] = rng_np.uniform(-0.3, 0.3, size=(t, 3))  # wrist
        for j in range(1, 21):
            hand[:, j, :] = hand[:, 0, :] + rng_np.normal(0, 0.05, size=(t, 3))
        clip[:, base:base + 63] = hand.reshape(t, 63)

    # Body: still
    clip[:, BODY_START:BODY_START + BODY_DIM] = rng_np.normal(
        0, 0.01, size=(t, 4, 3)).astype(np.float32).reshape(t, 12)

    # Face: still
    clip[:, FACE_START:FACE_START + FACE_DIM] = rng_np.normal(
        0, 0.01, size=(t, 7, 3)).astype(np.float32).reshape(t, 21)

    # Presence flags: all 0 (no meaningful detection)
    clip[:, FLAGS_START:FLAGS_START + FLAGS_DIM] = 0.0

    return clip


class PresenceDataset(Dataset):
    """
    Mixes word clips (presence=1 everywhere) with on-the-fly junk (presence=0
    everywhere).  Returns (pose, conf, valid, label, t, presence_tgt, is_word).
    """

    def __init__(self, word_clips: list[dict], num_joints: int,
                 max_len: int = 400, junk_ratio: float = 0.5, seed: int = 42):
        self.word_clips = word_clips
        self.J = num_joints
        self.max_len = max_len
        self.junk_ratio = junk_ratio
        self.rng = random.Random(seed)

        # Balanced item list
        self.items = [("w", i) for i in range(len(word_clips))]
        self.rng.shuffle(self.items)

    def __len__(self):
        return len(self.items) * 2  # word items + junk items

    def __getitem__(self, idx):
        if idx < len(self.items):
            kind, i = self.items[idx]
            return self._get_word(i)
        else:
            return self._get_junk()

    def _get_word(self, idx):
        item = self.word_clips[idx]
        raw = np.load(item["path"]).astype(np.float32)  # (T, 163)
        T = min(len(raw), self.max_len)
        raw = raw[:T, : self.J * 3]  # drop the 4 flags
        pose = torch.from_numpy(raw).view(T, self.J, 3)
        conf = torch.ones(T, self.J)
        valid = torch.ones(T, self.J, dtype=torch.bool)
        t = torch.arange(T, dtype=torch.float32) / 30.0
        presence_tgt = torch.ones(T)  # whole clip is one sign
        return pose, conf, valid, item["label"], t, presence_tgt, True

    def _get_junk(self):
        T = int(self.rng.integers(60, 181))  # 60–180 frames
        raw = make_junk_clip(T)
        raw = raw[:, : self.J * 3]
        pose = torch.from_numpy(raw).view(T, self.J, 3)
        conf = torch.ones(T, self.J)
        valid = torch.ones(T, self.J, dtype=torch.bool)
        t = torch.arange(T, dtype=torch.float32) / 30.0
        presence_tgt = torch.zeros(T)  # no sign at all
        return pose, conf, valid, -1, t, presence_tgt, False


def collate(batch):
    poses, confs, valids, labels, ts, pres_tgts, is_words = zip(*batch)
    B = len(poses)
    T = max(p.shape[0] for p in poses)
    J = poses[0].shape[1]
    pose_b = torch.zeros(B, T, J, 3)
    conf_b = torch.zeros(B, T, J)
    valid_b = torch.zeros(B, T, J, dtype=torch.bool)
    t_b = torch.zeros(B, T)
    pres_b = torch.zeros(B, T)
    mask_b = torch.zeros(B, T, dtype=bool)
    label_b = torch.full((B,), -1, dtype=torch.long)
    lengths = torch.zeros(B, dtype=torch.long)
    for i, (p, c, v, l, t_, pt, iw, length) in enumerate(zip(
            poses, confs, valids, labels, ts, pres_tgts, is_words,
            [p.shape[0] for p in poses])):
        n = p.shape[0]
        pose_b[i, :n] = p
        conf_b[i, :n] = c
        valid_b[i, :n] = v
        t_b[i, :n] = t_
        t_b[i, n:] = t_[-1]
        pres_b[i, :n] = pt
        mask_b[i, :n] = True
        label_b[i] = l
        lengths[i] = n
    return pose_b, conf_b, valid_b, label_b, t_b, pres_b, mask_b, lengths


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

    # ---- dataset / loader -------------------------------------------------
    ds = PresenceDataset(word_clips, args.joints,
                         max_len=args.max_len,
                         junk_ratio=args.junk_ratio,
                         seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
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

    # ---- FREEZE everything except presence head ---------------------------
    for p in model.parameters():
        p.requires_grad = False
    for p in model.yap.core.presence_head.parameters():
        p.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[FREEZE] Trainable: {n_trainable:,} / {n_total:,} "
          f"({100.0 * n_trainable / n_total:.2f}%) — presence head only")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.out_dir, exist_ok=True)
    best_loss = float("inf")
    history = {"train_loss": [], "true_loss": [], "lr": []}

    print("=" * 72)
    print("YAP Presence Trainer — presence head ONLY, everything else frozen")
    print("=" * 72)
    print(f"  device      : {device}")
    print(f"  word clips  : {len(word_clips)}")
    print(f"  epochs      : {args.epochs}  batch={args.batch_size}")
    print(f"  junk ratio  : {args.junk_ratio} (junk frames generated on the fly)")
    print(f"  out_dir     : {args.out_dir}")
    print("=" * 72)

    for ep in range(args.epochs):
        model.train()
        ep_loss = 0.0
        n_ok = 0

        for pose, conf, valid, labels, t, pres_target, mask, lengths in loader:
            pose, conf, valid, labels, t, pres_target, mask, lengths = (
                x.to(device) for x in (pose, conf, valid, labels, t, pres_target, mask, lengths))

            optimizer.zero_grad(set_to_none=True)
            out = model.yap.forward_frames(pose, conf, valid, t, lengths=lengths)
            presence = out["presence"]  # (B, T)

            pmask = mask.float()
            T_min = min(presence.shape[1], pres_target.shape[1])
            l_pres = focal_presence_loss(
                presence[:, :T_min], pres_target[:, :T_min], pmask[:, :T_min])

            loss = l_pres
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ep_loss += loss.item()
                n_ok += 1

        scheduler.step()
        avg = ep_loss / max(1, n_ok)
        print(f"[Ep {ep+1:02d}/{args.epochs}] pres={avg:.4f} | "
              f"lr {scheduler.get_last_lr()[0]:.2e} | {n_ok} steps")

        history["train_loss"].append(avg)
        history["true_loss"].append(avg)
        history["lr"].append(scheduler.get_last_lr()[0])

        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), os.path.join(args.out_dir, "yap_presence_best.pth"))
            print(f"  * new best -> yap_presence_best.pth (pres={avg:.4f})")

    print(f"\nDone. Best presence model saved to {args.out_dir}/yap_presence_best.pth")


def parse_args():
    p = argparse.ArgumentParser(description="YAP Presence Trainer — presence head only")
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
