"""
Train YAP's frame-level presence head to distinguish real word signs from
noise/garbage — WITHOUT modifying yap_conformer.py.

Approach (data-driven, not architecture):
  - Word clips (from tensors_include50_word_level): presence=1 during the sign,
    0 in idle padding (before/after). Spans are known from the clip itself.
  - Noise clips (from noise_clips/): presence=0 everywhere — all idle, no sign.

The presence head already exists in YAPCore. We just feed it both types of
clips and let focal_presence_loss push it to 1 on word frames and 0 on noise
frames.

Run:
    D:\MookVani\backend\venv\Scripts\python.exe \
        D:\MookVani\backend\segmentor\train_word_presence.py \
        --epochs 20 --batch_size 8 --noise_ratio 0.5
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
    presence_targets, focal_presence_loss, layer_conf_loss,
)
from utils.segmentor_utils import create_yap_model, SegmentorModelCfg

BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CLASSES = BACKEND_DIR / "models" / "word_vocabulary" / "classes.json"
DEFAULT_WORD_DIR = BACKEND_DIR / "data" / "tensors_include50_word_level"
DEFAULT_NOISE_DIR = Path(__file__).resolve().parent / "noise_clips"
DEFAULT_OUT = BACKEND_DIR / "segmentor" / "models" / "yap_stage1_presence"


class MixedClipDataset(Dataset):
    """
    Mixes word clips (presence=1 during sign) and noise clips (presence=0
    everywhere). Returns (pose, conf, valid, label, t, presence_target).
    label = class index for word clips, -1 for noise clips.
    """

    def __init__(self, word_clips, noise_clips, classes, num_joints,
                 max_len=400, noise_ratio=0.5, seed=42):
        self.word_clips = word_clips
        self.noise_clips = noise_clips
        self.classes = classes
        self.label_map = {c: i for i, c in enumerate(classes)}
        self.J = num_joints
        self.max_len = max_len
        self.noise_ratio = noise_ratio
        self.rng = random.Random(seed)

        # Build a balanced item list: each item is ('word', idx) or ('noise', idx)
        self.items = []
        for i in range(len(word_clips)):
            self.items.append(('word', i))
        for i in range(len(noise_clips)):
            self.items.append(('noise', i))
        self.rng.shuffle(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        kind, idx = self.items[index]
        if kind == 'word':
            return self._get_word(idx)
        else:
            return self._get_noise(idx)

    def _get_word(self, idx):
        item = self.word_clips[idx]
        raw = np.load(item["path"]).astype(np.float32)  # (T, J*3)
        T = min(len(raw), self.max_len)
        raw = raw[:T]
        pose = torch.from_numpy(raw).view(T, self.J, 3)
        conf = torch.ones(T, self.J)
        valid = torch.ones(T, self.J, dtype=torch.bool)
        t = torch.arange(T, dtype=torch.float32) / 30.0

        # The whole clip is one word: presence=1 for all valid frames
        spans = [(0.0, T / 30.0)]
        presence_t = presence_targets(t, spans, ramp_s=0.1)

        label = self.label_map.get(item["label"], -1)
        return pose, conf, valid, label, t, presence_t

    def _get_noise(self, idx):
        item = self.noise_clips[idx]
        raw = np.load(item["path"]).astype(np.float32)  # (T, 163)
        T = min(len(raw), self.max_len)
        raw = raw[:T]
        pose = torch.from_numpy(raw).view(T, self.J, 3)
        conf = torch.ones(T, self.J)
        valid = torch.ones(T, self.J, dtype=torch.bool)
        t = torch.arange(T, dtype=torch.float32) / 30.0

        # Noise: presence=0 everywhere (no sign)
        presence_t = torch.zeros(T)

        return pose, conf, valid, -1, t, presence_t


def collate(batch):
    poses, confs, valids, labels, ts, pres_ts = zip(*batch)
    B = len(poses)
    T = max(p.shape[0] for p in poses)
    J = poses[0].shape[1]
    pose_b = torch.zeros(B, T, J, 3)
    conf_b = torch.zeros(B, T, J)
    valid_b = torch.zeros(B, T, J, dtype=torch.bool)
    t_b = torch.zeros(B, T)
    mask_b = torch.zeros(B, T, dtype=bool)
    pres_b = torch.zeros(B, T)
    label_b = torch.full((B,), -1, dtype=torch.long)
    lengths = torch.zeros(B, dtype=torch.long)
    for i, (p, c, v, l, t_, pt, length) in enumerate(zip(
            poses, confs, valids, labels, ts, pres_ts, [p.shape[0] for p in poses])):
        n = p.shape[0]
        pose_b[i, :n] = p
        conf_b[i, :n] = c
        valid_b[i, :n] = v
        t_b[i, :n] = t_
        t_b[i, n:] = t_[-1]
        mask_b[i, :n] = True
        pres_b[i, :n] = pt
        label_b[i] = l
        lengths[i] = n
    return pose_b, conf_b, valid_b, label_b, t_b, pres_b, mask_b, lengths


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.classes, encoding="utf-8") as f:
        classes = json.load(f)
    num_classes = len(classes)
    print(f"[INFO] {num_classes} classes from classes.json")

    # ---- load word clips ---------------------------------------------------
    word_clips = []
    for d in args.data_dirs:
        d = Path(d)
        if not d.exists():
            continue
        for npy in d.rglob("*.npy"):
            lbl = npy.parent.name
            if lbl in {c: i for i, c in enumerate(classes)}:
                word_clips.append({"path": str(npy), "label": lbl})
    if args.max_per_class:
        from collections import defaultdict
        by_cls = defaultdict(list)
        for c in word_clips:
            by_cls[c["label"]].append(c)
        word_clips = []
        for items in by_cls.values():
            random.shuffle(items)
            word_clips.extend(items[:args.max_per_class])
    random.shuffle(word_clips)
    print(f"[INFO] {len(word_clips)} word clips loaded")

    # ---- load noise clips --------------------------------------------------
    noise_clips = []
    noise_dir = Path(args.noise_dir)
    if noise_dir.exists():
        idx_path = noise_dir / "index.json"
        if idx_path.exists():
            with open(idx_path) as f:
                entries = json.load(f)
            for e in entries:
                p = Path(e["path"])
                if p.exists():
                    noise_clips.append({"path": str(p), "label": "NOISE"})
    print(f"[INFO] {len(noise_clips)} noise clips loaded")

    if not word_clips and not noise_clips:
        raise RuntimeError("No clips found! Check data_dirs and noise_dir.")

    # ---- dataset / loader --------------------------------------------------
    ds = MixedClipDataset(word_clips, noise_clips, classes,
                          num_joints=args.joints,
                          max_len=args.max_len,
                          noise_ratio=args.noise_ratio,
                          seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate, num_workers=args.num_workers)

    # ---- model -------------------------------------------------------------
    sm_cfg = SegmentorModelCfg(
        layout=args.layout, d_model=args.d_model, n_mem_layers=args.n_mem_layers,
        summary_channels=args.summary_channels, hippo_order=args.hippo_order,
        entity_dim=args.entity_dim, num_words=num_classes,
        conformer_d_model=args.d_model, conformer_heads=4, conformer_layers=4,
    )
    model = create_yap_model(sm_cfg, device)

    if args.init and os.path.exists(args.init):
        from utils.segmentor_utils import load_yap_weights
        load_yap_weights(model, args.init, label="INIT")
        print(f"[INIT] loaded weights from {args.init}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.out_dir, exist_ok=True)
    best_loss = float("inf")

    print("=" * 72)
    print("YAP Word-Presence Training (data-driven, yap_conformer.py untouched)")
    print("=" * 72)
    print(f"  device     : {device}")
    print(f"  word clips : {len(word_clips)}")
    print(f"  noise clips: {len(noise_clips)}")
    print(f"  epochs     : {args.epochs}  batch={args.batch_size}")
    print(f"  out_dir    : {args.out_dir}")
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
            l_pres = focal_presence_loss(presence, pres_target, pmask)
            l_layer = layer_conf_loss(out["layer_conf"], pres_target, pmask)
            loss = l_pres + 0.5 * l_layer

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ep_loss += loss.item()
                n_ok += 1

        scheduler.step()
        avg = ep_loss / max(1, n_ok)
        print(f"[Ep {ep+1:02d}/{args.epochs}] LOSS {avg:.4f}")

        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), os.path.join(args.out_dir, "yap_presence_best.pth"))
            print(f"  * new best -> yap_presence_best.pth")

    print(f"\nDone. Best presence model saved to {args.out_dir}/yap_presence_best.pth")


def parse_args():
    p = argparse.ArgumentParser(description="Train YAP word-presence head (data-driven)")
    p.add_argument("--data_dirs", nargs="+",
                   default=[str(DEFAULT_WORD_DIR)])
    p.add_argument("--noise_dir", default=str(DEFAULT_NOISE_DIR))
    p.add_argument("--classes", default=str(DEFAULT_CLASSES))
    p.add_argument("--out_dir", default=str(DEFAULT_OUT))
    p.add_argument("--init", default=None, help="Pretrained YAPpingConformer weights")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--noise_ratio", type=float, default=0.5,
                   help="Fraction of noise clips in each epoch (0..1)")
    p.add_argument("--max_per_class", type=int, default=None)
    p.add_argument("--max_len", type=int, default=400)
    p.add_argument("--joints", type=int, default=53,
                   help="53 = corpus layout (hands+body+face), 42 = hands only")
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
