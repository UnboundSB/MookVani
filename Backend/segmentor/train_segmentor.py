"""
train_segmentor.py
==================

YAP Stage 2 — pure segmentation training on stitched word clips.

Phase A (--phase A): GESTURE SEGMENTATION on synthetic sentences.
    Stitches n word clips (1..max_words) into one continuous sentence.
    Runs YAP front end on the stitched stream and trains it to detect
    gesture boundaries. Ground truth: exact per-frame presence (1 inside
    each gesture span, 0 outside), exact end times (Gaussian bump).
    Losses:
        focal_presence_loss    per-pose presence BCE (soft targets)
        end_bce                per-pose end-prob Gaussian bump
        layer_conf_loss        deep supervision on layer confidences
        soft_count_loss        entity count should == word count
        min_gap_penalty        cuts shouldn't cluster in a window
        dip_prior_loss         cuts should happen at low speed
    No word classifier, no Conformer, no CTC. Just boundary detection.
    Trains: PoseCleaner-adjacent learnables (none — cleaner is param-free),
            MemoryLayer, presence/end heads, lift, readout, ctx_vec.

Phase B (--phase B): SYNTHETIC SENTENCE SEGMENTATION + CTC.
    Full YAPpingConformer with Conformer + CTC head.
    (Kept for later — Conformer learns entity→word mapping.)

Run Phase A first:
    .\venv\Scripts\python.exe segmentor\train_segmentor.py `
        --phase A --dataset_root segmentor/data `
        --init segmentor/models/yap_stage1/yap_noise_robust.pth `
        --out_dir segmentor/models/yap_stage2 `
        --epochs_a 30 --batch_size 4 --layout corpus --lr 1e-4
"""
from __future__ import annotations

import argparse, json, math, os, random, sys, gc
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.checkpoint_utils import load_flexible

from yap_conformer import (
    YAPConfig, YAP, ConformerConfig,
    focal_presence_loss, layer_conf_loss, end_targets, presence_targets,
    make_segments, soft_count_loss, min_gap_penalty, dip_prior_loss,
    sentence_ctc_loss, count_parameters, Segment,
)
from noise_aug import add_noise
from tqdm import tqdm


# ----------------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------------
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_SRC = os.path.join(BACKEND_DIR, "data", "tensors_include50_word_level")
DEFAULT_DATASET_ROOT = os.path.join(BACKEND_DIR, "segmentor", "data")
DEFAULT_CLASSES  = os.path.join(BACKEND_DIR, "models", "word_vocabulary", "classes.json")
DEFAULT_INIT     = os.path.join(BACKEND_DIR, "segmentor", "models", "yap_stage1", "yap_noise_robust.pth")
DEFAULT_OUT      = os.path.join(BACKEND_DIR, "segmentor", "models", "yap_stage2")

# Spatial-only augmentation (time axis untouched, boundaries stay valid)
AUG_CFG = None  # set in Phase B only


# ============================================================================
# Missing-coordinate fill (prev + next interpolation) for inputs
# ============================================================================
def fill_missing_coords(pose: torch.Tensor, valid: torch.Tensor, max_gap_frames: int = 15):
    T, J, _ = pose.shape
    pose_f = pose.clone()
    for j in range(J):
        v = valid[:, j]
        idx_valid = torch.nonzero(v).flatten()
        if idx_valid.numel() == 0:
            continue
        first = int(idx_valid[0]); last = int(idx_valid[-1])
        if first > 0: pose_f[:first, j] = pose[first, j]
        if last < T - 1: pose_f[last + 1:, j] = pose[last, j]
        for a, b in zip(idx_valid[:-1], idx_valid[1:]):
            a, b = int(a), int(b); gap = b - a - 1
            if gap == 0 or gap > max_gap_frames: continue
            t = torch.arange(1, gap + 1, dtype=pose.dtype, device=pose.device) / (gap + 1)
            pose_f[a + 1:b, j] = pose[a, j].unsqueeze(0) * (1 - t).unsqueeze(-1) + \
                                 pose[b, j].unsqueeze(0) * t.unsqueeze(-1)
    return pose_f


# ============================================================================
# Phase A dataset: stitched synthetic sentences from word clips
# ============================================================================
def load_dataset_mapping(root: str):
    import json as _json
    with open(os.path.join(root, "mapping.json"), encoding="utf-8") as f:
        return _json.load(f)


def load_word_clips_from_dataset(root: str, classes: list[str], max_per_class: int = 22):
    """Load .pt word clips grouped by class index from segmentor/data/."""
    label_map = {c: i for i, c in enumerate(classes)}
    clips_by_class = {}
    d = Path(root)
    for folder in d.iterdir():
        if folder.name not in label_map: continue
        idx = label_map[folder.name]
        pts = sorted(folder.rglob("*.pt"))
        random.shuffle(pts)
        items = []
        for p in pts[:max_per_class]:
            wp = p
            pp = p.with_name("processed.pt")
            if wp.exists() and pp.exists():
                try:
                    t = torch.load(wp, weights_only=True).float()
                    if t.numel() % 163 != 0:
                        continue  # not a valid word clip (T, 163)
                    items.append((str(wp), str(pp), idx))
                except Exception:
                    continue
        if items:
            clips_by_class[idx] = items
    return clips_by_class


def _blend(a: np.ndarray, b: np.ndarray, alpha: float = 0.3):
    n_blend = max(1, int(min(len(a), len(b)) * alpha))
    w = np.linspace(0, 1, n_blend, dtype=np.float32)[:, None]
    ab = (1 - w) * a[-n_blend:] + w * b[:n_blend]
    return a[:-n_blend], ab, b[n_blend:]


def build_synthetic_sentence(clips_by_class, classes, n_words: int, J: int,
                             fps: float = 30.0, idle_range=(10, 25),
                             blend_alpha=0.3, rng=None) -> tuple[np.ndarray, list[tuple[float, float]], list[int]]:
    """Stitch n_words word clips into one sentence tensor.

    Returns:
        raw   (T, J*3)  stitched pose tensor
        spans [(start_s, end_s), ...]  gesture boundaries in seconds
        word_ids  [class_idx, ...]     class index per gesture
    """
    if rng is None: rng = random.Random()
    available = [k for k, v in clips_by_class.items() if v]
    if not available:
        raise RuntimeError("No clips available")
    word_ids = rng.choices(available, k=n_words)
    parts, spans, t_cursor = [], [], rng.uniform(0.1, 0.5)
    for wid in word_ids:
        clip_info = rng.choice(clips_by_class[wid])
        wp = Path(clip_info[0])
        raw = torch.load(wp, weights_only=True).float()
        if raw.dim() == 1:
            raw = raw.view(-1, 163)
        clip = raw[:, :J * 3].contiguous().view(raw.shape[0], J, 3).numpy().astype(np.float32)
        sp = rng.uniform(0.8, 1.25)
        T_new = max(4, int(len(clip) * sp))
        idx = np.linspace(0, len(clip) - 1, T_new)
        i0, i1 = idx.astype(int), np.minimum(idx.astype(int) + 1, len(clip) - 1)
        a = idx - i0
        clip = clip[i0] * (1 - a[:, None, None]) + clip[i1] * a[:, None, None]
        clip = clip.reshape(T_new, J * 3)
        if rng.random() < 0.5:
            clip = clip.reshape(T_new, J, 3)
            clip[:, :, 0] = 1.0 - clip[:, :, 0]
            clip = clip.reshape(T_new, J * 3)
        start_s = t_cursor; end_s = t_cursor + len(clip) / fps
        spans.append((start_s, end_s)); parts.append(clip)
        t_cursor = end_s + rng.uniform(*idle_range) / fps
    if len(parts) == 1:
        merged = parts[0]
    else:
        seg0, bl, seg1 = _blend(parts[0], parts[1], blend_alpha)
        merged = np.concatenate([seg0, bl, seg1])
        for p in parts[2:]:
            tail, bl, head = _blend(merged, p, blend_alpha)
            merged = np.concatenate([tail, bl, head])
    idle_lead = np.zeros((rng.randint(*idle_range), J * 3), dtype=np.float32)
    idle_tail = np.zeros((rng.randint(*idle_range), J * 3), dtype=np.float32)
    raw = np.concatenate([idle_lead, merged, idle_tail])
    return raw, spans, word_ids


class SyntheticSentenceDataset(Dataset):
    def __init__(self, clips_by_class, classes, n_sentences, J, fps=30.0,
                 max_len=600, max_words=5, seed=0):
        self.clips_by_class = clips_by_class; self.classes = classes
        self.n = n_sentences; self.J = J; self.fps = fps
        self.max_len = max_len; self.max_words = max_words
        self.rng = random.Random(seed)

    def __len__(self): return self.n

    def __getitem__(self, _idx):
        n_words = self.rng.randint(1, self.max_words)
        raw, spans, word_ids = build_synthetic_sentence(
            self.clips_by_class, self.classes, n_words=n_words,
            J=self.J, fps=self.fps, rng=self.rng)
        T = min(len(raw), self.max_len); raw = raw[:T]
        pose = torch.from_numpy(raw).view(T, self.J, 3)
        conf = torch.ones(T, self.J); valid = torch.ones(T, self.J, dtype=torch.bool)
        t = torch.arange(T, dtype=torch.float32) / self.fps
        max_t = (T - 1) / self.fps
        spans = [(s, min(e, max_t)) for s, e in spans if s < max_t]
        return pose, conf, valid, t, spans, word_ids


def sent_collate(batch):
    poses, confs, valids, ts, spans_list, word_ids_list = zip(*batch)
    B = len(poses); T = max(p.shape[0] for p in poses); J = poses[0].shape[1]
    pose_b = torch.zeros(B, T, J, 3); conf_b = torch.zeros(B, T, J)
    valid_b = torch.zeros(B, T, J, dtype=torch.bool); t_b = torch.zeros(B, T)
    lengths = torch.zeros(B, dtype=torch.long)
    for i, (p, c, v, t_) in enumerate(zip(poses, confs, valids, ts)):
        n = p.shape[0]
        pose_b[i, :n] = p; conf_b[i, :n] = c; valid_b[i, :n] = v
        t_b[i, :n] = t_; t_b[i, n:] = t_[-1]; lengths[i] = n
    return pose_b, conf_b, valid_b, t_b, lengths, list(spans_list), list(word_ids_list)


# ============================================================================
# Phase B dataset (kept for later — Conformer + CTC)
# ============================================================================
def load_word_clips_npy(data_dir, classes, max_per_class=15):
    label_map = {c: i for i, c in enumerate(classes)}
    out = {}; d = Path(data_dir)
    for folder in d.iterdir():
        if folder.name not in label_map: continue
        idx = label_map[folder.name]; npys = list(folder.rglob("*.npy"))
        random.shuffle(npys)
        out[idx] = [np.load(p).astype(np.float32) for p in npys[:max_per_class]]
    return out


class SyntheticSentenceDatasetNPY(Dataset):
    def __init__(self, clips_by_class, classes, n_sentences, J, fps=30.0,
                 max_len=600, max_words=5, seed=0):
        self.clips_by_class = clips_by_class; self.classes = classes
        self.n = n_sentences; self.J = J; self.fps = fps
        self.max_len = max_len; self.max_words = max_words; self.rng = random.Random(seed)

    def __len__(self): return self.n

    def __getitem__(self, _idx):
        n_words = self.rng.randint(1, self.max_words)
        raw, spans, word_ids = build_synthetic_sentence(
            self.clips_by_class, self.classes, n_words=n_words,
            J=self.J, fps=self.fps, rng=self.rng)
        T = min(len(raw), self.max_len); raw = raw[:T]
        pose = torch.from_numpy(raw).view(T, self.J, 3)
        conf = torch.ones(T, self.J); valid = torch.ones(T, self.J, dtype=torch.bool)
        t = torch.arange(T, dtype=torch.float32) / self.fps
        max_t = (T - 1) / self.fps
        spans = [(s, min(e, max_t)) for s, e in spans if s < max_t]
        return pose, conf, valid, t, spans, word_ids


def sent_collate_npy(batch):
    poses, confs, valids, ts, spans_list, word_ids_list = zip(*batch)
    B = len(poses); T = max(p.shape[0] for p in poses); J = poses[0].shape[1]
    pose_b = torch.zeros(B, T, J, 3); conf_b = torch.zeros(B, T, J)
    valid_b = torch.zeros(B, T, J, dtype=torch.bool); t_b = torch.zeros(B, T)
    lengths = torch.zeros(B, dtype=torch.long)
    for i, (p, c, v, t_) in enumerate(zip(poses, confs, valids, ts)):
        n = p.shape[0]
        pose_b[i, :n] = p; conf_b[i, :n] = c; valid_b[i, :n] = v
        t_b[i, :n] = t_; t_b[i, n:] = t_[-1]; lengths[i] = n
    return pose_b, conf_b, valid_b, t_b, lengths, list(spans_list), list(word_ids_list)


# ============================================================================
# Plotting helper
# ============================================================================
def plot_training_curves(history: dict, out_dir: str, title: str, tag: str = "phaseA_"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))
    with open(out_dir / f"{tag}history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes[0, 0].plot(epochs, history["train_loss"], label="train")
        axes[0, 0].plot(epochs, history["val_loss"], label="val")
        axes[0, 0].set_title("Loss"); axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
        axes[1, 0].plot(epochs, history["lr"])
        axes[1, 0].set_title("Learning Rate"); axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("LR"); axes[1, 1].grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(out_dir / f"{tag}training_curves.png", dpi=100); plt.close(fig)
    except Exception as e:
        print(f"  [plot] matplotlib not available or plot failed: {e}")


# ============================================================================
# Training
# ============================================================================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed); torch.manual_seed(args.seed)

    with open(args.classes, encoding="utf-8") as f:
        classes = json.load(f)
    num_classes = len(classes)

    kw = dict(d_model=args.d_model, n_mem_layers=args.n_mem_layers,
              summary_channels=args.summary_channels, hippo_order=args.hippo_order,
              entity_dim=args.entity_dim)
    if args.layout == "corpus": ycfg = YAPConfig.corpus(**kw)
    elif args.layout == "holistic": ycfg = YAPConfig.holistic(**kw)
    else: ycfg = YAPConfig(**kw)
    J = ycfg.total_joints
    print(f"[LAYOUT] {args.layout}: J={J}")

    yap = YAP(ycfg).to(device)
    if args.init and os.path.exists(args.init):
        sd = torch.load(args.init, map_location=device, weights_only=True)
        m, u = load_flexible(yap, sd, label="STAGE1")
        print(f"[INIT] {args.init} | missing {len(m)}, unexpected {len(u)}")

    # ========================================================================
    # PHASE A: pure segmentation on stitched sentences
    # ========================================================================
    if args.phase in ("A", "both"):
        print("\n" + "=" * 60)
        print("PHASE A: YAP segmentation on stitched sentences (no Conformer)")
        print("=" * 60)

        # Build dataset from segmentor/data/ (.pt clips)
        clips_by_class = load_word_clips_from_dataset(args.dataset_root, classes, args.max_per_class)
        if not clips_by_class:
            raise RuntimeError(
                f"No clips found in {args.dataset_root}. "
                "Run build_word_dataset.py --phase 2 first to create processed.pt files."
            )
        n_clips = sum(len(v) for v in clips_by_class.values())
        print(f"[DATA] {n_clips} word clips across {len(clips_by_class)} classes")

        train_ds = SyntheticSentenceDataset(clips_by_class, classes, args.train_sentences,
                                            J, fps=args.fps, max_len=args.max_len,
                                            max_words=args.max_words, seed=args.seed)
        val_ds = SyntheticSentenceDataset(clips_by_class, classes, args.val_sentences,
                                          J, fps=args.fps, max_len=args.max_len,
                                          max_words=args.max_words, seed=args.seed + 1)
        wdl = DataLoader(train_ds, args.batch_size, shuffle=True,
                         collate_fn=sent_collate, num_workers=0)
        vdl = DataLoader(val_ds, args.batch_size, shuffle=False,
                         collate_fn=sent_collate, num_workers=0)

        params = list(yap.parameters())
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
        if args.lr_sched == "warm_restarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=max(1, args.epochs_a // 4), T_mult=args.t_mult)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs_a)

        # Resume support
        start_epoch = 0
        history = {"train_loss": [], "val_loss": [], "lr": []}
        phaseA_best_path = os.path.join(args.out_dir, "yap_phaseA_best.pth")
        if args.resume and os.path.exists(phaseA_best_path):
            sd = torch.load(phaseA_best_path, map_location=device, weights_only=False)
            yap.load_state_dict(sd["yap"], strict=False)
            if "optimizer" in sd:
                try:
                    optimizer.load_state_dict(sd["optimizer"])
                except ValueError:
                    print("  [RESUME] optimizer state mismatch — starting fresh")
            if "scheduler" in sd:
                try:
                    scheduler.load_state_dict(sd["scheduler"])
                except ValueError:
                    print("  [RESUME] scheduler state mismatch — starting fresh")
            if "history" in sd:
                history = sd["history"]; start_epoch = len(history["train_loss"])
            print(f"[RESUME] loaded {phaseA_best_path}, starting from epoch {start_epoch + 1}")

        yap.train(); best_loss = float("inf")
        if history["val_loss"]: best_loss = min(history["val_loss"])

        for ep in range(start_epoch, args.epochs_a):
            ep_loss = 0.0; n_ok = 0
            pbar = tqdm(wdl, desc=f"PhaseA Ep {ep + 1:02d}/{args.epochs_a} train", leave=False)
            for pose, conf, valid, t, lengths, spans_list, word_ids_list in pbar:
                pose, conf, valid, t, lengths = (pose.to(device), conf.to(device),
                                                  valid.to(device), t.to(device), lengths.to(device))
                B = pose.shape[0]
                gt_segments = [make_segments(t[b, :int(lengths[b])].tolist(), spans_list[b], ycfg)
                               for b in range(B)]
                out = yap.forward_frames(pose, conf, valid, t)
                yap_out = out

                # Presence targets (soft, 1 inside span, ramp outside)
                pres_full = pose.new_zeros(B, pose.shape[1])
                for b in range(B):
                    pres_full[b, :int(lengths[b])] = presence_targets(
                        t[b, :int(lengths[b])], spans_list[b])
                mask = torch.zeros_like(pres_full, dtype=torch.bool)
                for b in range(B): mask[b, :int(lengths[b])] = True
                l_pres = focal_presence_loss(yap_out["presence"], pres_full, mask.float())

                # End head: Gaussian bump at each span end
                end_full = pose.new_zeros(B, pose.shape[1])
                for b in range(B):
                    end_full[b, :int(lengths[b])] = end_targets(
                        t[b, :int(lengths[b])], spans_list[b])
                l_end = F.binary_cross_entropy_with_logits(
                    yap_out["end"], end_full, pos_weight=torch.tensor(10.0, device=device))

                # Deep supervision on layer confidences
                l_layer = layer_conf_loss(yap_out["layer_conf"], pres_full, mask.float())

                # Word count should match expected segments
                n_words_t = torch.tensor([len(w) for w in word_ids_list],
                                         dtype=torch.float32, device=device)
                l_count = soft_count_loss(yap_out["end"], n_words_t.long(), mask.float())
                l_gap = min_gap_penalty(yap_out["end"], k=6, mask=mask.float())
                l_dip = dip_prior_loss(yap_out["end"], yap_out["speed"], mask.float())

                loss = (l_pres + l_end + 0.5 * l_layer + 0.1 * l_count +
                        0.05 * l_gap + 0.02 * l_dip)
                if torch.isfinite(loss):
                    optimizer.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    optimizer.step()
                    ep_loss += loss.item(); n_ok += 1
                    pbar.set_postfix(loss=f"{ep_loss / max(1, n_ok):.4f}")
            pbar.close(); scheduler.step()

            train_loss = ep_loss / max(1, n_ok)

            # Validation
            yap.eval()
            val_loss = 0.0; val_n = 0
            with torch.no_grad():
                for pose, conf, valid, t, lengths, spans_list, word_ids_list in vdl:
                    pose, conf, valid, t, lengths = (pose.to(device), conf.to(device),
                                                      valid.to(device), t.to(device), lengths.to(device))
                    B = pose.shape[0]
                    gt_segments = [make_segments(t[b, :int(lengths[b])].tolist(), spans_list[b], ycfg)
                                   for b in range(B)]
                    res = yap.forward_frames(pose, conf, valid, t)
                    yap_out = res
                    pres_full = pose.new_zeros(B, pose.shape[1])
                    for b in range(B):
                        pres_full[b, :int(lengths[b])] = presence_targets(
                            t[b, :int(lengths[b])], spans_list[b])
                    mask = torch.zeros_like(pres_full, dtype=torch.bool)
                    for b in range(B): mask[b, :int(lengths[b])] = True
                    l_pres = focal_presence_loss(yap_out["presence"], pres_full, mask.float())
                    end_full = pose.new_zeros(B, pose.shape[1])
                    for b in range(B):
                        end_full[b, :int(lengths[b])] = end_targets(
                            t[b, :int(lengths[b])], spans_list[b])
                    l_end = F.binary_cross_entropy_with_logits(
                        yap_out["end"], end_full,
                        pos_weight=torch.tensor(10.0, device=device))
                    l_layer = layer_conf_loss(yap_out["layer_conf"], pres_full, mask.float())
                    n_words_t = torch.tensor([len(w) for w in word_ids_list],
                                             dtype=torch.float32, device=device)
                    l_count = soft_count_loss(yap_out["end"], n_words_t.long(), mask.float())
                    l_gap = min_gap_penalty(yap_out["end"], k=6, mask=mask.float())
                    l_dip = dip_prior_loss(yap_out["end"], yap_out["speed"], mask.float())
                    vloss = l_pres + l_end + 0.5 * l_layer + 0.1 * l_count + 0.05 * l_gap + 0.02 * l_dip
                    if torch.isfinite(vloss):
                        val_loss += vloss.item(); val_n += 1
            yap.train()

            val_loss_avg = val_loss / max(1, val_n)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss_avg)
            history["lr"].append(scheduler.get_last_lr()[0])

            print(f"[PhaseA Ep {ep + 1:02d}] train_loss {train_loss:.4f} | "
                  f"val_loss {val_loss_avg:.4f} | lr {scheduler.get_last_lr()[0]:.2e}")

            # Plot curves
            plot_training_curves(history, args.out_dir,
                                 title="Phase A - YAP segmentation",
                                 tag="phaseA_")

            if val_loss_avg < best_loss:
                best_loss = val_loss_avg
                torch.save({
                    "yap": yap.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "history": history,
                    "epoch": ep + 1,
                }, os.path.join(args.out_dir, "yap_phaseA_best.pth"))
                print(f"  * new best val_loss {best_loss:.4f} -> yap_phaseA_best.pth")

        os.makedirs(args.out_dir, exist_ok=True)
        torch.save(yap.state_dict(), os.path.join(args.out_dir, "yap_phaseA_final.pth"))
        with open(os.path.join(args.out_dir, "phaseA_history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        print(f"[PhaseA] done. Saved yap_phaseA_final.pth")

    # ========================================================================
    # PHASE B: synthetic sentence segmentation + CTC (Conformer trained here)
    # ========================================================================
    if args.phase in ("B", "both"):
        print("\n" + "=" * 60)
        print("PHASE B: synthetic sentences + segmentation + CTC (Conformer trained)")
        print("=" * 60)

        ccfg = ConformerConfig(num_words=num_classes, d_model=args.d_model,
                               heads=args.heads, layers=args.layers)
        model = YAPpingConformer(ycfg, ccfg).to(device)

        phaseA_path = os.path.join(args.out_dir, "yap_phaseA_best.pth")
        if args.init and os.path.exists(args.init):
            sd = torch.load(args.init, map_location=device, weights_only=True)
            m, u = load_flexible(model, sd, label="INIT")
            print(f"[INIT] {args.init} | missing {len(m)}, unexpected {len(u)}")
        elif os.path.exists(phaseA_path):
            sd = torch.load(phaseA_path, map_location=device, weights_only=True)
            model.yap.load_state_dict(sd["yap"], strict=False)
            print(f"[INIT] loaded Phase A YAP weights from {phaseA_path}")

        clips_by_class = load_word_clips_npy(args.data_src, classes, args.max_per_class)
        if not clips_by_class:
            raise RuntimeError("No .npy clips found. Run .pt -> .npy conversion first.")
        print(f"[DATA] {sum(len(v) for v in clips_by_class.values())} clips")

        bds = SyntheticSentenceDatasetNPY(clips_by_class, classes, args.train_sentences,
                                          J, fps=args.fps, max_len=args.max_len,
                                          max_words=args.max_words, seed=args.seed)
        bdv = SyntheticSentenceDatasetNPY(clips_by_class, classes, args.val_sentences,
                                          J, fps=args.fps, max_len=args.max_len,
                                          aug=False, seed=args.seed + 1)
        bld = DataLoader(bds, args.batch_size, shuffle=True,
                         collate_fn=sent_collate_npy, num_workers=args.num_workers)
        bvd = DataLoader(bdv, args.batch_size, shuffle=False,
                         collate_fn=sent_collate_npy, num_workers=args.num_workers)

        if args.freeze_yap:
            for p in model.yap.parameters(): p.requires_grad = False
            opt_params = [p for p in model.parameters() if p.requires_grad]
            print("[FREEZE] YAP frozen; training Conformer + head only")
        else:
            opt_params = model.parameters()
        optimizer = torch.optim.AdamW(opt_params, lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs_b)

        os.makedirs(args.out_dir, exist_ok=True)
        best_wer = float("inf")

        for ep in range(args.epochs_b):
            model.train(); ep_loss = 0.0; n_ok = 0
            pbar = tqdm(bld, desc=f"PhaseB Ep {ep + 1:02d}/{args.epochs_b}", leave=False)
            for pose, conf, valid, t, lengths, spans_list, word_ids_list in pbar:
                pose, conf, valid, t, lengths = (pose.to(device), conf.to(device),
                                                  valid.to(device), t.to(device), lengths.to(device))
                B = pose.shape[0]
                gt_segments = [make_segments(t[b, :int(lengths[b])].tolist(), spans_list[b], ycfg)
                               for b in range(B)]
                res = model(pose, conf, valid, t, lengths, segments=gt_segments)
                yap_out = res["yap"]

                pres_full = pose.new_zeros(B, pose.shape[1])
                for b in range(B):
                    pres_full[b, :int(lengths[b])] = presence_targets(
                        t[b, :int(lengths[b])], spans_list[b])
                mask = torch.zeros_like(pres_full, dtype=torch.bool)
                for b in range(B): mask[b, :int(lengths[b])] = True
                l_pres = focal_presence_loss(yap_out["presence"], pres_full, mask.float())

                end_full = pose.new_zeros(B, pose.shape[1])
                for b in range(B):
                    end_full[b, :int(lengths[b])] = end_targets(
                        t[b, :int(lengths[b])], spans_list[b])
                l_end = F.binary_cross_entropy_with_logits(
                    yap_out["end"], end_full, pos_weight=torch.tensor(10.0, device=device))

                l_layer = layer_conf_loss(yap_out["layer_conf"], pres_full, mask.float())
                n_words_t = torch.tensor([len(w) for w in word_ids_list],
                                         dtype=torch.float32, device=device)
                l_count = soft_count_loss(yap_out["end"], n_words_t.long(), mask.float())
                l_gap = min_gap_penalty(yap_out["end"], k=6, mask=mask.float())
                l_dip = dip_prior_loss(yap_out["end"], yap_out["speed"], mask.float())

                l_ctc = sentence_ctc_loss(res["word_logits"], res["token_mask"],
                                          torch.cat([torch.tensor(w, device=device) for w in word_ids_list]),
                                          torch.tensor([len(w) for w in word_ids_list], device=device))

                loss = (l_pres + l_end + 0.5 * l_layer + 0.1 * l_count +
                        0.05 * l_gap + 0.02 * l_dip + args.ctc_w * l_ctc)
                if torch.isfinite(loss):
                    optimizer.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
                    optimizer.step()
                    ep_loss += loss.item(); n_ok += 1
                    pbar.set_postfix(loss=f"{ep_loss / max(1, n_ok):.4f}")
            pbar.close(); scheduler.step()
            print(f"[PhaseB Ep {ep + 1:02d}] loss {ep_loss / max(1, n_ok):.4f}")

            if (ep + 1) % 5 == 0 or ep + 1 == args.epochs_b:
                model.eval(); total_s, total_n = 0, 0
                with torch.no_grad():
                    for pose, conf, valid, t, lengths, spans_list, word_ids_list in bvd:
                        pose, conf, valid, t, lengths = (pose.to(device), conf.to(device),
                                                          valid.to(device), t.to(device), lengths.to(device))
                        res = model(pose, conf, valid, t, lengths)
                        wl = res["word_logits"]; tm = res["token_mask"]
                        for b in range(pose.shape[0]):
                            ref = [w + 1 for w in word_ids_list[b]]
                            hyp = wl[b, tm[b]].argmax(-1).tolist()
                            hyp = [h for h in hyp if h != 0]
                            import editdistance
                            total_s += editdistance.eval(ref, hyp); total_n += len(ref)
                wer = total_s / max(1, total_n)
                print(f"  val WER {wer * 100:.1f}%")
                torch.save({"epoch": ep + 1, "model": model.state_dict()},
                           os.path.join(args.out_dir, "yap_seg_ckpt.pth"))
                if wer < best_wer:
                    best_wer = wer
                    torch.save(model.state_dict(), os.path.join(args.out_dir, "yap_seg_best.pth"))
                    print(f"  * new best WER {best_wer * 100:.1f}%")

    print("\nDone.")


def parse_args():
    p = argparse.ArgumentParser(description="YAP Stage 2: Phase A segmentation + Phase B sentence CTC")
    p.add_argument("--data_src", default=DEFAULT_DATA_SRC)
    p.add_argument("--dataset_root", default=DEFAULT_DATASET_ROOT)
    p.add_argument("--classes", default=DEFAULT_CLASSES)
    p.add_argument("--init", default=DEFAULT_INIT)
    p.add_argument("--out_dir", default=DEFAULT_OUT)
    p.add_argument("--phase", default="A", choices=["A", "B", "both"],
                   help="A=segmentation on stitched sentences, B=sentence CTC, both=A then B")
    p.add_argument("--epochs_a", type=int, default=30)
    p.add_argument("--epochs_b", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader worker threads (Phase B only; Phase A uses 0)")
    p.add_argument("--layout", default="corpus", choices=["corpus", "holistic", "hands"])
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--max_gap_frames", type=int, default=15,
                   help="max gap frames to interpolate across; longer gaps held")
    p.add_argument("--max_len", type=int, default=900)
    p.add_argument("--max_per_class", type=int, default=22,
                   help="max clips per word class (default 22)")
    p.add_argument("--train_sentences", type=int, default=4000)
    p.add_argument("--val_sentences", type=int, default=400)
    p.add_argument("--max_words", type=int, default=5,
                   help="max words per synthetic sentence")
    p.add_argument("--max_samples", type=int, default=0,
                   help="limit total training samples (0 = all, distributed evenly across classes)")
    p.add_argument("--train_ratio", type=float, default=0.8,
                   help="fraction of samples per class used for training (rest = validation)")
    p.add_argument("--resume", action="store_true",
                   help="resume Phase A from yap_phaseA_best.pth in out_dir")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    p.add_argument("--lr_sched", type=str, default="cos",
                   choices=["cos", "warm_restarts"],
                   help="LR scheduler: cos=decay to 0, warm_restarts=cosine cycles")
    p.add_argument("--t_mult", type=float, default=2.0,
                   help="T_mult for CosineAnnealingWarmRestarts (cycle length multiplier)")
    p.add_argument("--ctc_w", type=float, default=1.0, help="weight for CTC loss in Phase B")
    p.add_argument("--freeze_yap", action="store_true",
                   help="freeze YAP in Phase B (train Conformer + head only)")
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_mem_layers", type=int, default=3)
    p.add_argument("--summary_channels", type=int, default=32)
    p.add_argument("--hippo_order", type=int, default=16)
    p.add_argument("--entity_dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
