"""
train_segmentor.py
==================
Stage 2 of the segmentor curriculum: GESTURE SEGMENTATION on synthetic sentences.

Loads the noise-robust YAP weights from Stage 1 and fine-tunes the segmenter
heads (presence, end, layer_conf) using the synthetic sentence builder that
exists in the CTC trainers. The Conformer head is trained with CTC on the
entity stream, so the model learns both to segment AND to label in one shot.

The YAP queue data structure (YAPStream / the deque in YAPStream.buf) is the
live inference path; this script trains the OFFLINE equivalent (forward_frames +
segment_batch + encode_segments) which is guaranteed to produce identical results.

Training signal
---------------
  Synthetic sentence: N word clips stitched together with co-articulation blends
  (random speed, flip, shift just like train_synthetic_sentences.py).
  For each sentence we know:
    * exact per-frame presence (1 inside each gesture span, 0 outside) -> focal BCE
    * exact boundary times -> end-head Gaussian bump BCE
    * word sequence in signing order -> CTC loss on entity stream
    * count of words -> soft count loss on end probabilities
    * min gap constraint -> min_gap_penalty

After this stage you have a YAP that:
  1. Filters noise (Stage 1)
  2. Segments gestures correctly on synthetic sentences (this stage)
  3. Is ready for Stage 3 real-sentence fine-tuning (pass these weights as --init
     to train_real_sentences.py or future train_yap_real.py)
"""
from __future__ import annotations

import argparse, json, math, os, random, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from yap_conformer import (
    YAPConfig, YAPpingConformer, ConformerConfig,
    focal_presence_loss, layer_conf_loss, end_targets, presence_targets,
    make_segments, soft_count_loss, min_gap_penalty, dip_prior_loss,
    sentence_ctc_loss, count_parameters,
)
from noise_aug import add_noise

# ── defaults ─────────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR = "Backend/data/tensors_include50_word_level"
DEFAULT_CLASSES  = "Backend/data/classes.json"
DEFAULT_OUT      = "Backend/segmentor/models/yap_stage2"


# ==============================================================================
# Synthetic sentence builder
# ==============================================================================
def load_word_clips(data_dir: str, classes: list[str], max_per_class: int = 15) -> dict[int, list[np.ndarray]]:
    """Load word-level .npy clips grouped by class index."""
    label_map = {c: i for i, c in enumerate(classes)}
    clips_by_class: dict[int, list[np.ndarray]] = {}
    d = Path(data_dir)
    for folder in d.iterdir():
        if folder.name not in label_map:
            continue
        idx = label_map[folder.name]
        npys = list(folder.rglob("*.npy"))
        random.shuffle(npys)
        clips_by_class[idx] = [np.load(p).astype(np.float32) for p in npys[:max_per_class]]
    return clips_by_class


def _blend(a: np.ndarray, b: np.ndarray, alpha: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blend the tail of clip a with the head of clip b to make a smooth join."""
    n_blend = max(1, int(min(len(a), len(b)) * alpha))
    w = np.linspace(0, 1, n_blend, dtype=np.float32)[:, None]
    ab = (1 - w) * a[-n_blend:] + w * b[:n_blend]
    return a[:-n_blend], ab, b[n_blend:]


def build_synthetic_sentence(clips_by_class, classes, n_words: int = None, J: int = 42,
                              fps: float = 30.0, idle_range=(10, 25), blend_alpha=0.3,
                              rng: random.Random = None) -> tuple[np.ndarray, list[tuple[float,float]], list[int]]:
    """
    Stitch n_words word clips into one sentence. Returns:
      raw   (T, J*3)  flattened landmarks
      spans [(start_s, end_s), ...]  gesture boundaries in seconds
      word_ids  [class_idx, ...]
    """
    if rng is None:
        rng = random.Random()
    available = [k for k, v in clips_by_class.items() if v]
    if not available:
        raise RuntimeError("No clips available")
    n_words = n_words or rng.randint(1, 5)
    word_ids = rng.choices(available, k=n_words)

    parts = []
    spans = []
    t_cursor = rng.uniform(0.1, 0.5)   # leading idle

    for wi, wid in enumerate(word_ids):
        clip = rng.choice(clips_by_class[wid]).copy()
        # Speed warp
        sp = rng.uniform(0.8, 1.25)
        T_new = max(4, int(len(clip) * sp))
        idx = np.linspace(0, len(clip) - 1, T_new)
        i0, i1 = idx.astype(int), np.minimum(idx.astype(int) + 1, len(clip) - 1)
        a = idx - i0
        clip = clip[i0] * (1 - a[:, None]) + clip[i1] * a[:, None]
        # Random flip (horizontal: negate x channels every 3rd column: 0, 3, 6, ...)
        if rng.random() < 0.5:
            clip = clip.reshape(T_new, J, 3)
            clip[:, :, 0] = 1.0 - clip[:, :, 0]
            clip = clip.reshape(T_new, J * 3)

        start_s = t_cursor
        end_s   = t_cursor + len(clip) / fps
        spans.append((start_s, end_s))
        parts.append(clip)
        t_cursor = end_s + rng.uniform(idle_range[0], idle_range[1]) / fps

    # Blend joins
    merged = []
    if len(parts) == 1:
        merged = parts[0]
    else:
        seg0, blend01, seg1 = _blend(parts[0], parts[1], blend_alpha)
        merged = np.concatenate([seg0, blend01, seg1])
        for p in parts[2:]:
            tail, bl, head = _blend(merged, p, blend_alpha)
            merged = np.concatenate([tail, bl, head])

    # Leading + trailing idle frames
    idle_lead = np.zeros((rng.randint(*idle_range), J * 3), dtype=np.float32)
    idle_tail = np.zeros((rng.randint(*idle_range), J * 3), dtype=np.float32)
    raw = np.concatenate([idle_lead, merged if isinstance(merged, np.ndarray) else np.concatenate(merged), idle_tail])
    return raw, spans, word_ids


# ==============================================================================
# Dataset
# ==============================================================================
class SyntheticSentenceDataset(Dataset):
    def __init__(self, clips_by_class, classes, n_sentences: int, J: int, fps: float = 30.0,
                 max_len: int = 600, max_words: int = 5, aug: bool = True,
                 seed: int = 0):
        self.clips_by_class = clips_by_class
        self.classes = classes
        self.n = n_sentences
        self.J = J
        self.fps = fps
        self.max_len = max_len
        self.max_words = max_words
        self.aug = aug
        self.rng = random.Random(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, _idx):
        n_words = self.rng.randint(1, self.max_words)
        try:
            raw, spans, word_ids = build_synthetic_sentence(
                self.clips_by_class, self.classes, n_words=n_words,
                J=self.J, fps=self.fps, rng=self.rng)
        except Exception:
            # fallback to 1-word
            raw, spans, word_ids = build_synthetic_sentence(
                self.clips_by_class, self.classes, n_words=1,
                J=self.J, fps=self.fps, rng=self.rng)
        T = min(len(raw), self.max_len)
        raw = raw[:T]
        pose = torch.from_numpy(raw).view(T, self.J, 3)
        conf = torch.ones(T, self.J)
        valid = torch.ones(T, self.J, dtype=torch.bool)
        t = torch.arange(T, dtype=torch.float32) / self.fps
        # Clip spans to [0, T/fps]
        max_t = (T - 1) / self.fps
        spans = [(s, min(e, max_t)) for s, e in spans if s < max_t]
        return pose, conf, valid, t, spans, word_ids


def _sent_collate(batch):
    """Pads a batch of synthetic sentences; returns variable-length fields as lists."""
    poses, confs, valids, ts, spans_list, word_ids_list = zip(*batch)
    B = len(poses)
    T = max(p.shape[0] for p in poses)
    J = poses[0].shape[1]
    pose_b  = torch.zeros(B, T, J, 3)
    conf_b  = torch.zeros(B, T, J)
    valid_b = torch.zeros(B, T, J, dtype=torch.bool)
    t_b     = torch.zeros(B, T)
    lengths = torch.zeros(B, dtype=torch.long)
    for i, (p, c, v, t_) in enumerate(zip(poses, confs, valids, ts)):
        n = p.shape[0]
        pose_b[i, :n] = p; conf_b[i, :n] = c; valid_b[i, :n] = v
        t_b[i, :n] = t_; t_b[i, n:] = t_[-1]; lengths[i] = n
    return pose_b, conf_b, valid_b, t_b, lengths, list(spans_list), list(word_ids_list)


# ==============================================================================
# Training
# ==============================================================================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed); torch.manual_seed(args.seed)

    with open(args.classes) as f:
        classes = json.load(f)
    num_classes = len(classes)

    clips_by_class = load_word_clips(args.data_dir, classes, args.max_per_class)
    if not clips_by_class:
        raise RuntimeError(f"No word clips found in {args.data_dir}")
    print(f"[DATA] {sum(len(v) for v in clips_by_class.values())} clips across "
          f"{len(clips_by_class)} classes | {num_classes} total classes")

    ycfg = YAPConfig(d_model=args.d_model, n_mem_layers=args.n_mem_layers,
                     summary_channels=args.summary_channels, hippo_order=args.hippo_order,
                     entity_dim=args.entity_dim)
    ccfg = ConformerConfig(num_words=num_classes, d_model=args.d_model,
                           heads=args.heads, layers=args.layers)
    J = ycfg.total_joints

    model = YAPpingConformer(ycfg, ccfg).to(device)
    print(f"[MODEL] {count_parameters(model):,} parameters")

    # Load Stage 1 noise-robust weights
    if args.init and os.path.exists(args.init):
        sd = torch.load(args.init, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[INIT] {args.init} | missing {len(missing)}, unexpected {len(unexpected)}")
    if args.resume:
        ck_path = os.path.join(args.out_dir, "yap_seg_ckpt.pth")
        if os.path.exists(ck_path):
            ck = torch.load(ck_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            print(f"[RESUME] epoch {ck['epoch']}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    train_ds = SyntheticSentenceDataset(clips_by_class, classes, args.train_sentences, J,
                                         max_len=args.max_len, max_words=args.max_words, seed=args.seed)
    val_ds   = SyntheticSentenceDataset(clips_by_class, classes, args.val_sentences,   J,
                                         max_len=args.max_len, max_words=args.max_words, aug=False, seed=args.seed + 1)
    train_ld = DataLoader(train_ds, args.batch_size, shuffle=True,  collate_fn=_sent_collate, num_workers=0)
    val_ld   = DataLoader(val_ds,   args.batch_size, shuffle=False, collate_fn=_sent_collate, num_workers=0)

    os.makedirs(args.out_dir, exist_ok=True)
    best_wer = float("inf")

    for ep in range(args.epochs):
        model.train()
        ep_loss = 0.0; n_ok = 0

        for pose, conf, valid, t, lengths, spans_list, word_ids_list in train_ld:
            pose, conf, valid, t, lengths = (pose.to(device), conf.to(device),
                                              valid.to(device), t.to(device), lengths.to(device))
            B = pose.shape[0]

            # Build ground-truth segments from labelled spans (teacher forcing Stage 2)
            gt_segments = [make_segments(t[b, :int(lengths[b])].tolist(), spans_list[b], ycfg)
                           for b in range(B)]

            res = model(pose, conf, valid, t, lengths, segments=gt_segments)
            yap = res["yap"]

            # Presence loss (focal BCE with soft targets)
            pres_tgt = torch.stack([presence_targets(t[b, :int(lengths[b])], spans_list[b])
                                    for b in range(B)], dim=0)
            # Pad to full T
            pres_full = pose.new_zeros(B, pose.shape[1])
            for b in range(B):
                ln = int(lengths[b])
                pres_full[b, :ln] = pres_tgt[b]
            mask = torch.zeros_like(pres_full, dtype=torch.bool)
            for b in range(B):
                mask[b, :int(lengths[b])] = True
            l_pres = focal_presence_loss(yap["presence"], pres_full, mask.float())

            # End-head loss
            end_tgt = torch.stack([end_targets(t[b, :int(lengths[b])], spans_list[b])
                                   for b in range(B)])
            end_full = pose.new_zeros(B, pose.shape[1])
            for b in range(B):
                end_full[b, :int(lengths[b])] = end_tgt[b]
            l_end = F.binary_cross_entropy_with_logits(yap["end"], end_full,
                                                        pos_weight=torch.tensor(10.0, device=device))

            # Deep supervision on layer confidences
            l_layer = layer_conf_loss(yap["layer_conf"], pres_full, mask.float())

            # CTC on entity stream (entity = segment, word = target)
            word_seq_max = max(len(w) for w in word_ids_list)
            targets = torch.zeros(B, word_seq_max, dtype=torch.long, device=device)
            tgt_len = torch.zeros(B, dtype=torch.long, device=device)
            for b, wids in enumerate(word_ids_list):
                for k, w in enumerate(wids):
                    targets[b, k] = w + 1          # +1 because CTC blank = 0
                tgt_len[b] = len(wids)
            l_ctc = sentence_ctc_loss(res["word_logits"], res["token_mask"], targets, tgt_len)

            # Soft count + min-gap regulariser
            n_words_t = torch.tensor([len(w) for w in word_ids_list], dtype=torch.float32, device=device)
            l_count = soft_count_loss(yap["end"], n_words_t.long(), mask.float())
            l_gap   = min_gap_penalty(yap["end"], k=6, mask=mask.float())
            l_dip   = dip_prior_loss(yap["end"], yap["speed"], mask.float())

            loss = (l_pres + l_end + 0.5 * l_layer +
                    args.ctc_w * l_ctc +
                    0.1 * l_count + 0.05 * l_gap + 0.02 * l_dip)
            if torch.isfinite(loss):
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ep_loss += loss.item(); n_ok += 1

        scheduler.step()
        print(f"[Ep {ep+1:02d}/{args.epochs}] loss {ep_loss/max(1,n_ok):.4f} "
              f"| lr {scheduler.get_last_lr()[0]:.2e}")

        if (ep + 1) % 5 == 0 or ep + 1 == args.epochs:
            # Quick val: greedy decoding WER on synthetic val set
            model.eval()
            total_s, total_d, total_i, total_n = 0, 0, 0, 0
            with torch.no_grad():
                for pose, conf, valid, t, lengths, spans_list, word_ids_list in val_ld:
                    pose, conf, valid, t, lengths = (pose.to(device), conf.to(device),
                                                      valid.to(device), t.to(device), lengths.to(device))
                    res = model(pose, conf, valid, t, lengths)
                    wl = res["word_logits"]         # (B, K, V+1)
                    tm = res["token_mask"]           # (B, K)
                    for b in range(pose.shape[0]):
                        ref = [w + 1 for w in word_ids_list[b]]
                        hyp = wl[b, tm[b]].argmax(-1).tolist()
                        hyp = [h for h in hyp if h != 0]
                        import editdistance
                        ed = editdistance.eval(ref, hyp)
                        total_n += len(ref); total_s += ed
            val_wer = total_s / max(1, total_n)
            print(f"  val WER {val_wer*100:.1f}%")
            torch.save({"epoch": ep + 1, "model": model.state_dict()},
                       os.path.join(args.out_dir, "yap_seg_ckpt.pth"))
            if val_wer < best_wer:
                best_wer = val_wer
                torch.save(model.state_dict(), os.path.join(args.out_dir, "yap_seg_best.pth"))
                print(f"  ★ new best WER {best_wer*100:.1f}% -> yap_seg_best.pth")

    print(f"\nDone. Best val WER {best_wer*100:.1f}%")
    print(f"Stage 2 weights: {os.path.join(args.out_dir, 'yap_seg_best.pth')}")
    print("Pass these as --init to train_real_sentences.py for Stage 3.")


def parse_args():
    p = argparse.ArgumentParser(description="YAP Stage 2: Synthetic Sentence Segmentation")
    p.add_argument("--data_dir",       default=DEFAULT_DATA_DIR)
    p.add_argument("--classes",        default=DEFAULT_CLASSES)
    p.add_argument("--out_dir",        default=DEFAULT_OUT)
    p.add_argument("--init",           default=None,  help="Stage 1 yap_noise_robust.pth")
    p.add_argument("--resume",         action="store_true")
    p.add_argument("--epochs",         type=int,   default=60)
    p.add_argument("--batch_size",     type=int,   default=4)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--ctc_w",          type=float, default=1.0)
    p.add_argument("--train_sentences",type=int,   default=4000)
    p.add_argument("--val_sentences",  type=int,   default=400)
    p.add_argument("--max_len",        type=int,   default=600)
    p.add_argument("--max_words",      type=int,   default=5)
    p.add_argument("--max_per_class",  type=int,   default=15)
    p.add_argument("--seed",           type=int,   default=42)
    # YAP / Conformer architecture
    p.add_argument("--d_model",         type=int, default=128)
    p.add_argument("--n_mem_layers",    type=int, default=3)
    p.add_argument("--summary_channels",type=int, default=32)
    p.add_argument("--hippo_order",     type=int, default=16)
    p.add_argument("--entity_dim",      type=int, default=256)
    p.add_argument("--heads",           type=int, default=4)
    p.add_argument("--layers",          type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
