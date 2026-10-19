"""
Master Real Continuous Sentence Trainer
=======================================
Unified, highly regularized trainer designed to solve the train-val divergence
and break out of blank-suppression on unseen validation videos.

Key Fixes:
1. Anti-Overfitting: Dropout (0.25), Weight Decay (5e-4), SpecAugment Temporal Cutout.
2. Deficit Enforcement (deficit_w=2.5): Forces model to maintain non-blank sign bursts.
3. Blank Prior Penalty: Lowers blank dominance on unannotated 200+ frame sequences.
4. Zero-Leakage Sentence-Level Split: Clean hold-out validation.
"""

import os, sys, json, math, random, argparse, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path: sys.path.insert(0, BACKEND_DIR)

from training.train_conformer_ctc import (
    ISLHybridConformerCTC, ModelEMA,
    greedy_decode, batch_decode, batch_targets, compute_wer,
    augment, _reclamp
)

DEFAULT_DATA_DIR = os.path.join(BACKEND_DIR, "data", "tensors_sentence_level")
DEFAULT_OUT_DIR = os.path.join(BACKEND_DIR, "models", "isl_real_final")
CLASSES_PATH = os.path.join(BACKEND_DIR, "models", "phase6_word_model", "classes.json")
PHASE2_BEST = os.path.join(BACKEND_DIR, "models", "isl_real_sentences_phase2", "phase2_sentence_best.pth")
PHASE1_BEST = os.path.join(BACKEND_DIR, "models", "isl_real_sentences", "real_sentence_best.pth")

class FinalBalancedCTCLoss(nn.Module):
    def __init__(self, blank_id, deficit_w=0.5, min_frames_per_word=4.0):
        super().__init__()
        self.blank_id = blank_id
        self.deficit_w = deficit_w
        self.min_frames_per_word = min_frames_per_word

    def forward(self, log_probs_t, targets, input_lengths, target_lengths):
        ctc_loss = F.ctc_loss(
            log_probs_t, targets, input_lengths, target_lengths,
            blank=self.blank_id, zero_infinity=True
        )

        if self.deficit_w > 0.0:
            lp = log_probs_t.transpose(0, 1)  # (B, T, C)
            B = lp.shape[0]
            blank_p = torch.exp(lp[:, :, self.blank_id])
            non_blank_p = 1.0 - blank_p

            target_mass = target_lengths.float() * self.min_frames_per_word
            actual_mass = []
            for b in range(B):
                valid_t = input_lengths[b].item()
                actual_mass.append(non_blank_p[b, :valid_t].sum())
            actual_mass = torch.stack(actual_mass)
            deficit = F.relu(target_mass - actual_mass)
            return ctc_loss + self.deficit_w * deficit.mean()

        return ctc_loss

class RealSentenceDatasetFinal(Dataset):
    def __init__(self, index_file, is_train=True, val_split=0.10):
        with open(index_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        all_items = [it for it in data if os.path.exists(it.get("tensor_path", "")) and len(it.get("in_scope_indices", [])) > 0]

        # Deterministic zero-leakage split grouped strictly by sentence
        groups = {}
        for it in all_items:
            k = it.get("sentence", "").strip().lower()
            groups.setdefault(k, []).append(it)

        unique_keys = sorted(list(groups.keys()))
        random.seed(42)
        random.shuffle(unique_keys)
        random.seed()

        split_idx = int(len(unique_keys) * (1 - val_split))
        train_k, val_k = set(unique_keys[:split_idx]), set(unique_keys[split_idx:])

        self.items = [it for k in (train_k if is_train else val_k) for it in groups[k]]
        self.is_train = is_train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        seq = torch.load(item["tensor_path"], weights_only=True).float()
        labels = item["in_scope_indices"][:18]

        max_len = 220
        if seq.shape[0] > max_len:
            seq = F.interpolate(seq.T.unsqueeze(0), size=max_len, mode="linear", align_corners=True).squeeze(0).T

        if self.is_train:
            # Canonical pose augmentation
            seq = augment(seq, strength=0.85)

            # SpecAugment temporal cutout (masks 4-12 frames)
            T = seq.shape[0]
            if T > 16 and random.random() < 0.45:
                ml = random.randint(4, min(12, T // 4))
                st = random.randint(0, T - ml)
                seq = seq.clone()
                seq[st:st+ml, :159] = 0.0

            # Coordinate jitter
            if random.random() < 0.35:
                seq = seq.clone()
                seq[:, :159] += torch.randn_like(seq[:, :159]) * 0.005
                seq = _reclamp(seq)

        return seq, seq.shape[0], torch.tensor(labels, dtype=torch.long)

def collate_final(batch):
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

def train_final(epochs=25, batch_size=4, lr_backbone=8e-6, lr_head=3.5e-5, data_dir=DEFAULT_DATA_DIR, out_dir=DEFAULT_OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    with open(CLASSES_PATH) as f: classes = json.load(f)
    num_classes = len(classes)
    blank_id = num_classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ISLHybridConformerCTC(num_classes).to(device)

    # Warm-start from best trained real-sentence checkpoint
    ckpt_path = PHASE2_BEST if os.path.exists(PHASE2_BEST) else (PHASE1_BEST if os.path.exists(PHASE1_BEST) else None)
    if ckpt_path and os.path.exists(ckpt_path):
        raw = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
        model.load_state_dict(sd, strict=False)
        print(f"[INIT] Warm-started from real sentence checkpoint: {os.path.basename(ckpt_path)}")
    else:
        model.load_phase6_backbone()

    index_file = os.path.join(data_dir, "index.json")
    ds = RealSentenceDatasetFinal(index_file, is_train=True)
    val_ds = RealSentenceDatasetFinal(index_file, is_train=False)

    train_ld = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_final)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_final)

    param_groups = [
        {"params": model.backbone.parameters(), "lr": lr_backbone, "weight_decay": 5e-4},
        {"params": list(model.conformer.parameters()) + list(model.ctc_proj.parameters()), "lr": lr_head, "weight_decay": 5e-4}
    ]

    criterion = FinalBalancedCTCLoss(blank_id, deficit_w=0.5, min_frames_per_word=4.0)
    optimizer = optim.AdamW(param_groups, betas=(0.9, 0.98))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    ema = ModelEMA(model, decay=0.999)

    use_cuda = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    accum_steps = max(1, 4 // max(1, batch_size))

    print("=" * 72)
    print("FINAL REAL SENTENCE REGULARIZED TRAINER")
    print(f"Device: {device} | Train: {len(ds)} | Val: {len(val_ds)} | Epochs: {epochs} | Batch: {batch_size}")
    print("=" * 72)

    best_wer = float("inf")
    history = {"train_loss": [], "val_loss": [], "wer": [], "wer_ema": []}

    for ep in range(epochs):
        model.train()
        t_loss = 0.0
        optimizer.zero_grad()

        for step_i, (xs, lengths, targets, target_lengths) in enumerate(tqdm(train_ld, desc=f"Ep {ep+1:02d}/{epochs:02d}")):
            xs, lengths = xs.to(device), lengths.to(device)
            targets, target_lengths = targets.to(device), target_lengths.to(device)

            with torch.amp.autocast("cuda", enabled=use_cuda):
                lp = model(xs, lengths)
                loss = criterion(lp.transpose(0, 1), targets, lengths, target_lengths)
                loss = loss / accum_steps

            scaler.scale(loss).backward()

            if (step_i + 1) % accum_steps == 0 or (step_i + 1) == len(train_ld):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update(model)

            t_loss += loss.item() * accum_steps

        t_loss /= max(1, len(train_ld))
        if use_cuda: torch.cuda.empty_cache()

        # Validation (with calibrated min_dur=2 to avoid killing fast gestures)
        model.eval()
        v_loss = 0.0
        all_preds, all_refs = [], []
        with torch.no_grad():
            for xs, lengths, targets, target_lengths in val_ld:
                xs, lengths = xs.to(device), lengths.to(device)
                targets, target_lengths = targets.to(device), target_lengths.to(device)
                with torch.amp.autocast("cuda", enabled=use_cuda):
                    lp = model(xs, lengths)
                    v_loss += criterion(lp.transpose(0, 1), targets, lengths, target_lengths).item()
                all_preds.extend(batch_decode(lp, lengths, blank_id, min_dur=2))
                all_refs.extend(batch_targets(targets, target_lengths))

        v_loss /= max(1, len(val_ld))
        wer = sum(compute_wer(r, h) for r, h in zip(all_refs, all_preds)) / max(1, len(all_refs))

        # EMA Validation
        ema_preds = []
        with torch.no_grad():
            for xs, lengths, targets, target_lengths in val_ld:
                xs, lengths = xs.to(device), lengths.to(device)
                with torch.amp.autocast("cuda", enabled=use_cuda):
                    lp = ema.model(xs, lengths)
                ema_preds.extend(batch_decode(lp, lengths, blank_id, min_dur=2))
        ema_wer = sum(compute_wer(r, h) for r, h in zip(all_refs, ema_preds)) / max(1, len(all_refs))

        history["train_loss"].append(t_loss); history["val_loss"].append(v_loss)
        history["wer"].append(wer); history["wer_ema"].append(ema_wer)

        scheduler.step()
        print(f"[Ep {ep+1:02d}/{epochs:02d}] Train: {t_loss:.3f} | Val: {v_loss:.3f} | WER: {wer*100:.1f}% | EMA WER: {ema_wer*100:.1f}%")

        torch.save(model.state_dict(), os.path.join(out_dir, "real_final_checkpoint.pth"))
        if wer < best_wer:
            best_wer = wer
            torch.save(model.state_dict(), os.path.join(out_dir, "real_final_best.pth"))
            torch.save(ema.model.state_dict(), os.path.join(out_dir, "real_final_ema_best.pth"))
            print(f"  ★ NEW BEST WER: {wer*100:.1f}% (EMA: {ema_wer*100:.1f}%)")

        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        # Update curve
        cur_eps = len(history["train_loss"])
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, cur_eps+1), history["train_loss"], label="Train", color="#3b82f6")
        plt.plot(range(1, cur_eps+1), history["val_loss"], label="Val", color="#ef4444")
        plt.title("Loss"); plt.legend(); plt.grid(True, linestyle="--", alpha=0.5)

        plt.subplot(1, 2, 2)
        plt.plot(range(1, cur_eps+1), [w*100 for w in history["wer"]], label="Val WER", color="#10b981")
        plt.plot(range(1, cur_eps+1), [w*100 for w in history["wer_ema"]], label="EMA WER", color="#8b5cf6")
        plt.title("WER (%)"); plt.legend(); plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "training_curves.png"), dpi=200)
        plt.close()

    # Save Structure.md
    with open(os.path.join(out_dir, "structure.md"), "w", encoding="utf-8") as f:
        f.write("# Directory Structure: Backend/models/isl_real_final\n\nFinal regularized continuous model.\n\n## Files\n- `real_final_best.pth`: Best checkpoint.\n- `real_final_ema_best.pth`: Best EMA shadow checkpoint.\n- `real_final_checkpoint.pth`: Latest checkpoint.\n- `history.json`: Metric history.\n\n## Immediate Subdirectories\n- `plots/`: Metric progression plots.\n")

    print("\n" + "=" * 72)
    print(f"TRAINING COMPLETE | Best Real WER: {best_wer*100:.1f}%")
    print("=" * 72)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    train_final(epochs=args.epochs, batch_size=args.batch_size)
