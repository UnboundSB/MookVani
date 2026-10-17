"""
Real Continuous Sentence Trainer
================================
Fine-tunes ISL Hybrid Conformer-CTC on real continuous human sign videos.

Features:
1. Selective Reward/Penalty Relabeling:
   - Evaluates and penalizes ONLY on in-scope trained words.
   - Out-of-scope words in the middle naturally align to CTC blank/transition.
2. Anti-Collapse Regularization:
   - Linear duration deficit penalty prevents blank collapse.
   - Blank silence injection (idle resting frames with empty targets) anchors silence.
3. Discriminative Fine-Tuning:
   - Low backbone LR (8e-6) + Conformer LR (5e-5).
   - Model EMA tracking for smooth evaluation.
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
    ISLHybridConformerCTC, ConformityCTCLoss, ModelEMA,
    greedy_decode, batch_decode, batch_targets, compute_wer,
    augment, _reclamp
)

DEFAULT_DATA_DIR = os.path.join(BACKEND_DIR, "data", "tensors_sentence_level")
DEFAULT_OUT_DIR = os.path.join(BACKEND_DIR, "models", "isl_real_sentences")
REAL_BEST = os.path.join(DEFAULT_OUT_DIR, "real_sentence_best.pth")
STAGE3_BEST = os.path.join(BACKEND_DIR, "models", "isl_conformer_ctc", "stage3_best.pth")
STAGE1_BEST = os.path.join(BACKEND_DIR, "models", "isl_conformer_ctc", "stage1_best.pth")

if os.path.exists(REAL_BEST):
    SOURCE_CONFORMER = REAL_BEST
elif os.path.exists(STAGE3_BEST):
    SOURCE_CONFORMER = STAGE3_BEST
else:
    SOURCE_CONFORMER = STAGE1_BEST
CLASSES_PATH = os.path.join(BACKEND_DIR, "models", "phase6_word_model", "classes.json")

class RealSentenceDataset(Dataset):
    def __init__(self, index_file, is_train=True, blank_injection_p=0.05):
        with open(index_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Filter valid items
        self.items = [item for item in data if os.path.exists(item.get("tensor_path", "")) and len(item.get("in_scope_indices", [])) > 0]
        self.is_train = is_train
        self.blank_p = blank_injection_p if is_train else 0.0

        # Collect rest frame segments for blank injection
        self.rest_segments = []
        for it in self.items[:30]:
            try:
                t = torch.load(it["tensor_path"], weights_only=True)
                if t.shape[0] >= 12:
                    self.rest_segments.append(t[:6].float())
                    self.rest_segments.append(t[-6:].float())
            except Exception:
                pass

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        # 1. Blank sentence injection: idle resting frames with target = []
        if self.is_train and random.random() < self.blank_p and self.rest_segments:
            seg = random.choice(self.rest_segments)
            noise = torch.randn_like(seg[:, :159]) * 0.005
            seg = seg.clone()
            seg[:, :159] += noise
            return _reclamp(seg), seg.shape[0], torch.tensor([], dtype=torch.long)

        item = self.items[idx]
        seq = torch.load(item["tensor_path"], weights_only=True).float()
        labels = item["in_scope_indices"]

        # Temporal sub-sampling to prevent quadratic attention OOM on long sentences
        max_len = 160
        if seq.shape[0] > max_len:
            step = int(math.ceil(seq.shape[0] / max_len))
            seq = seq[::step][:max_len]

        if self.is_train:
            seq = augment(seq, strength=0.5)

        return seq, seq.shape[0], torch.tensor(labels, dtype=torch.long)

def collate_real(batch):
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

def train_real_sentences(epochs=20, batch_size=4, lr_backbone=8e-6, lr_head=5e-5, data_dir=DEFAULT_DATA_DIR, out_dir=DEFAULT_OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    with open(CLASSES_PATH) as f: classes = json.load(f)
    num_classes = len(classes)
    blank_id = num_classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ISLHybridConformerCTC(num_classes).to(device)

    # Load pretrained Conformer-CTC weights
    if os.path.exists(SOURCE_CONFORMER):
        raw = torch.load(SOURCE_CONFORMER, map_location=device, weights_only=False)
        sd = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
        model.load_state_dict(sd, strict=False)
        print(f"[WARM-START] Loaded Conformer-CTC weights from {os.path.basename(SOURCE_CONFORMER)}")
    else:
        model.load_phase6_backbone()

    index_file = os.path.join(data_dir, "index.json")
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"Missing index.json in {data_dir}. Run extract_real_sentences.py first!")

    ds = RealSentenceDataset(index_file, is_train=True)
    val_ds = RealSentenceDataset(index_file, is_train=False)

    train_ld = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_real)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_real)

    param_groups = [
        {"params": model.backbone.parameters(), "lr": lr_backbone, "weight_decay": 1e-4},
        {"params": list(model.conformer.parameters()) + list(model.ctc_proj.parameters()), "lr": lr_head, "weight_decay": 1e-4}
    ]

    criterion = ConformityCTCLoss(blank_id, connectivity_w=0.10, deficit_w=0.2, min_frames_per_word=6.0)
    optimizer = optim.AdamW(param_groups, betas=(0.9, 0.98))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    ema = ModelEMA(model, decay=0.999)
    use_cuda = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    accum_steps = max(1, 4 // max(1, batch_size))

    print("=" * 72)
    print("STARTING REAL SENTENCE TRAINING")
    print(f"Device: {device} | Total Sentences: {len(ds)} | Epochs: {epochs} | Batch: {batch_size} (Accum: {accum_steps})")
    print("=" * 72)

    best_wer = float("inf")
    history = {"train_loss":[], "val_loss":[], "wer":[], "wer_ema":[]}

    for ep in range(epochs):
        model.train()
        t_loss = 0.0
        optimizer.zero_grad()

        for step_i, (xs, lengths, targets, target_lengths) in enumerate(tqdm(train_ld, desc=f"Ep {ep+1:02d}/{epochs:02d} Train")):
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

        if use_cuda:
            torch.cuda.empty_cache()

        # Evaluation
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
                all_preds.extend(batch_decode(lp, lengths, blank_id, min_dur=3))
                all_refs.extend(batch_targets(targets, target_lengths))

        v_loss /= max(1, len(val_ld))
        wer = sum(compute_wer(r, h) for r, h in zip(all_refs, all_preds)) / max(1, len(all_refs))

        # EMA Eval
        ema_preds = []
        with torch.no_grad():
            for xs, lengths, targets, target_lengths in val_ld:
                xs, lengths = xs.to(device), lengths.to(device)
                with torch.amp.autocast("cuda", enabled=use_cuda):
                    lp = ema.model(xs, lengths)
                ema_preds.extend(batch_decode(lp, lengths, blank_id, min_dur=3))
        ema_wer = sum(compute_wer(r, h) for r, h in zip(all_refs, ema_preds)) / max(1, len(all_refs))

        if use_cuda:
            torch.cuda.empty_cache()

        history["train_loss"].append(t_loss); history["val_loss"].append(v_loss)
        history["wer"].append(wer); history["wer_ema"].append(ema_wer)

        scheduler.step()
        print(f"[Ep {ep+1:02d}/{epochs:02d}] Train Loss: {t_loss:.3f} | Val Loss: {v_loss:.3f} | WER: {wer*100:.1f}% | EMA WER: {ema_wer*100:.1f}%")

        # Save latest checkpoint each epoch
        torch.save(model.state_dict(), os.path.join(out_dir, "real_sentence_checkpoint.pth"))

        if wer < best_wer:
            best_wer = wer
            torch.save(model.state_dict(), os.path.join(out_dir, "real_sentence_best.pth"))
            torch.save(ema.model.state_dict(), os.path.join(out_dir, "real_sentence_ema_best.pth"))

        # Save history and update training curves live each epoch
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        cur_epochs = len(history["train_loss"])
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, cur_epochs + 1), history["train_loss"], label="Train Loss", marker="o", color="#3b82f6")
        plt.plot(range(1, cur_epochs + 1), history["val_loss"], label="Val Loss", marker="s", color="#ef4444")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("CTC Loss Progression")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)

        plt.subplot(1, 2, 2)
        plt.plot(range(1, cur_epochs + 1), [w * 100 for w in history["wer"]], label="Val WER (%)", marker="o", color="#10b981")
        plt.plot(range(1, cur_epochs + 1), [w * 100 for w in history["wer_ema"]], label="EMA WER (%)", marker="s", color="#8b5cf6")
        plt.xlabel("Epoch")
        plt.ylabel("WER (%)")
        plt.title("Word Error Rate Progression")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plot_file = os.path.join(plots_dir, "real_sentence_training_curves.png")
        plt.savefig(plot_file, dpi=200)
        plt.close()

    # Save Structure.md
    with open(os.path.join(out_dir, "structure.md"), "w", encoding="utf-8") as f:
        f.write("# Directory Structure: Backend/models/isl_real_sentences\n\nReal continuous human sign recognition model.\n\n## Files\n- `real_sentence_best.pth`: Best checkpoint.\n- `real_sentence_ema_best.pth`: Best EMA shadow checkpoint.\n- `history.json`: Training metrics.\n\n## Immediate Subdirectories\n- `plots/`: Contains training loss and WER progression curves (`real_sentence_training_curves.png`).\n")

    print("\n" + "=" * 72)
    print(f"TRAINING COMPLETE | Best Real Video WER: {best_wer*100:.1f}%")
    print("=" * 72)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()
    train_real_sentences(epochs=args.epochs, batch_size=args.batch_size)
