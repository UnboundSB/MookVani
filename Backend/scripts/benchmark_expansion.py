"""Controlled comparison of the temporal-expansion architectures.

Variants
--------
  baseline     stride=1, the GRU smooths (residual)            -> the historical model
  compress     stride=2, no expansion                          -> the CTC head sees T/2 frames
  expand-before stride=2, the GRU expands BEFORE the attention -> "the transformer receives
                                                                    the expanded tensors"
  expand-after  stride=2, the attention runs on T/2 frames and the GRU expands back to T
                                                                    before the CTC head
                                                                    (the Conformer-CTC design)

Everything else is identical: the same data, the same seed, the same loss, the
same number of epochs, the same optimiser.

Run:  python scripts/benchmark_expansion.py --epochs 3 --samples 400
"""
import argparse
import json
import os
import random
import sys
import time

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import numpy as np
import torch

from training.train_conformer_ctc import (
    AntiCollapseCTCLoss,
    ISLHybridConformerCTC,
    SyntheticSentenceDataset,
    collate,
    compute_wer,
    greedy_decode,
)
from utils.dataloader_word import canonical_classes

DATA_DIR = os.path.join(BACKEND_DIR, "data", "tensors_include50_word_level")
PHASE6_WEIGHTS = os.path.join(BACKEND_DIR, "models", "phase6_word_model", "best_phase6_model.pth")
PHASE6_CLASSES = os.path.join(BACKEND_DIR, "models", "phase6_word_model", "classes.json")

VARIANTS = [
    ("baseline",      dict(stride=1, gru_expand="none")),
    ("compress",      dict(stride=2, gru_expand="none")),
    ("expand-before", dict(stride=2, gru_expand="before")),
    ("before+keepGRU", dict(stride=2, gru_expand="before_keep_gru")),
    ("expand-after",  dict(stride=2, gru_expand="after")),
]


def build_data(classes, samples, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    train_ds = SyntheticSentenceDataset(DATA_DIR, classes, samples_per_epoch=samples,
                                        is_train=True, max_words=6, min_words=1,
                                        aug_strength=0.5)
    val_ds = SyntheticSentenceDataset(DATA_DIR, classes, samples_per_epoch=max(80, samples // 4),
                                      is_train=False, max_words=6, min_words=1, aug_strength=0.0)
    return train_ds, val_ds


def evaluate(model, loader, device, blank_id, stride):
    model.eval()
    wer_sum, n = 0.0, 0
    with torch.no_grad():
        for xs, lengths, targets, target_lengths in loader:
            xs, lengths = xs.to(device), lengths.to(device)
            targets, target_lengths = targets.to(device), target_lengths.to(device)
            lp = model(xs, lengths)
            out_len = model.output_lengths(lengths)
            idx = 0
            for i in range(xs.size(0)):
                ref = targets[idx:idx + int(target_lengths[i])].tolist()
                idx += int(target_lengths[i])
                hyp = greedy_decode(lp[i], int(out_len[i]), blank_id,
                                    min_dur=max(1, 3 // max(1, stride)))
                wer_sum += compute_wer(ref, hyp)
                n += 1
    return wer_sum / max(1, n)


def latency(model, device, T=180, B=8, reps=30):
    x = torch.randn(B, T, 163, device=device)
    lengths = torch.full((B,), T, dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        # Two rounds: the first one pays the CUDA kernel-compilation cost, so
        # the second one is the number that matters.
        for _ in range(10):
            model(x, lengths)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            model(x, lengths)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t0) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(BACKEND_DIR, "eda", "plots",
                                                  "expansion_benchmark.json"))
    ap.add_argument("--warm_start", type=int, default=1,
                    help="Warm-start the backbone from the Phase-6 word model (the "
                         "real pipeline does this; without it the CTC collapses on "
                         "the blank and nothing can be compared)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = canonical_classes()
    print(f"device={device}  classes={len(classes)}  epochs={args.epochs}  "
          f"samples/epoch={args.samples}")

    train_ds, val_ds = build_data(classes, args.samples, args.seed)
    train_ld = torch.utils.data.DataLoader(train_ds, args.batch_size, shuffle=True,
                                           collate_fn=collate)
    val_ld = torch.utils.data.DataLoader(val_ds, args.batch_size, shuffle=False,
                                         collate_fn=collate)

    results = []
    for name, cfg in VARIANTS:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        model = ISLHybridConformerCTC(len(classes), **cfg).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        blank_id = len(classes)

        if args.warm_start and os.path.exists(PHASE6_WEIGHTS):
            # The real pipeline warm-starts from the Phase-6 word backbone AND
            # transfers the pretrained word classifiers into the CTC head BY
            # CLASS NAME.  Without the head transfer the CTC head starts from
            # scratch and collapses onto the blank (the documented stage-1
            # failure of this repo), so nothing can be compared.
            phase6_classes = json.load(open(PHASE6_CLASSES))
            if isinstance(phase6_classes, dict):
                phase6_classes = phase6_classes["classes"]
            src = {c: i for i, c in enumerate(phase6_classes)}
            class_map = {i: src[c] for i, c in enumerate(classes) if c in src}
            model.load_phase6_backbone(PHASE6_WEIGHTS, class_map=class_map)
            print(f"    warm-started from {os.path.basename(PHASE6_WEIGHTS)} "
                  f"({len(class_map)} head rows transferred by name)")

        # the emission barrier counts OUTPUT frames
        out_stride = cfg["stride"] if cfg["gru_expand"] in ("none",) else 1
        criterion = AntiCollapseCTCLoss(
            blank_id, blank_target=0.97, blank_w=0.5, emission_w=1.0,
            min_frames_per_sign=max(1.0, 10.0 / max(1, out_stride)), spike_w=0.10)

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        lat = latency(model, device)
        print(f"\n--- {name} ({cfg}) | params {n_params:,} | {lat:.1f} ms/batch of 8x180")
        wers, epoch_times = [], []
        for ep in range(1, args.epochs + 1):
            model.train()
            t0 = time.perf_counter()
            tot, nb = 0.0, 0
            for xs, lengths, targets, target_lengths in train_ld:
                xs, lengths = xs.to(device), lengths.to(device)
                targets, target_lengths = targets.to(device), target_lengths.to(device)
                opt.zero_grad()
                lp = model(xs, lengths)
                out_len = model.output_lengths(lengths)
                loss = criterion(lp.transpose(0, 1), targets, out_len, target_lengths)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += float(loss); nb += 1
            dt = time.perf_counter() - t0
            wer = evaluate(model, val_ld, device, blank_id, cfg["stride"])
            wers.append(wer)
            epoch_times.append(dt)
            print(f"    epoch {ep}: loss={tot/max(1,nb):8.3f}  val WER={wer*100:6.2f}%  "
                  f"({dt:.1f}s)")

        # the median epoch time is the reliable speed measure
        med_epoch = float(np.median(epoch_times[1:])) if len(epoch_times) > 1 else epoch_times[0]
        results.append({
            "name": name, "cfg": cfg, "params": n_params,
            "latency_ms_per_batch": round(lat, 2),
            "median_epoch_seconds": round(med_epoch, 2),
            "wer_per_epoch": [round(w * 100, 2) for w in wers],
            "best_wer": round(min(wers) * 100, 2),
        })

    print("\n" + "=" * 96)
    print(f"{'variant':<16}{'params':>12}{'epoch s':>10}   {'WER per epoch (last 6)':<40}{'best WER':>11}")
    print("=" * 96)
    for r in results:
        wers = ", ".join(f"{w:.0f}" for w in r["wer_per_epoch"][-6:])
        print(f"{r['name']:<16}{r['params']:>12,}{r['median_epoch_seconds']:>10.1f}"
              f"   {wers:<40}{r['best_wer']:>10.2f}%")
    print("=" * 96)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"epochs": args.epochs, "samples_per_epoch": args.samples,
                   "seed": args.seed, "results": results}, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
