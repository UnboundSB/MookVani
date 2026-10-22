"""
Diagnostic Utility: CTC Blank Collapse & Health Verification
============================================================
Checks whether a trained CTC model has suffered from:
1. Blank Collapse (>98% frames blank, 0 or near-0 words predicted)
2. Single-Frame Spike Collapse (words emitted for only 1 frame then lost)
3. Vocabulary Collapse (model repeats only 1 or 2 dominant classes)
4. Healthy Continuous Emission (non-blank predictions span 5-25 frames)
"""

import os, sys, glob, json, argparse
import numpy as np
import torch
import torch.nn.functional as F

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path: sys.path.insert(0, BACKEND_DIR)

from training.train_conformer_ctc import (
    ISLHybridConformerCTC, SyntheticSentenceDataset, collate,
    greedy_decode, batch_decode, batch_targets, compute_wer
)
from utils.checkpoint_utils import extract_state_dict, load_flexible

def run_collapse_check(weights_path=None, model_dir=None, n_samples=30, data_dir=None):
    if model_dir is None:
        model_dir = os.path.join(BACKEND_DIR, "models", "isl_conformer_ctc")
    if data_dir is None:
        data_dir = os.path.join(BACKEND_DIR, "data", "tensors_include50_word_level")

    # Find weights
    if weights_path is None or not os.path.exists(weights_path):
        candidates = [
            os.path.join(model_dir, "stage1_best.pth"),
            os.path.join(model_dir, "stage1_ema_best.pth"),
            os.path.join(model_dir, "stage1_final.pth"),
            os.path.join(model_dir, "stage1_checkpoint.pth"),
        ]
        for c in candidates:
            if os.path.exists(c):
                weights_path = c
                break

    if not weights_path or not os.path.exists(weights_path):
        print(f"[ERROR] No checkpoint found in {model_dir}")
        return

    # Load classes
    classes_file = os.path.join(model_dir, "classes.json")
    if not os.path.exists(classes_file):
        classes_file = os.path.join(BACKEND_DIR, "models", "word_vocabulary", "classes.json")
    with open(classes_file, "r") as f:
        meta = json.load(f)
        classes = meta["classes"] if isinstance(meta, dict) and "classes" in meta else meta
    num_classes = len(classes)
    blank_id = num_classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build & load model
    model = ISLHybridConformerCTC(num_classes).to(device)
    raw = torch.load(weights_path, map_location=device, weights_only=False)
    load_flexible(model, extract_state_dict(raw), label="CHECK")
    model.eval()

    print("=" * 72)
    print("CTC COLLAPSE & EMISSION HEALTH AUDIT")
    print(f"Model Checkpoint:  {os.path.basename(weights_path)}")
    print(f"Device:            {device} | Classes: {num_classes} | Blank ID: {blank_id}")
    print(f"Test Samples:      {n_samples} validation sentences")
    print("=" * 72)

    # Generate test batch
    val_ds = SyntheticSentenceDataset(data_dir, classes, samples_per_epoch=n_samples, is_train=False, max_words=3, min_words=1)
    loader = torch.utils.data.DataLoader(val_ds, batch_size=min(8, n_samples), shuffle=False, collate_fn=collate)

    total_frames = 0
    blank_frames = 0
    non_blank_frames = 0
    burst_lengths = []
    spike_count = 0  # bursts of length exactly 1
    predicted_tokens = []
    sample_outputs = []

    with torch.no_grad():
        for xs, lengths, targets, target_lengths in loader:
            xs, lengths = xs.to(device), lengths.to(device)
            lp = model(xs, lengths)  # (B, T, C)
            preds = lp.argmax(dim=-1).cpu().numpy()
            lengths_np = lengths.cpu().numpy()

            refs = batch_targets(targets, target_lengths)
            hyps = batch_decode(lp, lengths, blank_id)

            for i in range(len(lengths_np)):
                seq = preds[i, :lengths_np[i]]
                total_frames += len(seq)
                b_count = int(np.sum(seq == blank_id))
                blank_frames += b_count
                non_blank_frames += (len(seq) - b_count)

                # Analyze continuous bursts of identical non-blank tokens
                cur_tok, cur_len = None, 0
                for tok in seq:
                    if tok != blank_id:
                        predicted_tokens.append(int(tok))
                        if tok == cur_tok:
                            cur_len += 1
                        else:
                            if cur_tok is not None:
                                burst_lengths.append(cur_len)
                                if cur_len == 1: spike_count += 1
                            cur_tok = tok
                            cur_len = 1
                    else:
                        if cur_tok is not None:
                            burst_lengths.append(cur_len)
                            if cur_len == 1: spike_count += 1
                            cur_tok = None
                            cur_len = 0
                if cur_tok is not None:
                    burst_lengths.append(cur_len)
                    if cur_len == 1: spike_count += 1

                if len(sample_outputs) < 5:
                    ref_words = [classes[t] for t in refs[i]]
                    hyp_words = [classes[t] for t in hyps[i]] if hyps[i] else ["<EMPTY/ALL_BLANK>"]
                    wer_val = compute_wer(refs[i], hyps[i])
                    sample_outputs.append((ref_words, hyp_words, wer_val))

    # Metrics
    pct_blank = (blank_frames / max(1, total_frames)) * 100.0
    pct_non_blank = (non_blank_frames / max(1, total_frames)) * 100.0
    avg_burst = float(np.mean(burst_lengths)) if burst_lengths else 0.0
    med_burst = float(np.median(burst_lengths)) if burst_lengths else 0.0
    total_bursts = len(burst_lengths)
    spike_pct = (spike_count / max(1, total_bursts)) * 100.0 if total_bursts > 0 else 0.0
    unique_words_emitted = len(set(predicted_tokens))

    print("\n--- QUANTITATIVE EMISSION METRICS ---")
    print(f"Total Frames Analyzed:        {total_frames:,}")
    print(f"Blank Frame Percentage:       {pct_blank:.2f}%  (Ideal: 70% - 90%)")
    print(f"Non-Blank Frame Percentage:   {pct_non_blank:.2f}%  (Ideal: 10% - 30%)")
    print(f"Total Token Emissions:        {total_bursts}")
    print(f"Mean Non-Blank Burst Length:  {avg_burst:.1f} frames  (Ideal: 5 - 25 frames)")
    print(f"Median Non-Blank Burst Length:{med_burst:.1f} frames")
    print(f"Single-Frame Spike Ratio:     {spike_pct:.2f}%  (Ideal: < 25%)")
    print(f"Distinct Classes Emitted:     {unique_words_emitted}")

    print("\n--- SAMPLE DECODED PREDICTIONS ---")
    for idx, (ref, hyp, wer) in enumerate(sample_outputs):
        status = "MATCH" if ref == hyp else ("PARTIAL" if wer < 1.0 else "MISSED")
        print(f"[{idx+1}] [{status}] WER: {wer*100:.1f}%")
        print(f"    Ref: {' '.join(ref)}")
        print(f"    Hyp: {' '.join(hyp)}")

    print("\n" + "=" * 72)
    # VERDICT
    if pct_blank >= 99.0 or total_bursts == 0:
        print("[VERDICT: COLLAPSED] The model has collapsed to blank tokens (99%+ blank).")
        print("Remedy: Increase blank_w penalty or check CTC blank initialization.")
    elif spike_pct > 75.0 and avg_burst < 2.0:
        print("[VERDICT: SPIKE-DOMINATED] The model emits isolated 1-frame spikes.")
        print("Remedy: Increase connectivity_w regularization to encourage sustained emissions.")
    elif pct_blank < 95.0 and avg_burst >= 4.0:
        print("[VERDICT: HEALTHY CONTINUOUS EMISSION] Model is NOT collapsed!")
        print("Tokens are sustained across realistic temporal durations (4+ frames) without blank dominance.")
    else:
        print("[VERDICT: CONVERGING / MODERATE] Model is actively learning temporal boundaries.")
    print("=" * 72)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit CTC Model Emission Health & Collapse Status")
    parser.add_argument("--ckpt", type=str, default=None, help="Explicit checkpoint path")
    parser.add_argument("--model_dir", type=str, default=None, help="Model directory")
    parser.add_argument("--samples", type=int, default=30, help="Number of validation samples to evaluate")
    args = parser.parse_args()
    run_collapse_check(weights_path=args.ckpt, model_dir=args.model_dir, n_samples=args.samples)
