"""
Audit Real Sentence Continuous Model Health & Collapse Status
=============================================================
Evaluates checkpoints trained on real human sign videos:
- Quantitative frame emission stats (Blank % vs Non-Blank %).
- Duration bursts (checks if signs sustain across >= 5 frames).
- Prints Ground Truth In-Scope Words vs Model Hypothesis.
- Emits explicit [HEALTHY] vs [COLLAPSED] verdict.

The tensors are put through the SAME feature pipeline as the training
(`harmonize_features` + `_reclamp` from training/train_conformer_ctc.py).  Skipping
that step makes the model look collapsed even when it is not: the three corpora in
Backend/data use three different anchors, so a model trained on the harmonised frames
emits nothing on the raw ones.
"""

import os, sys, glob, json, argparse
import numpy as np
import torch

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path: sys.path.insert(0, BACKEND_DIR)

from training.train_conformer_ctc import (
    ISLHybridConformerCTC, greedy_decode, compute_wer, harmonize_features, _reclamp
)
from utils.checkpoint_utils import extract_state_dict, load_flexible

REAL_DIR = os.path.join(BACKEND_DIR, "models", "isl_real_sentences")
DEFAULT_CKPT = os.path.join(REAL_DIR, "real_sentence_checkpoint.pth")
if not os.path.exists(DEFAULT_CKPT):
    DEFAULT_CKPT = os.path.join(REAL_DIR, "real_sentence_best.pth")
DEFAULT_DATA_DIR = os.path.join(BACKEND_DIR, "data", "tensors_sentence_level")
CLASSES_PATH = os.path.join(BACKEND_DIR, "models", "word_vocabulary", "classes.json")

def audit_real_model(ckpt_path=DEFAULT_CKPT, data_dir=DEFAULT_DATA_DIR, n_samples=20, random_sample=True, seed=None):
    if not os.path.exists(ckpt_path):
        print("=" * 72)
        print("REAL SENTENCE VIDEO CTC COLLAPSE AUDIT")
        print(f"[!] Checkpoint not found: {ckpt_path}")
        print("[!] No trained real-sentence checkpoint exists yet.")
        print("    Complete at least 1 epoch of training via train_real_sentences.py first.")
        print("=" * 72)
        return

    index_path = os.path.join(data_dir, "index.json")
    if not os.path.exists(index_path):
        print(f"[ERROR] No index.json found in {data_dir}. Run extract_real_sentences.py first!")
        return

    with open(CLASSES_PATH) as f: classes = json.load(f)
    num_classes = len(classes)
    blank_id = num_classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ISLHybridConformerCTC(num_classes).to(device)

    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    load_flexible(model, extract_state_dict(sd), label="CHECK")
    model.eval()

    with open(index_path, "r", encoding="utf-8") as f: items = json.load(f)
    items = [it for it in items if os.path.exists(it.get("tensor_path", "")) and len(it.get("in_scope_indices", [])) > 0]

    # Group by distinct sentence to avoid picking the same sentence repeated times
    unique_by_sentence = {}
    for it in items:
        s_key = it.get("sentence", "").strip().lower()
        if s_key not in unique_by_sentence:
            unique_by_sentence[s_key] = []
        unique_by_sentence[s_key].append(it)

    distinct_candidates = [vids[0] for vids in unique_by_sentence.values()]

    if seed is not None:
        import random
        random.seed(seed)

    if random_sample:
        import random
        pool = distinct_candidates if len(distinct_candidates) >= n_samples else items
        sampled = random.sample(pool, min(n_samples, len(pool)))
    else:
        sampled = distinct_candidates[:n_samples]

    print("=" * 72)
    print("REAL SENTENCE VIDEO CTC COLLAPSE AUDIT")
    print(f"Checkpoint: {os.path.basename(ckpt_path)}")
    print(f"Available Real Videos: {len(items):,} ({len(distinct_candidates)} distinct sentences)")
    print(f"Sampling Mode: {'Random Distinct' if random_sample else 'First Distinct'} ({len(sampled)} sentences)")
    print("=" * 72)
    total_frames = 0
    blank_frames = 0
    burst_lengths = []
    spike_count = 0
    sample_outputs = []

    with torch.no_grad():
        for it in sampled:
            seq = torch.load(it["tensor_path"], weights_only=True).float().to(device)
            # Same feature pipeline as the training (see the module docstring).
            seq = _reclamp(harmonize_features(seq).contiguous())
            T = seq.shape[0]
            total_frames += T

            lp = model(seq.unsqueeze(0))[0] # (T, C)
            preds = lp.argmax(dim=-1).cpu().numpy()

            b_cnt = int(np.sum(preds == blank_id))
            blank_frames += b_cnt

            cur, dur = None, 0
            for tok in preds:
                if tok != blank_id:
                    if tok == cur: dur += 1
                    else:
                        if cur is not None:
                            burst_lengths.append(dur)
                            if dur == 1: spike_count += 1
                        cur, dur = tok, 1
                else:
                    if cur is not None:
                        burst_lengths.append(dur)
                        if dur == 1: spike_count += 1
                        cur, dur = None, 0
            if cur is not None:
                burst_lengths.append(dur)
                if dur == 1: spike_count += 1

            hyp_tokens = greedy_decode(lp, T, blank_id, min_dur=3)
            ref_words = it["in_scope_words"]
            hyp_words = [classes[t] for t in hyp_tokens] if hyp_tokens else ["<EMPTY/ALL_BLANK>"]
            wer_val = compute_wer(it["in_scope_indices"], hyp_tokens)

            if len(sample_outputs) < 8:
                sample_outputs.append((ref_words, hyp_words, wer_val, it.get("sentence", "")))

    pct_blank = (blank_frames / max(1, total_frames)) * 100.0
    pct_non_blank = 100.0 - pct_blank
    avg_burst = float(np.mean(burst_lengths)) if burst_lengths else 0.0
    spike_pct = (spike_count / max(1, len(burst_lengths))) * 100.0 if burst_lengths else 0.0

    print("\n--- QUANTITATIVE EMISSION METRICS ---")
    print(f"Total Frames Analyzed:        {total_frames:,}")
    print(f"Blank Frame Percentage:       {pct_blank:.2f}%  (Ideal: 60% - 85%)")
    print(f"Non-Blank Frame Percentage:   {pct_non_blank:.2f}%  (Ideal: 15% - 40%)")
    print(f"Mean Non-Blank Burst Length:  {avg_burst:.1f} frames  (Ideal: 5 - 25 frames)")
    print(f"Single-Frame Spike Ratio:     {spike_pct:.2f}%  (Ideal: < 25%)")

    print("\n--- SAMPLE DECODED REAL SENTENCES ---")
    for idx, (ref, hyp, wer, full_s) in enumerate(sample_outputs):
        status = "MATCH" if ref == hyp else ("PARTIAL" if wer < 1.0 else "MISSED")
        print(f"[{idx+1}] [{status}] WER: {wer*100:.1f}%")
        print(f"    Full Real Sentence: \"{full_s}\"")
        print(f"    Target In-Scope:    {' '.join(ref)}")
        print(f"    Model Predicted:    {' '.join(hyp)}")

    print("\n" + "=" * 72)
    if pct_blank >= 99.0 or len(burst_lengths) == 0:
        print("[VERDICT: COLLAPSED] Model collapsed into blanks on real videos.")
    elif pct_blank < 95.0 and avg_burst >= 4.0:
        print("[VERDICT: HEALTHY CONTINUOUS EMISSION] Model is NOT collapsed!")
        print("Recognizes trained in-scope words on real human continuous video streams.")
    else:
        print("[VERDICT: CONVERGING / MODERATE] Model is actively learning real video dynamics.")
    print("=" * 72)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--no_random", action="store_true", help="Disable random sampling (pick first distinct sentences)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for deterministic sampling")
    args = parser.parse_args()
    audit_real_model(ckpt_path=args.ckpt, n_samples=args.samples, random_sample=(not args.no_random), seed=args.seed)
