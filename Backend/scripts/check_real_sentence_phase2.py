"""
Phase 2 Real Continuous Sentence Zero-Leakage Checker
====================================================
Audits Phase 2 continuous checkpoints on the UNSEEN VALIDATION SET:
- Zero Data Leakage: Evaluates strictly on the 10% hold-out validation partition (seed=42).
- Quantitative emission analysis (Blank % vs Non-Blank %, Burst durations).
- Displays Ground Truth In-Scope Words vs Model Hypothesis.
- Emits explicit [HEALTHY CONTINUOUS EMISSION] vs [LOCAL MINIMUM] vs [COLLAPSED] verdict.
"""

import os, sys, json, argparse, random
import numpy as np
import torch

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path: sys.path.insert(0, BACKEND_DIR)

from training.train_conformer_ctc import ISLHybridConformerCTC, greedy_decode, compute_wer

FINAL_DIR = os.path.join(BACKEND_DIR, "models", "isl_real_final")
PHASE2_DIR = os.path.join(BACKEND_DIR, "models", "isl_real_sentences_phase2")
PHASE1_DIR = os.path.join(BACKEND_DIR, "models", "isl_real_sentences")

DEFAULT_CKPT = os.path.join(FINAL_DIR, "real_final_best.pth")
if not os.path.exists(DEFAULT_CKPT):
    DEFAULT_CKPT = os.path.join(FINAL_DIR, "real_final_ema_best.pth")
if not os.path.exists(DEFAULT_CKPT):
    DEFAULT_CKPT = os.path.join(PHASE2_DIR, "phase2_sentence_best.pth")
if not os.path.exists(DEFAULT_CKPT):
    DEFAULT_CKPT = os.path.join(PHASE2_DIR, "phase2_sentence_checkpoint.pth")
if not os.path.exists(DEFAULT_CKPT):
    DEFAULT_CKPT = os.path.join(PHASE1_DIR, "real_sentence_best.pth")

DEFAULT_DATA_DIR = os.path.join(BACKEND_DIR, "data", "tensors_sentence_level")
CLASSES_PATH = os.path.join(BACKEND_DIR, "models", "phase6_word_model", "classes.json")

def get_zero_leakage_split(index_path, val_split=0.10, seed=42):
    """Recreates the exact deterministic sentence-level split used by training."""
    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_items = [it for it in data if os.path.exists(it.get("tensor_path", "")) and len(it.get("in_scope_indices", [])) > 0]

    sentence_groups = {}
    for it in all_items:
        s_key = it.get("sentence", "").strip().lower()
        if s_key not in sentence_groups:
            sentence_groups[s_key] = []
        sentence_groups[s_key].append(it)

    unique_sentences = sorted(list(sentence_groups.keys()))
    random.seed(seed)
    random.shuffle(unique_sentences)
    random.seed()

    split_idx = int(len(unique_sentences) * (1 - val_split))
    train_keys = set(unique_sentences[:split_idx])
    val_keys = set(unique_sentences[split_idx:])

    train_items = [it for k in train_keys for it in sentence_groups[k]]
    val_items = [it for k in val_keys for it in sentence_groups[k]]

    return train_items, val_items

def audit_phase2(ckpt_path=DEFAULT_CKPT, data_dir=DEFAULT_DATA_DIR, n_samples=15, split="val", seed=None, min_dur=2):
    if not os.path.exists(ckpt_path):
        print("=" * 72)
        print("PHASE 2 REAL SENTENCE CONTINUOUS AUDIT")
        print(f"[!] Checkpoint not found: {ckpt_path}")
        print("    Run training/train_real_sentences_phase2.py first.")
        print("=" * 72)
        return

    index_path = os.path.join(data_dir, "index.json")
    if not os.path.exists(index_path):
        print(f"[ERROR] Missing index.json in {data_dir}")
        return

    with open(CLASSES_PATH) as f: classes = json.load(f)
    num_classes = len(classes)
    blank_id = num_classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ISLHybridConformerCTC(num_classes).to(device)

    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "model_state_dict" in sd: sd = sd["model_state_dict"]
    model.load_state_dict(sd, strict=False)
    model.eval()

    # Get zero-leakage partitions
    train_items, val_items = get_zero_leakage_split(index_path)

    if split == "val":
        pool = val_items
        split_desc = "UNSEEN VALIDATION SET (ZERO-LEAKAGE)"
    elif split == "train":
        pool = train_items
        split_desc = "TRAINING SET"
    else:
        pool = train_items + val_items
        split_desc = "FULL DATASET"

    # Group distinct sentences in pool
    unique_pool = {}
    for it in pool:
        s_key = it.get("sentence", "").strip().lower()
        if s_key not in unique_pool:
            unique_pool[s_key] = it

    distinct_candidates = list(unique_pool.values())

    if seed is not None:
        random.seed(seed)
    sampled = random.sample(distinct_candidates, min(n_samples, len(distinct_candidates)))

    print("=" * 72)
    print("PHASE 2 REAL SENTENCE ZERO-LEAKAGE AUDIT")
    print(f"Checkpoint:       {os.path.basename(ckpt_path)}")
    print(f"Evaluation Split: {split_desc} ({len(pool):,} videos, {len(distinct_candidates):,} distinct sentences)")
    print(f"Audit Sample:     {len(sampled)} distinct sentences")
    print("=" * 72)

    total_frames = 0
    blank_frames = 0
    burst_lengths = []
    spike_count = 0
    sample_outputs = []
    all_wers = []

    with torch.no_grad():
        for it in sampled:
            seq = torch.load(it["tensor_path"], weights_only=True).float().to(device)
            T = seq.shape[0]
            total_frames += T

            lp = model(seq.unsqueeze(0))[0]  # (T, C)
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

            hyp_tokens = greedy_decode(lp, T, blank_id, min_dur=min_dur)
            ref_words = it["in_scope_words"]
            hyp_words = [classes[t] for t in hyp_tokens] if hyp_tokens else ["<EMPTY/ALL_BLANK>"]
            wer_val = compute_wer(it["in_scope_indices"][:18], hyp_tokens)
            all_wers.append(wer_val)

            if len(sample_outputs) < 10:
                sample_outputs.append((ref_words, hyp_words, wer_val, it.get("sentence", "")))

    pct_blank = (blank_frames / max(1, total_frames)) * 100.0
    pct_non_blank = 100.0 - pct_blank
    avg_burst = float(np.mean(burst_lengths)) if burst_lengths else 0.0
    spike_pct = (spike_count / max(1, len(burst_lengths))) * 100.0 if burst_lengths else 0.0
    mean_wer = float(np.mean(all_wers)) if all_wers else 1.0

    print("\n--- QUANTITATIVE EMISSION METRICS ---")
    print(f"Total Frames Analyzed:        {total_frames:,}")
    print(f"Blank Frame Percentage:       {pct_blank:.2f}%  (Ideal: 70% - 90%)")
    print(f"Non-Blank Frame Percentage:   {pct_non_blank:.2f}%  (Ideal: 10% - 30%)")
    print(f"Mean Non-Blank Burst Length:  {avg_burst:.1f} frames  (Ideal: 5 - 20 frames)")
    print(f"Single-Frame Spike Ratio:     {spike_pct:.2f}%  (Ideal: < 20%)")
    print(f"Mean Sample WER:              {mean_wer*100:.1f}%")

    print("\n--- SAMPLE DECODED ZERO-LEAKAGE SENTENCES ---")
    for idx, (ref, hyp, wer, full_s) in enumerate(sample_outputs):
        status = "MATCH" if ref == hyp else ("PARTIAL" if wer < 1.0 else "MISSED")
        print(f"[{idx+1}] [{status}] WER: {wer*100:.1f}%")
        print(f"    Full Video Sentence: \"{full_s}\"")
        print(f"    Ground Truth:        {' '.join(ref)}")
        print(f"    Model Predicted:     {' '.join(hyp)}")

    print("\n" + "=" * 72)
    if pct_blank >= 99.0 or len(burst_lengths) == 0:
        print("[VERDICT: COLLAPSED] Model collapsed into blanks.")
    elif mean_wer <= 0.40:
        print("[VERDICT: EXCELLENT HIGH ACCURACY] Continuous model achieves low WER on unseen videos!")
    elif mean_wer <= 0.65:
        print("[VERDICT: CONVERGED GENERALIZATION] Escaped local minimum with strong word predictions.")
    elif mean_wer <= 0.80:
        print("[VERDICT: CONVERGING / LOCAL MINIMUM] Model predicts real signs, but has pronoun bias.")
        print("Execute train_real_sentences_phase2.py with CosineWarmRestarts to break lower.")
    else:
        print("[VERDICT: EARLY STAGE] Model requires additional training epochs.")
    print("=" * 72)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--split", type=str, default="val", choices=["val", "train", "all"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min_dur", type=int, default=2)
    args = parser.parse_args()
    audit_phase2(ckpt_path=args.ckpt, n_samples=args.samples, split=args.split, seed=args.seed, min_dur=args.min_dur)
