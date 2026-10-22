"""
Real Sentence Video Feature Extractor
=====================================
Extracts real human continuous sentence videos into canonical 163-dim MediaPipe tensors:
1. Scans ISL-CSLRT and iSign sentence datasets.
2. Relabels each sentence so the model is ONLY evaluated on in-scope words
   (un-trained words in the middle are mapped to blank/unpenalized transitions).
3. Extracts MediaPipe video features into PyTorch .pt tensors.
4. Generates master metadata index: Backend/data/tensors_sentence_level/index.json
"""

import os, sys, glob, json, argparse, hashlib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path: sys.path.insert(0, BACKEND_DIR)
SCRIPTS_DIR = os.path.join(BACKEND_DIR, "scripts")
if SCRIPTS_DIR not in sys.path: sys.path.insert(0, SCRIPTS_DIR)

from extract_video import extract_video_features
from plotter import match_sentence_tokens

CLASSES_PATH = os.path.join(BACKEND_DIR, "models", "word_vocabulary", "classes.json")
DEFAULT_OUT_DIR = os.path.join(BACKEND_DIR, "data", "tensors_sentence_level")

ISL_CSLRT_CSV = os.path.join(BACKEND_DIR, "data", "ISL_CSLRT_Corpus", "corpus_csv_files", "ISL Corpus sign glosses.csv")
ISL_CSLRT_VIDS = os.path.join(BACKEND_DIR, "data", "ISL_CSLRT_Corpus", "Videos_Sentence_Level")
ISIGN_ROOT = r"F:\dataset\isign_extracted"
ISIGN_CSV = os.path.join(ISIGN_ROOT, "word-presence-dataset_v1.1.csv")
ISIGN_VIDS = os.path.join(ISIGN_ROOT, "iSign-videos_v1.1")

def get_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]

def _process_single_candidate(task_args):
    item, output_dir = task_args
    v_path = item["video_path"]
    v_name = os.path.splitext(os.path.basename(v_path))[0]
    out_pt = os.path.join(output_dir, f"{item['dataset'].lower()}_{v_name}.pt")

    if os.path.exists(out_pt):
        try:
            t = torch.load(out_pt, weights_only=True)
            T = t.shape[0] if isinstance(t, torch.Tensor) else 0
            res = dict(item)
            res["tensor_path"] = out_pt
            res["frames"] = T
            return res
        except Exception:
            pass

    try:
        tensor = extract_video_features(v_path)
        if tensor is not None and tensor.shape[0] >= 8:
            torch.save(tensor, out_pt)
            res = dict(item)
            res["tensor_path"] = out_pt
            res["frames"] = tensor.shape[0]
            return res
    except Exception:
        pass
    return None

def run_extraction(output_dir=DEFAULT_OUT_DIR, max_sentences=None, num_workers=6):
    os.makedirs(output_dir, exist_ok=True)

    with open(CLASSES_PATH, "r") as f:
        classes = json.load(f)
    vocab_set = set(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    compound_classes = sorted([c for c in classes if " " in c], key=len, reverse=True)

    print("=" * 72)
    print("REAL CONTINUOUS SENTENCE EXTRACTION PIPELINE")
    print(f"Vocab: {len(classes)} classes | Output: {output_dir}")
    print("=" * 72)

    candidates = []

    # 1. Collect ISL-CSLRT Sentences
    if os.path.exists(ISL_CSLRT_CSV) and os.path.exists(ISL_CSLRT_VIDS):
        df_c = pd.read_csv(ISL_CSLRT_CSV)
        print(f"[ISL-CSLRT] Checking {len(df_c)} sentences...")
        for _, row in df_c.iterrows():
            s_name = str(row["Sentence"]).strip()
            gloss = str(row["SIGN GLOSSES"]).strip() if pd.notna(row["SIGN GLOSSES"]) else s_name
            in_scope, out_scope = match_sentence_tokens(gloss, vocab_set, compound_classes)
            if not in_scope:
                continue  # Skip sentences with 0 in-scope words

            s_dir = os.path.join(ISL_CSLRT_VIDS, s_name)
            if not os.path.exists(s_dir): continue
            vids = glob.glob(os.path.join(s_dir, "*.mp4")) + glob.glob(os.path.join(s_dir, "*.avi"))
            for v in vids:
                candidates.append({
                    "dataset": "ISL-CSLRT",
                    "video_path": v,
                    "sentence": s_name,
                    "in_scope_words": in_scope,
                    "in_scope_indices": [class_to_idx[w] for w in in_scope],
                    "out_scope_words": out_scope
                })

    # 2. Collect iSign Sentences
    if os.path.exists(ISIGN_CSV) and os.path.exists(ISIGN_VIDS):
        print(f"[iSign] Checking sentences from {os.path.basename(ISIGN_CSV)}...")
        df_i = pd.read_csv(ISIGN_CSV, usecols=["sentence_id", "sentence"]).drop_duplicates()
        for _, row in df_i.iterrows():
            sid = str(row["sentence_id"]).strip()
            s_text = str(row["sentence"]).strip()
            in_scope, out_scope = match_sentence_tokens(s_text, vocab_set, compound_classes)
            if not in_scope:
                continue

            v_path = os.path.join(ISIGN_VIDS, f"{sid}.mp4")
            if os.path.exists(v_path):
                candidates.append({
                    "dataset": "iSign",
                    "video_path": v_path,
                    "sentence": s_text,
                    "in_scope_words": in_scope,
                    "in_scope_indices": [class_to_idx[w] for w in in_scope],
                    "out_scope_words": out_scope
                })

    print(f"\nTotal Candidate Real Sentence Videos with In-Scope Words: {len(candidates):,}")
    if max_sentences:
        candidates = candidates[:max_sentences]
        print(f"Limiting to first {max_sentences} sentences as requested.")

    # Process & Extract MediaPipe Tensors
    index_entries = []
    tasks = [(item, output_dir) for item in candidates]

    from concurrent.futures import ProcessPoolExecutor, as_completed

    if num_workers > 1:
        print(f"Running multi-core extraction using {num_workers} parallel workers...")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_process_single_candidate, t) for t in tasks]
            for f in tqdm(as_completed(futures), total=len(futures), desc="Extracting (Multi-Core)"):
                res = f.result()
                if res is not None:
                    index_entries.append(res)
    else:
        for t in tqdm(tasks, desc="Extracting MediaPipe Features"):
            res = _process_single_candidate(t)
            if res is not None:
                index_entries.append(res)

    # Save Master Index
    index_file = os.path.join(output_dir, "index.json")
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(index_entries, f, indent=2)

    print("\n" + "=" * 72)
    print(f"EXTRACTION COMPLETE | Successfully processed: {len(index_entries)}/{len(candidates)} videos")
    print(f"Index Saved: {index_file}")
    print("=" * 72)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Real Human Continuous Sentence Videos")
    parser.add_argument("--output", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max", type=int, default=None, help="Cap maximum videos to extract")
    parser.add_argument("--workers", type=int, default=6, help="Number of parallel worker processes (default: 6)")
    args = parser.parse_args()
    run_extraction(output_dir=args.output, max_sentences=args.max, num_workers=args.workers)
