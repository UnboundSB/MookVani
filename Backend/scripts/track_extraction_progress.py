"""
MediaPipe Sentence Extraction Progress Tracker
==============================================
Monitors progress percentage of continuous sentence videos converted
into 163-dim MediaPipe tensors in Backend/data/tensors_sentence_level/.
"""

import os, sys, json, glob
from datetime import datetime

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TENSORS_DIR = os.path.join(BACKEND_DIR, "data", "tensors_sentence_level")
ISL_CSLRT_VIDS = os.path.join(BACKEND_DIR, "data", "ISL_CSLRT_Corpus", "Videos_Sentence_Level")
ISIGN_ROOT = r"F:\dataset\isign_extracted"
ISIGN_CSV = os.path.join(ISIGN_ROOT, "word-presence-dataset_v1.1.csv")
CLASSES_PATH = os.path.join(BACKEND_DIR, "models", "phase6_word_model", "classes.json")

def get_total_candidates():
    # 1. ISL_CSLRT candidates
    isl_count = 0
    if os.path.exists(ISL_CSLRT_VIDS):
        for ext in ("*.mp4", "*.avi"):
            isl_count += len(glob.glob(os.path.join(ISL_CSLRT_VIDS, "**", ext), recursive=True))

    # 2. iSign candidates with in-scope vocabulary
    isign_count = 0
    if os.path.exists(CLASSES_PATH) and os.path.exists(ISIGN_CSV):
        try:
            with open(CLASSES_PATH, "r") as f:
                vocab = set(json.load(f))
            import pandas as pd
            df = pd.read_csv(ISIGN_CSV, usecols=["sentence_id", "sentence"]).drop_duplicates()
            for s in df["sentence"].astype(str):
                words = set(s.lower().replace(".", "").replace(",", "").split())
                if words & vocab:
                    isign_count += 1
        except Exception:
            isign_count = 1450  # approximate fallback from plotter analysis

    total = isl_count + isign_count
    return isl_count, isign_count, total

def track():
    if not os.path.exists(TENSORS_DIR):
        print(f"[!] Output directory not found: {TENSORS_DIR}")
        return

    pt_files = glob.glob(os.path.join(TENSORS_DIR, "*.pt"))
    extracted_total = len(pt_files)

    isl_done = sum(1 for f in pt_files if os.path.basename(f).startswith("isl-cslrt_"))
    isign_done = sum(1 for f in pt_files if os.path.basename(f).startswith("isign_"))

    isl_cand, isign_cand, total_cand = get_total_candidates()
    pct = (extracted_total / total_cand * 100) if total_cand > 0 else 0.0

    # Total size
    total_bytes = sum(os.path.getsize(f) for f in pt_files)
    mb = total_bytes / (1024 * 1024)

    # Latest file
    latest_str = "None"
    if pt_files:
        latest = max(pt_files, key=os.path.getmtime)
        mtime = datetime.fromtimestamp(os.path.getmtime(latest)).strftime("%Y-%m-%d %H:%M:%S")
        latest_str = f"{os.path.basename(latest)} ({mtime})"

    # Progress Bar
    bar_len = 35
    filled = int(round(bar_len * pct / 100))
    bar = "=" * filled + "-" * (bar_len - filled)

    print("\n" + "=" * 65)
    print("       MEDIAPIPE SENTENCE EXTRACTION PIPELINE TRACKER")
    print("=" * 65)
    print(f" Overall Progress: [{bar}] {pct:5.2f}% ({extracted_total:,}/{total_cand:,})")
    print("-" * 65)
    print(f"  • ISL_CSLRT Videos : {isl_done:,} / {isl_cand:,} done")
    print(f"  • iSign Videos     : {isign_done:,} / {isign_cand:,} done")
    print(f"  • Output Storage   : {mb:.1f} MB in tensors_sentence_level/")
    print(f"  • Most Recent File : {latest_str}")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    track()
