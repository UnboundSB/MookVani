"""
Sentence Dataset Vocabulary Coverage & Distribution Plotter
===========================================================
Pipeline analyzing:
1. iSign Continuous Dataset ("F:\dataset\isign_extracted")
2. ISL-CSLRT Nested ("D:\MookVani\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Videos_Sentence_Level")
3. ISL-CSLRT Backend Data ("D:\MookVani\Backend\data\ISL_CSLRT_Corpus\Videos_Sentence_Level")

Features:
- Strips non-ISL English function/stop words.
- Identifies 100% full coverage, 1-word out-of-scope, and 2-words out-of-scope sentences.
- Visualizes:
  * Y-axis: Number of words in sentence.
  * X-axis: Number of available sentences in dataset.
  * Bar breakdown: Full Match, 1-Word Missing, 2-Words Missing.
"""

import os, sys, re, glob, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
    except: pass

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES_PATH = os.path.join(BACKEND_DIR, "models", "word_vocabulary", "classes.json")
EDA_PLOTS_DIR = os.path.join(BACKEND_DIR, "eda", "plots")
MODEL_PLOTS_DIR = os.path.join(BACKEND_DIR, "models", "isl_conformer_ctc", "plots")

ISL_CSLRT_CSV = os.path.join(BACKEND_DIR, "data", "ISL_CSLRT_Corpus", "corpus_csv_files", "ISL Corpus sign glosses.csv")
DIR_CSLRT_NESTED = r"D:\MookVani\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Videos_Sentence_Level"
DIR_CSLRT_DATA   = os.path.join(BACKEND_DIR, "data", "ISL_CSLRT_Corpus", "Videos_Sentence_Level")
ISIGN_ROOT       = r"F:\dataset\isign_extracted"
ISIGN_CSV        = os.path.join(ISIGN_ROOT, "word-presence-dataset_v1.1.csv")
ISIGN_VIDS       = os.path.join(ISIGN_ROOT, "iSign-videos_v1.1")

NON_ISL_STOPWORDS = {
    'a', 'an', 'the', 'is', 'are', 'am', 'was', 'were', 'be', 'been', 'being',
    'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'about', 'into', 'through',
    'and', 'or', 'so', 'as', 'than', 'that', 'this', 'these', 'those', 'it', 'its',
    'do', 'does', 'did', 'have', 'has', 'had', 'will', 'would', 'shall', 'should',
    'can', 'could', 'may', 'might', 'must'
}

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())

def match_sentence_tokens(text, vocab_set, compound_classes):
    """Tokenise a sentence gloss into (in_scope, out_of_scope) words.

    The in-scope words are returned in SIGNING ORDER.  Multi-word glosses
    ("thank you", "how are you", ...) are matched IN PLACE, so "i am fine. thank you
    sir" yields ['i', 'thank you'] and not ['thank you', 'i'].

    The previous implementation appended every compound phrase first and the single
    words afterwards, so the target sequence was out of order for every sentence that
    contains a multi-word gloss - 18 of the 1579 extracted clips, which for a CTC model
    is a wrong alignment target and not just a wrong label.
    """
    if not isinstance(text, str):
        return [], []
    norm_text = text.lower()
    norm_text = re.sub(r"[^\w\s]", " ", norm_text)   # keep the character positions intact

    # 1. Locate the compound phrases: one occurrence per class, longest match wins.
    spans = []
    for comp in compound_classes:
        m = re.search(r'\b' + re.escape(comp) + r'\b', norm_text)
        if m:
            spans.append((m.start(), m.end(), comp))
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    chosen, last_end = [], -1
    for s, e, c in spans:
        if s >= last_end:
            chosen.append((s, e, c))
            last_end = e

    # 2. The single words that are not part of a compound phrase.
    covered = [False] * len(norm_text)
    for s, e, _ in chosen:
        for i in range(s, e):
            covered[i] = True
    words = []
    for m in re.finditer(r'\S+', norm_text):
        if covered[m.start()]:
            continue
        w = m.group(0)
        if w in NON_ISL_STOPWORDS:
            continue
        words.append((m.start(), w))

    # 3. Merge by position -> the true signing order.
    tokens = [(s, c) for s, e, c in chosen] + words
    tokens.sort(key=lambda x: x[0])

    in_scope = [t for _, t in tokens if t in vocab_set]
    out_of_scope = [t for _, t in tokens if t not in vocab_set]
    return in_scope, out_of_scope

def run_analysis(output_dir=None):
    if output_dir is None: output_dir = EDA_PLOTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(MODEL_PLOTS_DIR, exist_ok=True)

    with open(CLASSES_PATH, "r") as f: vocab = json.load(f)
    vocab_set = set(vocab)
    compound_classes = sorted([c for c in vocab if " " in c], key=len, reverse=True)

    print("=" * 72, flush=True)
    print(f"SENTENCE DATASET VOCABULARY COVERAGE PIPELINE", flush=True)
    print(f"Trained Sign Vocabulary: {len(vocab_set)} classes", flush=True)
    print(f"Output Plot Directory:   {output_dir}", flush=True)
    print("=" * 72, flush=True)

    records = []
    out_scope_counter = Counter()

    # --- 1. Process ISL-CSLRT Corpus ---
    cslrt_vids_dir = DIR_CSLRT_DATA if os.path.exists(DIR_CSLRT_DATA) else DIR_CSLRT_NESTED
    if os.path.exists(ISL_CSLRT_CSV):
        df_cslrt = pd.read_csv(ISL_CSLRT_CSV)
        print(f"[ISL-CSLRT] Analyzing {len(df_cslrt)} annotated sentences from {os.path.basename(ISL_CSLRT_CSV)}...", flush=True)
        for _, row in df_cslrt.iterrows():
            sent_name = str(row["Sentence"]).strip()
            gloss_str = str(row["SIGN GLOSSES"]).strip() if pd.notna(row["SIGN GLOSSES"]) else sent_name
            
            in_scope, out_scope = match_sentence_tokens(gloss_str, vocab_set, compound_classes)
            total_words = len(in_scope) + len(out_scope)
            if total_words == 0: continue

            for o in out_scope: out_scope_counter[o] += 1

            cov_type = "Full Match (100%)" if len(out_scope) == 0 else (
                "1 Word Missing" if len(out_scope) == 1 else (
                    "2 Words Missing" if len(out_scope) == 2 else "3+ Words Missing"
                )
            )

            records.append({
                "dataset": "ISL-CSLRT",
                "sentence": sent_name,
                "total_words": total_words,
                "in_scope_words": len(in_scope),
                "out_scope_words": len(out_scope),
                "in_scope_list": ", ".join(in_scope),
                "out_scope_list": ", ".join(out_scope),
                "coverage_type": cov_type
            })

    # --- 2. Process iSign Dataset ---
    if os.path.exists(ISIGN_CSV):
        print(f"[iSign] Reading unique sentences from {ISIGN_CSV}...", flush=True)
        df_isign = pd.read_csv(ISIGN_CSV, usecols=["sentence_id", "sentence"])
        sentence_counts = df_isign.groupby("sentence")["sentence_id"].count()
        print(f"[iSign] Found {len(sentence_counts):,} unique sentences.", flush=True)

        for s_text, n_vids in sentence_counts.items():
            s_text = str(s_text).strip()
            in_scope, out_scope = match_sentence_tokens(s_text, vocab_set, compound_classes)
            total_words = len(in_scope) + len(out_scope)
            if total_words == 0: continue

            for o in out_scope: out_scope_counter[o] += 1

            cov_type = "Full Match (100%)" if len(out_scope) == 0 else (
                "1 Word Missing" if len(out_scope) == 1 else (
                    "2 Words Missing" if len(out_scope) == 2 else "3+ Words Missing"
                )
            )

            records.append({
                "dataset": "iSign",
                "sentence": s_text,
                "total_words": total_words,
                "in_scope_words": len(in_scope),
                "out_scope_words": len(out_scope),
                "in_scope_list": ", ".join(in_scope),
                "out_scope_list": ", ".join(out_scope),
                "coverage_type": cov_type
            })

    df = pd.DataFrame(records)
    print(f"\nTotal Analyzed Sentences across Datasets: {len(df):,}", flush=True)

    print("\n--- COVERAGE BREAKDOWN ---", flush=True)
    cov_counts = df["coverage_type"].value_counts()
    for k, v in cov_counts.items():
        print(f"  * {k:<20}: {v:,} sentences ({v/len(df)*100:.1f}%)", flush=True)

    # Save CSV summary
    csv_out = os.path.join(output_dir, "sentence_coverage_summary.csv")
    df.to_csv(csv_out, index=False)
    print(f"[SAVED] CSV Summary: {csv_out}", flush=True)

    # =========================================================================
    # PLOTTING
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Restrict to sentences having <= 8 words for clean visual grouping
    df_plot = df[df["total_words"] <= 8].copy()
    pivot_df = pd.crosstab(df_plot["total_words"], df_plot["coverage_type"])

    cat_order = ["Full Match (100%)", "1 Word Missing", "2 Words Missing", "3+ Words Missing"]
    colors    = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]
    
    available_cols = [c for c in cat_order if c in pivot_df.columns]
    plot_data = pivot_df[available_cols]

    # Horizontal Bar Chart: Y = Number of Words, X = Number of Sentences
    plot_data.plot(kind="barh", stacked=True, ax=ax1, color=[colors[cat_order.index(c)] for c in available_cols], width=0.75)
    ax1.set_title("Sentences by Word Count (Y-Axis) vs Dataset Count (X-Axis)", fontsize=13, weight="bold")
    ax1.set_xlabel("Number of Sentences in Datasets", fontsize=11)
    ax1.set_ylabel("Number of Words in Sentence (Y)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5, axis="x")
    ax1.legend(title="Coverage Type", frameon=True)

    # 2. Top Out-of-Scope Words
    top_out = out_scope_counter.most_common(12)
    words, counts = zip(*top_out) if top_out else ([], [])
    ax2.barh(range(len(words)), counts, color="#9467bd", height=0.65)
    ax2.set_yticks(range(len(words)))
    ax2.set_yticklabels(words, fontsize=10)
    ax2.invert_yaxis()
    ax2.set_title("Top 12 Out-of-Scope Words\n(Adding these unlocks maximum sentences)", fontsize=13, weight="bold")
    ax2.set_xlabel("Occurrences across Dataset Sentences", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="x")

    plt.tight_layout()
    plot_out1 = os.path.join(output_dir, "sentence_coverage_distribution.png")
    plot_out2 = os.path.join(MODEL_PLOTS_DIR, "sentence_coverage_distribution.png")
    plt.savefig(plot_out1, dpi=200)
    plt.savefig(plot_out2, dpi=200)
    plt.close()

    print(f"[SAVED] Plot: {plot_out1}", flush=True)
    print(f"[SAVED] Plot copy: {plot_out2}", flush=True)
    print("=" * 72, flush=True)

if __name__ == "__main__":
    run_analysis()
