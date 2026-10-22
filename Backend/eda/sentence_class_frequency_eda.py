"""
Sentence Dataset In-Scope Class Frequency EDA (All 262 Classes)
==============================================================
Analyzes the distribution and frequency of ALL 262 trained vocabulary classes
appearing in the continuous sentence dataset (tensors_sentence_level/index.json).

Generates a publication-grade, high-resolution multi-panel plot showing every
single word from the 262 classes (from most frequent down to 0-count words).
"""

import os
import sys
import json
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(BACKEND_DIR, "data", "tensors_sentence_level", "index.json")
CLASSES_PATH = os.path.join(BACKEND_DIR, "models", "word_vocabulary", "classes.json")
PLOTS_DIR = os.path.join(BACKEND_DIR, "eda", "plots")
OUT_JSON = os.path.join(BACKEND_DIR, "eda", "sentence_class_frequencies.json")

def analyze_all_262_class_frequencies():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    if not os.path.exists(CLASSES_PATH):
        raise FileNotFoundError(f"Classes file not found at {CLASSES_PATH}")
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"Sentence dataset index not found at {INDEX_PATH}")

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes_data = json.load(f)
        if isinstance(classes_data, dict) and "classes" in classes_data:
            classes = classes_data["classes"]
        else:
            classes = classes_data

    num_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        index_items = json.load(f)

    total_sentences = len(index_items)
    class_counts = {c: 0 for c in classes}

    # Count word occurrences across all sentences
    for item in index_items:
        # In-scope words specifically matched to vocabulary
        in_scope = item.get("in_scope_words", [])
        for w in in_scope:
            w_clean = w.strip().lower()
            if w_clean in class_counts:
                class_counts[w_clean] += 1

    # Sort all 262 classes by frequency descending
    sorted_classes = sorted(class_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)

    present_classes = [(w, c) for w, c in sorted_classes if c > 0]
    zero_classes = [(w, c) for w, c in sorted_classes if c == 0]

    total_tokens = sum(c for _, c in sorted_classes)

    summary = {
        "total_classes": num_classes,
        "total_sentences": total_sentences,
        "active_classes_count": len(present_classes),
        "zero_count_classes_count": len(zero_classes),
        "total_word_occurrences": total_tokens,
        "all_262_classes_ranked": [{"rank": r+1, "word": w, "count": c} for r, (w, c) in enumerate(sorted_classes)],
        "zero_count_classes": [w for w, _ in zero_classes]
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # -------------------------------------------------------------------------
    # High-Resolution Publication Plot: All 262 Classes in 4 Clear Panels
    # -------------------------------------------------------------------------
    # 262 classes split across 4 panels: ~66 classes per panel
    n_panels = 4
    chunk_size = int(np.ceil(num_classes / n_panels))  # 66 per column

    fig, axes = plt.subplots(1, n_panels, figsize=(28, 22), dpi=200)
    fig.patch.set_facecolor("#0f172a")  # Dark sleek theme

    for p in range(n_panels):
        ax = axes[p]
        ax.set_facecolor("#1e293b")
        chunk = sorted_classes[p * chunk_size : (p + 1) * chunk_size]
        if not chunk:
            continue

        words = [x[0] for x in chunk]
        counts = [x[1] for x in chunk]
        y_pos = np.arange(len(words))

        # Color mapping: blue for top tier, cyan for mid, amber for low, red/muted for 0
        colors = []
        for c in counts:
            if c >= 50:
                colors.append("#38bdf8")  # Vibrant sky blue
            elif c >= 10:
                colors.append("#34d399")  # Emerald green
            elif c >= 2:
                colors.append("#fbbf24")  # Warm amber
            elif c == 1:
                colors.append("#f87171")  # Light coral
            else:
                colors.append("#475569")  # Slate gray for 0 count

        bars = ax.barh(y_pos, counts, height=0.72, color=colors, edgecolor="none")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words, fontsize=8.5, color="#f1f5f9", fontweight="medium")
        ax.invert_yaxis()  # Top-down order

        max_c = max(counts) if counts else 1
        ax.set_xlim(0, max(max_c * 1.25, 4))

        # Add numeric labels to each bar
        for i, (b, c) in enumerate(zip(bars, counts)):
            txt_color = "#38bdf8" if c >= 50 else ("#34d399" if c >= 10 else ("#fbbf24" if c >= 2 else ("#f87171" if c == 1 else "#94a3b8")))
            ax.text(c + (max_c * 0.02 + 0.1), i, f"{c}", va="center", ha="left",
                    fontsize=8, color=txt_color, fontweight="bold")

        start_r = p * chunk_size + 1
        end_r = min((p + 1) * chunk_size, num_classes)
        ax.set_title(f"Rank {start_r} to {end_r}", fontsize=12, color="#38bdf8", fontweight="bold", pad=10)
        ax.tick_params(colors="#94a3b8")
        ax.grid(axis="x", linestyle="--", alpha=0.15, color="#ffffff")

        for spine in ax.spines.values():
            spine.set_color("#334155")

    fig.suptitle(
        f"ISL Sentence Dataset: Complete Frequency Distribution Across All {num_classes} Vocabulary Classes\n"
        f"Total Sentences: {total_sentences:,} | Total Word Instances: {total_tokens:,} | "
        f"Active Words: {len(present_classes)} ({len(present_classes)/num_classes*100:.1f}%) | "
        f"Unseen Words (Count=0): {len(zero_classes)} ({len(zero_classes)/num_classes*100:.1f}%)",
        fontsize=16, color="#ffffff", fontweight="bold", y=0.985
    )

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])
    plot_file = os.path.join(PLOTS_DIR, "sentence_all_262_class_frequencies.png")
    plt.savefig(plot_file, dpi=200, facecolor=fig.get_facecolor())
    plt.close()

    # -------------------------------------------------------------------------
    # Second High-Resolution Plot: ALL 181 ACTIVE OCCURRING WORDS (Count >= 1)
    # -------------------------------------------------------------------------
    # Focused purely on the words that actually appeared in the sentences
    n_active = len(present_classes)
    n_cols = 3
    active_chunk = int(np.ceil(n_active / n_cols))

    fig2, axes2 = plt.subplots(1, n_cols, figsize=(24, 20), dpi=200)
    fig2.patch.set_facecolor("#0b0f19")

    for col_idx in range(n_cols):
        ax = axes2[col_idx]
        ax.set_facecolor("#111827")
        chunk = present_classes[col_idx * active_chunk : (col_idx + 1) * active_chunk]
        if not chunk:
            continue

        words = [x[0] for x in chunk]
        counts = [x[1] for x in chunk]
        y_pos = np.arange(len(words))

        # Dynamic gradient palette
        colors = []
        for c in counts:
            if c >= 50:
                colors.append("#38bdf8")
            elif c >= 15:
                colors.append("#34d399")
            elif c >= 5:
                colors.append("#a78bfa")
            elif c >= 2:
                colors.append("#facc15")
            else:
                colors.append("#fb923c")

        bars = ax.barh(y_pos, counts, height=0.74, color=colors, edgecolor="none")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words, fontsize=9, color="#f9fafb", fontweight="medium")
        ax.invert_yaxis()

        max_c = max(counts) if counts else 1
        ax.set_xlim(0, max(max_c * 1.25, 5))

        for i, (b, c) in enumerate(zip(bars, counts)):
            txt_color = "#38bdf8" if c >= 50 else ("#34d399" if c >= 15 else ("#a78bfa" if c >= 5 else ("#facc15" if c >= 2 else "#fb923c")))
            ax.text(c + (max_c * 0.02 + 0.1), i, f"{c}", va="center", ha="left",
                    fontsize=8.5, color=txt_color, fontweight="bold")

        start_r = col_idx * active_chunk + 1
        end_r = min((col_idx + 1) * active_chunk, n_active)
        ax.set_title(f"Rank {start_r} to {end_r}", fontsize=13, color="#60a5fa", fontweight="bold", pad=12)
        ax.tick_params(colors="#9ca3af")
        ax.grid(axis="x", linestyle="--", alpha=0.18, color="#ffffff")

        for spine in ax.spines.values():
            spine.set_color("#374151")

    fig2.suptitle(
        f"ISL Sentence Dataset: All {n_active} Words That Occurred In Sentences (Rank 1 to {n_active})\n"
        f"Total Sentences Analyzed: {total_sentences:,} | Total Word Instances: {total_tokens:,}",
        fontsize=16, color="#ffffff", fontweight="bold", y=0.985
    )

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])
    plot_active_file = os.path.join(PLOTS_DIR, "sentence_active_words_frequency.png")
    plt.savefig(plot_active_file, dpi=200, facecolor=fig2.get_facecolor())
    plt.close()

    print("=" * 72)
    print("ALL CLASSES FREQUENCY ANALYSIS COMPLETE")
    print("=" * 72)
    print(f"Total Model Classes:        {num_classes}")
    print(f"Total Sentences:            {total_sentences:,}")
    print(f"Active Words in Sentences:  {len(present_classes)} classes")
    print(f"Unseen Classes (Count=0):   {len(zero_classes)} classes")
    print(f"Saved All 262 Plot to:      {plot_file}")
    print(f"Saved Active Words Plot to: {plot_active_file}")
    print(f"Saved Full JSON Report to:  {OUT_JSON}")
    print("=" * 72)

if __name__ == "__main__":
    analyze_all_262_class_frequencies()
