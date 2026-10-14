"""
Analyse frame counts (and derived duration) of pose-sequence tensors
stored in Backend/data/tensors_include50_word_level.

Each .pt file is a tensor of shape (T, F) where T = number of frames.
Assumes source videos were recorded at 25 fps (ISL-CSLRT standard).
Saves distribution plots to Backend/eda/plots/.
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data", "tensors_include50_word_level")
PLOTS_DIR = os.path.join(BASE_DIR, "eda", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

FPS = 25.0          # ISL-CSLRT source fps

# ── Scan ─────────────────────────────────────────────────────────────────────
print(f"[INFO] Scanning  {DATA_DIR} …")
by_class   = defaultdict(list)   # class -> list of frame counts
all_frames = []

total_files = 0
errors      = 0
for cls in sorted(os.listdir(DATA_DIR)):
    cls_path = os.path.join(DATA_DIR, cls)
    if not os.path.isdir(cls_path):
        continue
    for fn in os.listdir(cls_path):
        if not fn.endswith(".pt"):
            continue
        fpath = os.path.join(cls_path, fn)
        try:
            t = torch.load(fpath, map_location="cpu", weights_only=True)
            frames = t.shape[0]
            by_class[cls].append(frames)
            all_frames.append(frames)
            total_files += 1
        except Exception as e:
            errors += 1
            print(f"  [WARN] {fpath}: {e}")

print(f"[INFO] Loaded {total_files} tensors across {len(by_class)} classes  ({errors} errors)")

arr = np.array(all_frames)
dur = arr / FPS

# ── Console stats ─────────────────────────────────────────────────────────────
def blk(label, a):
    print(f"\n{label}")
    print(f"  Min    = {a.min():.2f}")
    print(f"  Max    = {a.max():.2f}")
    print(f"  Median = {np.median(a):.2f}")
    print(f"  Mean   = {a.mean():.2f}")
    print(f"  Std    = {a.std():.2f}")
    print(f"  P25    = {np.percentile(a,25):.2f}")
    print(f"  P75    = {np.percentile(a,75):.2f}")
    print(f"  P95    = {np.percentile(a,95):.2f}")

blk("Frame Count  (T per tensor)", arr)
blk("Duration (s) @ 25 fps",       dur)

med_f = np.median(arr)
med_d = med_f / FPS
print(f"\n[KEY] Median = {med_f:.0f} frames  /  {med_d:.2f} s")

# ── Report txt ────────────────────────────────────────────────────────────────
report_lines = [
    "tensors_include50_word_level — Frame / Duration Analysis",
    f"Tensors : {total_files}   Classes : {len(by_class)}   Source FPS : {FPS}",
    "",
    f"Frame Count:  min={arr.min():.0f}  median={med_f:.0f}  mean={arr.mean():.1f}  max={arr.max():.0f}  std={arr.std():.1f}",
    f"Duration (s): min={dur.min():.2f}  median={med_d:.2f}  mean={dur.mean():.2f}  max={dur.max():.2f}  std={dur.std():.2f}",
    f"P25/P75 frames: {np.percentile(arr,25):.0f} / {np.percentile(arr,75):.0f}",
    f"P95 frames: {np.percentile(arr,95):.0f}",
]
report_path = os.path.join(PLOTS_DIR, "tensors_include50_duration_report.txt")
with open(report_path, "w") as f:
    f.write("\n".join(report_lines))
print(f"[SAVED] {report_path}")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — global distributions (histogram + CDF side by side)
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0f1117")
for ax in axes:
    ax.set_facecolor("#1a1d27")
    ax.tick_params(colors="#c9d1d9")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

BLUE  = "#58a6ff"
RED   = "#f85149"
GREEN = "#3fb950"

# — Histogram of durations ——
ax = axes[0]
n_bins = 40
ax.hist(dur, bins=n_bins, color=BLUE, edgecolor="#1a1d27", alpha=0.88, linewidth=0.4)
ax.axvline(med_d,       color=RED,   lw=2.2, linestyle="--", label=f"Median  {med_d:.2f} s")
ax.axvline(dur.mean(),  color=GREEN, lw=1.8, linestyle=":",  label=f"Mean  {dur.mean():.2f} s")
ax.axvline(np.percentile(dur,95), color="#d29922", lw=1.5, linestyle="-.",
           label=f"P95  {np.percentile(dur,95):.2f} s")
ax.set_title("Duration Distribution (seconds)", color="#e6edf3", fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Duration (s)", color="#8b949e", fontsize=11)
ax.set_ylabel("Tensor Count", color="#8b949e", fontsize=11)
ax.legend(fontsize=9, facecolor="#21262d", labelcolor="#c9d1d9", edgecolor="#30363d")
ax.grid(True, linestyle="--", alpha=0.2, color="#8b949e")

# — CDF of frame counts ——
ax = axes[1]
sorted_f = np.sort(arr)
cdf = np.arange(1, len(sorted_f)+1) / len(sorted_f)
ax.plot(sorted_f, cdf, color=BLUE, lw=2)
ax.axvline(med_f, color=RED,   lw=2, linestyle="--", label=f"Median  {med_f:.0f} frames")
ax.axhline(0.5,   color=RED,   lw=1, linestyle="--", alpha=0.4)
ax.axvline(np.percentile(arr,95), color="#d29922", lw=1.5, linestyle="-.",
           label=f"P95  {np.percentile(arr,95):.0f} frames")
ax.set_title("CDF — Frame Count", color="#e6edf3", fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Frame Count (T)", color="#8b949e", fontsize=11)
ax.set_ylabel("Cumulative Fraction", color="#8b949e", fontsize=11)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
ax.legend(fontsize=9, facecolor="#21262d", labelcolor="#c9d1d9", edgecolor="#30363d")
ax.grid(True, linestyle="--", alpha=0.2, color="#8b949e")

plt.suptitle(
    f"tensors_include50_word_level  ·  {total_files} videos  ·  {len(by_class)} classes  ·  25 fps",
    color="#e6edf3", fontsize=14, fontweight="bold", y=1.02
)
plt.tight_layout()
p1 = os.path.join(PLOTS_DIR, "tensors_include50_duration_dist.png")
plt.savefig(p1, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"[SAVED] {p1}")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — per-class strip plot (sorted by median)
# ═════════════════════════════════════════════════════════════════════════════
cls_order   = sorted(by_class, key=lambda c: np.median(by_class[c]))
cls_medians = [np.median(by_class[c]) for c in cls_order]
cls_labels  = [c[:22] + "…" if len(c) > 22 else c for c in cls_order]
n_cls       = len(cls_order)

fig_h = max(8, n_cls * 0.18)
fig, ax = plt.subplots(figsize=(26, fig_h))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#1a1d27")
ax.tick_params(colors="#c9d1d9", labelsize=6.5)
for spine in ax.spines.values():
    spine.set_edgecolor("#30363d")

for i, cls in enumerate(cls_order):
    vals = np.array(by_class[cls])
    ax.scatter([i]*len(vals), vals, s=14, color=BLUE, alpha=0.55, zorder=3)
    ax.plot([i], [np.median(vals)], "D", color=RED, markersize=4.5, zorder=5)

ax.axhline(med_f,        color=RED,   lw=1.8, linestyle="--",
           label=f"Global Median  {med_f:.0f} f  ({med_d:.2f} s)")
ax.axhline(arr.mean(),   color=GREEN, lw=1.4, linestyle=":",
           label=f"Global Mean  {arr.mean():.1f} f")
ax.axhline(np.percentile(arr,95), color="#d29922", lw=1.2, linestyle="-.",
           label=f"P95  {np.percentile(arr,95):.0f} f")

ax.set_xticks(range(n_cls))
ax.set_xticklabels(cls_labels, rotation=90, fontsize=6.5, color="#c9d1d9")
ax.set_ylabel("Frame Count (T)", color="#8b949e", fontsize=11)
ax.set_title(
    "Per-Class Frame Count  (sorted by median · red ◆ = class median)",
    color="#e6edf3", fontsize=13, fontweight="bold", pad=10
)
ax.legend(fontsize=9, facecolor="#21262d", labelcolor="#c9d1d9", edgecolor="#30363d")
ax.grid(True, axis="y", linestyle="--", alpha=0.18, color="#8b949e")

plt.tight_layout()
p2 = os.path.join(PLOTS_DIR, "tensors_include50_per_class_frames.png")
plt.savefig(p2, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"[SAVED] {p2}")

print("\n[SUCCESS] Done.")
