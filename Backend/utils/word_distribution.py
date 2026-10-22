"""
Word-frequency diagnostics for the sentence-level training mix.
==============================================================

The continuous corpus is extremely skewed: in its training split the six
pronouns (``i``, ``you``, ``he``, ``she``, ``we``, ``they``) hold 64% of all
2 122 word instances, and 50 of the 228 vocabulary words never appear at all.
A CTC head trained on that distribution learns the LABEL PRIOR instead of the
signs: it emits ``i`` everywhere and never a content word.

The three training sources have very different priors:

    WORD      one isolated sign per clip, the class drawn UNIFORMLY  -> FLAT
    SYNTHETIC N signs per sentence, the classes drawn UNIFORMLY     -> FLAT
    REAL      the continuous corpus                                 -> SKEWED

so the mixture is already a de-skewing mechanism: with
``--mix_weights 0.40 0.35 0.25`` the effective prior of ``i`` falls from
20.8% (real only) to ~5.6%, and the effective support of the distribution
rises from ~8 classes to ~120.

This module measures that, so the decision about the frequency cap is made on
numbers instead of on intuition:

    effective_support = exp(entropy of the distribution)

which is the number of classes the model effectively sees (228 for a uniform
distribution, 8 for the raw continuous corpus).

The historical measurements of this repository are the reason the caps are OFF
by default:

    no cap                 75.0% WER
    --mix_word_cap 60      87.2% WER   (the cap DROPS ~40% of the real corpus)
    --class_balance_alpha  98.9% WER   (the prior is fixed in the LOSS, which
                                        destroys the CTC alignment)
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Counting
# --------------------------------------------------------------------------- #
def word_instance_counts(items: Iterable[dict], key: str = "in_scope_indices") -> Counter:
    """How often does every word occur in the targets of these items?"""
    counts: Counter = Counter()
    for it in items:
        for w in it.get(key, []) or []:
            counts[w] += 1
    return counts


def counts_to_distribution(counts: Counter, n_classes: int) -> np.ndarray:
    """The probability distribution of a counter over ``n_classes`` classes."""
    dist = np.zeros(n_classes, dtype=np.float64)
    for cls, c in counts.items():
        if 0 <= cls < n_classes:
            dist[cls] = c
    total = dist.sum()
    if total > 0:
        dist /= total
    return dist


def uniform_distribution(n_classes: int) -> np.ndarray:
    if n_classes <= 0:
        return np.zeros(0)
    return np.full(n_classes, 1.0 / n_classes)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def collapse_metrics(dist: np.ndarray) -> Dict[str, float]:
    """How peaked is the label prior?"""
    dist = np.asarray(dist, dtype=np.float64)
    n = len(dist)
    if n == 0 or dist.sum() <= 0:
        return {"n_classes": n, "top1_share": 0.0, "top6_share": 0.0,
                "max_min_ratio": 0.0, "effective_support": 0.0, "entropy": 0.0}

    nz = dist[dist > 0]
    entropy = float(-(nz * np.log(nz)).sum())
    top = np.sort(dist)[::-1]
    top1 = float(top[0])
    top6 = float(top[:6].sum())
    ratio = float(top[0] / max(dist[dist > 0].min(), 1e-12)) if (dist > 0).any() else 0.0
    return {
        "n_classes": int(n),
        "top1_share": top1,
        "top6_share": top6,
        "max_min_ratio": ratio,
        "effective_support": float(math.exp(entropy)),
        "entropy": entropy,
    }


def mixed_distribution(parts: Sequence[Tuple[float, np.ndarray]]) -> np.ndarray:
    """The distribution of a mixture of sources: ``[(weight, distribution), ...]``."""
    if not parts:
        return np.zeros(0)
    total_w = sum(w for w, _ in parts)
    if total_w <= 0:
        return np.zeros(len(parts[0][1]))
    out = np.zeros_like(parts[0][1], dtype=np.float64)
    for w, dist in parts:
        out += (w / total_w) * np.asarray(dist, dtype=np.float64)
    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def format_distribution_report(
    dist: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    top: int = 10,
    title: str = "",
) -> str:
    """A readable report of a label prior."""
    metrics = collapse_metrics(dist)
    lines = []
    if title:
        lines.append(f"  {title}")
    lines.append(f"    top-1 share        {metrics['top1_share']*100:6.2f}%")
    lines.append(f"    top-6 share        {metrics['top6_share']*100:6.2f}%")
    lines.append(f"    effective support  {metrics['effective_support']:8.1f} "
                 f"of {metrics['n_classes']} classes")
    lines.append(f"    max/min ratio      {metrics['max_min_ratio']:8.1f}x")
    if class_names is not None:
        order = np.argsort(dist)[::-1][:top]
        parts = []
        for idx in order:
            if dist[idx] <= 0:
                continue
            name = class_names[idx] if idx < len(class_names) else str(idx)
            parts.append(f"{name}={dist[idx]*100:.1f}%")
        if parts:
            lines.append("    most frequent:     " + ", ".join(parts))
    return "\n".join(lines)


def print_source_report(
    name: str,
    dist: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    top: int = 10,
) -> None:
    print(format_distribution_report(dist, class_names=class_names, top=top,
                                     title=f"[PRIOR] {name}"))
