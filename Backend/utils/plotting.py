"""
Plotting utilities for every model family and every metric.
=============================================================

The project trains three different kinds of models (a word classifier, a CTC
sentence model and a gesture segmentor) and each of them has its own history
format.  Instead of letting every trainer draw its own figures, they all come
here, so a report always looks the same and always contains the same
information.

What it produces
----------------
* ``plot_training_curves``  - loss, accuracy, F1 / precision / recall, WER and
  the exact-match, plus the learning rate of every parameter group and the
  epoch time.  One figure per metric family, all in the model's own directory.
* ``plot_confusion_matrix`` - **the confusion matrix**, the one figure that
  actually tells you what the model has learned.  It can be row-normalised
  (the per-class recall, which is the honest view of a class imbalance), it can
  show only the N most confused classes, and it ships a TEXT report next to it
  (the per-class precision / recall / F1 and the worst confusions), because a
  picture of a 228x228 matrix is not readable.
* ``plot_per_class_metrics`` - the per-class bars (precision / recall / F1),
  sorted by the worst class.
* ``plot_label_distribution`` - the label prior of a corpus or of a mixture
  (the companion of ``utils/word_distribution.py``).

Every function returns the path(s) it wrote, and nothing is ever written
outside the directory it is given.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:      # pragma: no cover - seaborn is optional
    _HAS_SNS = False


# --------------------------------------------------------------------------- #
# History helpers (the two naming conventions of the project)
# --------------------------------------------------------------------------- #
#: The word-model phases use short keys, the sentence trainers use long ones.
_ALIASES = {
    "train_loss": ("tl", "train_loss"),
    "val_loss": ("vl", "val_loss"),
    "train_acc": ("ta", "train_acc"),
    "val_acc": ("va", "val_acc"),
    "train_f1": ("tf1", "train_f1"),
    "val_f1": ("vf1", "val_f1"),
    "val_precision": ("vp", "val_precision"),
    "val_recall": ("vr", "val_recall"),
    "val_wer": ("wer", "val_wer"),
    "val_exact": ("exact", "val_exact"),
    "val_wer_ema": ("wer_ema",),
    "val_loss_ema": ("val_loss_ema",),
}


def get_series(history: Dict, key: str) -> Optional[List[float]]:
    """Fetch a metric from a history dict, whichever naming convention it uses."""
    for name in _ALIASES.get(key, (key,)):
        if name in history and history[name] is not None:
            values = list(history[name])
            if values:
                return values
    return None


def _pct(values: Sequence[float], scale: float = 100.0) -> List[float]:
    return [float(v) * scale for v in values]


def _save(fig, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Training curves
# --------------------------------------------------------------------------- #
def plot_training_curves(history: Dict, out_dir: str, title: str = "",
                         tag: str = "", chance: Optional[float] = None) -> List[str]:
    """Draw every metric family of a training history.

    ``history`` is the dictionary a trainer writes to ``history.json``.
    ``out_dir``  where the figures go (the model's own directory).
    ``chance``   the chance level of the accuracy (e.g. 1/num_classes), drawn
                 as a reference line.
    """
    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []
    prefix = f"{tag}_" if tag else ""

    epochs = None
    for key in ("train_loss", "tl", "val_acc", "va", "train_acc", "ta", "val_loss", "vl"):
        s = get_series(history, key if key in _ALIASES else key)
        if s:
            epochs = list(range(1, len(s) + 1))
            break
    if epochs is None:
        return written

    # --- 1. the loss ------------------------------------------------------
    tr, va = get_series(history, "train_loss"), get_series(history, "val_loss")
    if tr or va:
        fig, ax = plt.subplots(figsize=(9, 5))
        if tr:
            ax.plot(epochs[:len(tr)], tr, label="train loss", color="#1E88E5", lw=2)
        if va:
            ax.plot(epochs[:len(va)], va, label="val loss", color="#E53935", lw=2)
        ax.set_xlabel("epoch"); ax.set_ylabel("loss")
        ax.set_title(f"{title} - loss".strip(" -")); ax.grid(True, ls="--", alpha=0.5)
        ax.legend()
        written.append(_save(fig, os.path.join(out_dir, f"{prefix}loss.png")))

    # --- 2. the accuracy --------------------------------------------------
    tr, va = get_series(history, "train_acc"), get_series(history, "val_acc")
    if tr or va:
        fig, ax = plt.subplots(figsize=(9, 5))
        if tr:
            ax.plot(epochs[:len(tr)], _pct(tr), label="train acc", color="#1E88E5", lw=2)
        if va:
            ax.plot(epochs[:len(va)], _pct(va), label="val acc", color="#43A047", lw=2)
        if chance is not None:
            ax.axhline(chance * 100, color="gray", ls=":", lw=1.5,
                       label=f"chance ({chance*100:.2f}%)")
        ax.set_xlabel("epoch"); ax.set_ylabel("accuracy (%)")
        ax.set_title(f"{title} - accuracy".strip(" -")); ax.grid(True, ls="--", alpha=0.5)
        ax.legend()
        written.append(_save(fig, os.path.join(out_dir, f"{prefix}accuracy.png")))

    # --- 3. F1 / precision / recall ---------------------------------------
    f1, pr, rc = (get_series(history, "val_f1"), get_series(history, "val_precision"),
                   get_series(history, "val_recall"))
    if f1 or pr or rc:
        fig, ax = plt.subplots(figsize=(9, 5))
        for series, name, colour in ((f1, "F1", "#8E24AA"), (pr, "precision", "#00897B"),
                                     (rc, "recall", "#FB8C00")):
            if series:
                ax.plot(epochs[:len(series)], _pct(series), label=name, lw=2, color=colour)
        ax.set_xlabel("epoch"); ax.set_ylabel("score (%)")
        ax.set_title(f"{title} - F1 / precision / recall".strip(" -"))
        ax.grid(True, ls="--", alpha=0.5); ax.legend()
        written.append(_save(fig, os.path.join(out_dir, f"{prefix}f1_prec_rec.png")))

    # --- 4. the WER and the exact match -----------------------------------
    wer, exact = get_series(history, "val_wer"), get_series(history, "val_exact")
    if wer or exact:
        fig, ax = plt.subplots(figsize=(9, 5))
        if wer:
            ax.plot(epochs[:len(wer)], _pct(wer), label="WER", color="#D81B60", lw=2)
        if exact:
            ax2 = ax.twinx()
            ax2.plot(epochs[:len(exact)], _pct(exact), label="exact match",
                     color="#00ACC1", lw=2, ls="--")
            ax2.set_ylabel("exact match (%)")
            ax2.legend(loc="lower right")
        ax.set_xlabel("epoch"); ax.set_ylabel("WER (%)")
        ax.set_title(f"{title} - WER / exact match".strip(" -"))
        ax.grid(True, ls="--", alpha=0.5); ax.legend(loc="upper right")
        written.append(_save(fig, os.path.join(out_dir, f"{prefix}wer_exact.png")))

    # --- 5. the learning rates --------------------------------------------
    lr_keys = [k for k in history if k.startswith("lr") and isinstance(history[k], list)]
    if lr_keys:
        fig, ax = plt.subplots(figsize=(9, 5))
        for k in lr_keys:
            ax.plot(epochs[:len(history[k])], history[k], label=k, lw=1.8)
        ax.set_xlabel("epoch"); ax.set_ylabel("learning rate")
        ax.set_title(f"{title} - learning rate".strip(" -"))
        ax.set_yscale("log"); ax.grid(True, ls="--", alpha=0.5); ax.legend()
        written.append(_save(fig, os.path.join(out_dir, f"{prefix}lr.png")))

    return written


# --------------------------------------------------------------------------- #
# The confusion matrix
# --------------------------------------------------------------------------- #
def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int) -> np.ndarray:
    """The raw confusion matrix (rows = the truth, columns = the prediction)."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        t, p = int(t), int(p)
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Optional[Sequence[str]] = None,
    out_path: str = "confusion_matrix.png",
    normalize: str = "true",
    top_n: Optional[int] = 30,
    title: str = "",
    report_path: Optional[str] = None,
    min_class_support: int = 1,
    prefix: str = "",
) -> Tuple[str, Optional[str]]:
    """The confusion matrix + a readable TEXT report.

    ``normalize``
        ``"true"``  row-normalised (the per-class RECALL - the honest view when
                    the classes are imbalanced),
        ``"pred"``  column-normalised (the per-class PRECISION),
        ``"none"``  the raw counts.
    ``top_n``   keep only the N classes with the most errors, so the figure is
                readable (a 228x228 matrix is not).  ``None`` keeps them all.

    Returns ``(figure_path, report_path)``.
    """
    y_true = np.asarray(list(y_true), dtype=np.int64)
    y_pred = np.asarray(list(y_pred), dtype=np.int64)
    n = int(max(y_true.max(initial=0), y_pred.max(initial=0)) + 1)
    if class_names is not None:
        n = max(n, len(class_names))
    cm = confusion_matrix(y_true, y_pred, n)

    # ---- the classes to show --------------------------------------------
    support = cm.sum(axis=1)
    errors = support - np.diag(cm)
    if top_n is not None and n > top_n:
        keep = np.argsort(-errors)[:top_n]
        keep = np.sort(keep)
        cm_show = cm[np.ix_(keep, keep)]
        names_show = ([class_names[i] for i in keep] if class_names else [str(i) for i in keep])
    else:
        cm_show = cm
        names_show = list(class_names) if class_names else [str(i) for i in range(n)]

    # ---- the normalisation ------------------------------------------------
    if normalize == "true":
        with np.errstate(invalid="ignore", divide="ignore"):
            mat = cm_show / np.maximum(cm_show.sum(axis=1, keepdims=True), 1)
    elif normalize == "pred":
        with np.errstate(invalid="ignore", divide="ignore"):
            mat = cm_show / np.maximum(cm_show.sum(axis=0, keepdims=True), 1)
    else:
        mat = cm_show.astype(float)

    size = max(8, min(28, 0.35 * len(names_show) + 6))
    fig, ax = plt.subplots(figsize=(size, size))
    if _HAS_SNS:
        sns.heatmap(mat, ax=ax, cmap="Blues", xticklabels=names_show,
                    yticklabels=names_show, annot=False,
                    cbar_kws={"label": {"true": "row-normalised recall",
                                        "pred": "column-normalised precision"}.get(normalize, "count")})
    else:
        im = ax.imshow(mat, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(names_show))); ax.set_xticklabels(names_show, rotation=90, fontsize=6)
        ax.set_yticks(range(len(names_show))); ax.set_yticklabels(names_show, fontsize=6)
        fig.colorbar(im, ax=ax)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(title or f"Confusion matrix ({len(names_show)} classes, {normalize}-normalised)")
    figure_path = _save(fig, out_path)

    # ---- the text report --------------------------------------------------
    report_file = None
    if report_path:
        per_class = []
        for i in range(n):
            s = int(cm[i].sum())
            if s < min_class_support:
                continue
            tp = int(cm[i, i])
            rec = tp / s if s else 0.0
            col = int(cm[:, i].sum())
            prec = tp / col if col else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
            per_class.append((i, names_show[i] if i < len(names_show) else str(i),
                              s, tp, prec, rec, f1))
        per_class.sort(key=lambda r: r[6])

        lines = ["=" * 78, "CONFUSION MATRIX REPORT", "=" * 78,
                 f"classes: {n}   samples: {len(y_true)}   accuracy: "
                 f"{100*float((y_true == y_pred).mean()):.2f}%",
                 "", f"{'class':<24}{'support':>8}{'correct':>9}{'prec':>8}{'rec':>8}{'F1':>8}",
                 "-" * 78]
        for i, name, s, tp, prec, rec, f1 in per_class:
            lines.append(f"{name[:23]:<24}{s:>8}{tp:>9}{prec*100:>7.1f}%{rec*100:>7.1f}%{f1*100:>7.1f}%")
        lines += ["", "the 20 worst confusions:", "-" * 78]
        pairs = []
        for i in range(n):
            for j in range(n):
                if i != j and cm[i, j] > 0:
                    pairs.append((int(cm[i, j]), names_show[i] if i < len(names_show) else str(i),
                                  names_show[j] if j < len(names_show) else str(j)))
        pairs.sort(reverse=True)
        for cnt, a, b in pairs[:20]:
            lines.append(f"  {cnt:>5}   {a}  ->  {b}")
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        report_file = report_path

    return figure_path, report_file


def plot_per_class_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Optional[Sequence[str]] = None,
    out_path: str = "per_class_metrics.png",
    title: str = "",
) -> str:
    """The per-class precision / recall / F1 as three sorted bar charts."""
    y_true = np.asarray(list(y_true), dtype=np.int64)
    y_pred = np.asarray(list(y_pred), dtype=np.int64)
    n = int(max(y_true.max(initial=0), y_pred.max(initial=0)) + 1)
    cm = confusion_matrix(y_true, y_pred, n)

    prec, rec, f1, names = [], [], [], []
    for i in range(n):
        s = int(cm[i].sum())
        if s == 0:
            continue
        tp = int(cm[i, i])
        col = int(cm[:, i].sum())
        p = tp / col if col else 0.0
        r = tp / s
        prec.append(p * 100); rec.append(r * 100)
        f1.append(0.0 if (p + r) == 0 else 2 * p * r / (p + r) * 100)
        names.append(class_names[i] if class_names and i < len(class_names) else str(i))

    if not names:
        return ""
    order = np.argsort(f1)
    fig, axes = plt.subplots(1, 3, figsize=(18, max(4, 0.25 * len(names) + 3)))
    for ax, values, name, colour in ((axes[0], prec, "precision", "#00897B"),
                                     (axes[1], rec, "recall", "#FB8C00"),
                                     (axes[2], f1, "F1", "#8E24AA")):
        ax.barh(range(len(values)), [values[i] for i in order], color=colour)
        ax.set_yticks(range(len(values)))
        ax.set_yticklabels([names[i] for i in order], fontsize=5)
        ax.set_xlabel(f"{name} (%)"); ax.grid(True, axis="x", ls="--", alpha=0.5)
    fig.suptitle(title or "Per-class metrics (worst first)")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_label_distribution(
    counts: Dict[str, int],
    out_path: str = "label_distribution.png",
    title: str = "",
    top: int = 40,
) -> str:
    """The label prior of a corpus: the classes and how often they occur."""
    if not counts:
        return ""
    items = sorted(counts.items(), key=lambda kv: -kv[1])[:top]
    total = sum(counts.values())
    fig, ax = plt.subplots(figsize=(max(8, 0.28 * len(items) + 4), 6))
    ax.bar(range(len(items)), [c for _, c in items], color="#1E88E5")
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([k for k, _ in items], rotation=90, fontsize=6)
    ax.set_ylabel("word instances")
    ax.set_title(title or f"Label distribution ({len(counts)} classes, {total} instances)")
    ax.grid(True, axis="y", ls="--", alpha=0.5)
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #
def write_report(history: Dict, out_dir: str, title: str = "",
                 chance: Optional[float] = None, prefix: str = "",
                 y_true: Optional[Sequence[int]] = None,
                 y_pred: Optional[Sequence[int]] = None,
                 class_names: Optional[Sequence[str]] = None) -> List[str]:
    """The full report of a training run: the figures + a text summary.

    When the validation predictions are given, the confusion matrix and the
    per-class metrics are drawn as well (they are the most informative figures
    of a classifier, and they need the (true, predicted) pairs - the
    classification report only has the aggregated numbers).
    """
    os.makedirs(out_dir, exist_ok=True)
    written = plot_training_curves(history, out_dir, title=title, chance=chance,
                                   tag=prefix.rstrip("_"))

    if y_true is not None and y_pred is not None:
        fig, rep = plot_confusion_matrix(
            y_true, y_pred, class_names=class_names,
            out_path=os.path.join(out_dir, f"{prefix}confusion_matrix.png"),
            normalize="true", top_n=40,
            title=f"{title} - confusion matrix (row-normalised recall)".strip(" -"),
            report_path=os.path.join(out_dir, f"{prefix}confusion_report.txt"))
        if fig:
            written.append(fig)
        if rep:
            written.append(rep)
        per = plot_per_class_metrics(
            y_true, y_pred, class_names=class_names,
            out_path=os.path.join(out_dir, f"{prefix}per_class_metrics.png"),
            title=f"{title} - per-class metrics".strip(" -"))
        if per:
            written.append(per)

    lines = ["=" * 78, f"TRAINING REPORT - {title}" if title else "TRAINING REPORT", "=" * 78]
    # the metrics where a HIGHER value is better
    higher_is_better = {"train_acc", "val_acc", "val_f1", "val_precision",
                         "val_recall", "val_exact"}
    for key in ("train_loss", "val_loss", "train_acc", "val_acc", "val_f1",
                "val_precision", "val_recall", "val_wer", "val_exact"):
        s = get_series(history, key)
        if not s:
            continue
        best = max(s) if key in higher_is_better else min(s)
        lines.append(f"  {key:<16} n={len(s):>3}  first={s[0]:.4f}  "
                     f"best={best:.4f}  last={s[-1]:.4f}")
    if chance is not None:
        lines.append(f"  chance level    {chance*100:.2f}%")
    lines.append("=" * 78)
    with open(os.path.join(out_dir, f"{prefix}training_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    written.append(os.path.join(out_dir, f"{prefix}training_report.txt"))
    return written
