"""Build the full plotting report of a trained model.

It reads what the trainers write (``history.json`` and, when it is there, the
classification report) and produces every figure in one place:

    loss.png              the train / val loss
    accuracy.png          the train / val accuracy (+ the chance level)
    f1_prec_rec.png       the validation F1, precision and recall
    wer_exact.png         the validation WER and the exact-match
    lr.png                the learning rate of every parameter group
    confusion_matrix.png  the confusion matrix (row-normalised = per-class recall)
    per_class_metrics.png the per-class precision / recall / F1, worst first
    confusion_report.txt  the readable report next to the matrix
    training_report.txt    the summary of the run

Usage
-----
    python scripts/plot_model_report.py --model_dir models/phase6_word_model
    python scripts/plot_model_report.py --model_dir models/phase6_word_model \
        --classes models/word_vocabulary/classes.json
"""
import argparse
import json
import os
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from utils.plotting import (  # noqa: E402
    plot_confusion_matrix,
    plot_per_class_metrics,
    write_report,
)

DEFAULT_CLASSES = os.path.join(BACKEND_DIR, "models", "word_vocabulary", "classes.json")


def load_classes(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["classes"] if isinstance(payload, dict) else payload


def parse_report(text):
    """The (true, predicted) pairs of a sklearn classification report are not
    recoverable, so the confusion matrix is rebuilt from the saved predictions
    when they exist, and otherwise the matrix is skipped."""
    return None


def main():
    ap = argparse.ArgumentParser(description="Plotting report of a trained model")
    ap.add_argument("--model_dir", required=True,
                    help="The model directory (it holds history.json)")
    ap.add_argument("--classes", default=DEFAULT_CLASSES,
                    help="The class list (for the labels of the matrix)")
    ap.add_argument("--out_dir", default=None,
                    help="Where the figures go (default: the model directory)")
    ap.add_argument("--predictions", default=None,
                    help="A .json with {'y_true': [...], 'y_pred': [...]} to draw the "
                         "confusion matrix from")
    ap.add_argument("--title", default=None, help="The title of the figures")
    args = ap.parse_args()

    out_dir = args.out_dir or args.model_dir
    os.makedirs(out_dir, exist_ok=True)
    title = args.title or os.path.basename(os.path.normpath(args.model_dir))
    classes = load_classes(args.classes)

    history_path = os.path.join(args.model_dir, "history.json")
    history = {}
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)

    chance = (1.0 / len(classes)) if classes else None
    written = write_report(history, out_dir, title=title, chance=chance)
    for p in written:
        print(f"  wrote {p}")

    # ---- the confusion matrix -------------------------------------------
    y_true = y_pred = None
    if args.predictions and os.path.exists(args.predictions):
        with open(args.predictions, "r", encoding="utf-8") as f:
            payload = json.load(f)
        y_true, y_pred = payload["y_true"], payload["y_pred"]

    if y_true is not None:
        fig, rep = plot_confusion_matrix(
            y_true, y_pred, class_names=classes,
            out_path=os.path.join(out_dir, "confusion_matrix.png"),
            normalize="true", top_n=40,
            title=f"{title} - confusion matrix (row-normalised recall)",
            report_path=os.path.join(out_dir, "confusion_report.txt"))
        print(f"  wrote {fig}")
        if rep:
            print(f"  wrote {rep}")
        per = plot_per_class_metrics(y_true, y_pred, class_names=classes,
                                     out_path=os.path.join(out_dir, "per_class_metrics.png"),
                                     title=f"{title} - per-class metrics")
        if per:
            print(f"  wrote {per}")
    else:
        print("  (no --predictions given: the confusion matrix was skipped.  Save the "
              "validation predictions as {'y_true': [...], 'y_pred': [...]} to draw it.)")

    print(f"\nReport written to {out_dir}")


if __name__ == "__main__":
    main()
