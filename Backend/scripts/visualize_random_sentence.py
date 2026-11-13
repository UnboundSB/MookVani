"""
MookVani — Random Sentence Visualizer (OpenCV)
===============================================

Picks a random continuous sentence from the sentence-level tensor corpus,
loads its (T, 163) landmark tensor, and pops up an OpenCV window showing:

  * the ground-truth sentence (in-signing-order words)
  * a few sampled frames as stick-figure pose previews
  * per-frame info (T frames, duration, etc.)

Run from the repo root:

    python Backend/scripts/visualize_random_sentence.py

Options:
  --data_dir   Backend/data/tensors_sentence_level   (default)
  --index      index.json                            (default: inside data_dir)
  --classes    Backend/models/word_vocabulary/classes.json
  --n          1                                     sentences to show
  --wait_ms    2000                                  ms to show each sentence (0 = wait for key)
  --save       none                                  optional output png path

Press 'q' to quit early, any other key to advance (when --wait_ms 0).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = BACKEND_DIR / "data" / "tensors_sentence_level"
DEFAULT_CLASSES = BACKEND_DIR / "models" / "word_vocabulary" / "classes.json"


def load_index(data_dir: Path) -> list[dict]:
    index_file = data_dir / "index.json"
    if not index_file.exists():
        raise FileNotFoundError(f"Missing {index_file} — run scripts/extract_real_sentences.py first.")
    with open(index_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        payload = payload.get("items", payload.get("clips", []))
    return list(payload)


def load_classes(classes_path: Path) -> list[str]:
    with open(classes_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return list(payload["classes"])
    return list(payload)


def draw_stick_frame(
    frame: np.ndarray,
    pose: np.ndarray,
    classes: list[str],
    word_idx: int | None = None,
    word_text: str = "",
    frame_idx: int = 0,
    total_frames: int = 0,
) -> np.ndarray:
    """Draw a single pose frame as a stick-figure preview + HUD."""
    h, w = frame.shape[:2]
    overlay = np.zeros_like(frame)

    # Pose layout (163 dims):
    #   [0:63]   left hand  (21 pts x 3)
    #   [63:126] right hand (21 pts x 3)
    #   [126:138] arms      (4 pts x 3: ls, rs, le, re)
    #   [138:159] face      (7 pts x 3)
    #   [159:163] flags     [lh, rh, arms, face]

    def pts(block: tuple[int, int]) -> list[tuple[int, int]]:
        lo, hi = block
        block_vals = pose[lo:hi]                      # (J*3,)
        block_pts = block_vals.reshape(-1, 3)         # (J, 3)
        out = []
        for j in range(block_pts.shape[0]):
            x, y, _ = block_pts[j]
            # coordinates are anchor-relative / normalised; map into image space
            px = int((x + 4.0) / 8.0 * w)   # rough mapping: x in [-4, +4] -> [0, w]
            py = int((y + 4.0) / 8.0 * h)   # rough mapping: y in [-4, +4] -> [0, h]
            px = max(0, min(w - 1, px))
            py = max(0, min(h - 1, py))
            out.append((px, py))
        return out

    # draw connections for hands (simple: draw all points)
    for block, color in (
        ((0, 63), (0, 255, 0)),      # left hand — green
        ((63, 126), (0, 255, 255)),  # right hand — yellow
        ((126, 138), (255, 255, 0)), # arms — cyan
        ((138, 159), (255, 0, 255)), # face — magenta
    ):
        for px, py in pts(block):
            cv2.circle(overlay, (px, py), 3, color, -1)

    # HUD
    cv2.rectangle(overlay, (0, 0), (w, 120), (20, 20, 20), -1)
    cv2.putText(
        overlay,
        f"SENTENCE: {word_text}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        overlay,
        f"frame {frame_idx + 1}/{total_frames}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
    )
    cv2.putText(
        overlay,
        "q=quit  any key=next",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (150, 150, 150),
        1,
    )

    # blend pose overlay with a dark background
    bg = np.full_like(frame, 30)
    out = cv2.addWeighted(bg, 0.6, overlay, 0.9, 0)
    return out


def visualize_sentence(
    item: dict,
    classes: list[str],
    wait_ms: int = 2000,
    save_path: str | None = None,
) -> None:
    tensor_path = item.get("tensor_path", "")
    if not tensor_path or not os.path.exists(tensor_path):
        print(f"[SKIP] tensor not found: {tensor_path}")
        return

    seq = torch.load(tensor_path, weights_only=True).float()
    T = seq.shape[0]
    words = item.get("in_scope_words") or item.get("target_words") or []
    sentence = " ".join(words)

    print(f"[SHOW] {os.path.basename(tensor_path)} | T={T} frames | '{sentence}'")

    # show a few sampled frames
    sample_idxs = np.linspace(0, T - 1, min(6, T)).astype(int)
    for i, fi in enumerate(sample_idxs):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pose = seq[fi].numpy()
        out = draw_stick_frame(
            frame,
            pose,
            classes,
            word_text=sentence,
            frame_idx=int(fi),
            total_frames=T,
        )
        cv2.imshow("MookVani — Random Sentence", out)
        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            return

    if save_path:
        cv2.imwrite(save_path, out)
        print(f"[SAVED] {save_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize a random ISL sentence in OpenCV.")
    p.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR), help="Sentence tensor dir (contains index.json)")
    p.add_argument("--classes", default=str(DEFAULT_CLASSES), help="Published vocabulary classes.json")
    p.add_argument("--n", type=int, default=1, help="How many random sentences to show")
    p.add_argument("--wait_ms", type=int, default=2000, help="ms per sentence (0 = wait for keypress)")
    p.add_argument("--save", default=None, help="Optional path to save last frame as PNG")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    classes = load_classes(Path(args.classes))
    items = load_index(data_dir)

    if not items:
        print("[ERR] No items in index.json.")
        return

    # remap targets onto current vocabulary (same as dataloader does)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    valid_items = []
    for it in items:
        words = it.get("in_scope_words") or []
        if words and all(w in class_to_idx for w in words):
            valid_items.append(it)

    print(f"[INFO] {len(valid_items)}/{len(items)} sentences fully in vocabulary.")

    rng = random.Random(42)
    chosen = rng.sample(valid_items, min(args.n, len(valid_items)))

    for it in chosen:
        visualize_sentence(it, classes, wait_ms=args.wait_ms, save_path=args.save)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
