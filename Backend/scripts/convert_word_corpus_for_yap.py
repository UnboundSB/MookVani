"""Convert the word corpus into the format the YAP segmentor expects.

The segmentor (``Backend/segmentor/``) was written against a DIFFERENT corpus
layout than the one we have:

    what the segmentor expects          what we have
    --------------------------------    ---------------------------------------
    .npy files, (T, J*3) float32        .pt tensors, (T, 163) float32
    J = 42 (2 MediaPipe hands)           163 = hands(126) + arms(12)
                                                + face(21) + flags(4)

``load_clips()`` in ``train_noise_filter.py`` scans for ``*.npy`` and
``WordClipDataset.__getitem__`` does ``np.load(...)`` -> on our corpus it finds
NOTHING and the stage-1 training would silently run on an empty dataset.

This script converts the corpus:

    data/tensors_include50_word_level/<class>/<clip>.pt   (T, 163)
        ->  data/yap_word_level/<class>/<clip>.npy          (T, 53*3)

The 53 landmarks are EXACTLY the layout of ``YAPConfig.corpus()``:

    landmarks  0..20  the left hand   (MediaPipe HandLandmarker, 21 points)
    landmarks 21..41  the right hand  (MediaPipe HandLandmarker, 21 points)
    landmarks 42..45  the upper body  (MediaPipe Pose: the 2 shoulders + the 2 elbows)
    landmarks 46..52  the face         (MediaPipe FaceMesh: 2 eyes, nose tip,
                                         2 cheeks, 2 mouth corners)

which is also the order of the parts inside the YAP pose vector
(left_hand, right_hand, upper_body, face).  The 4 presence flags are dropped
(the segmentor derives its own "valid" flags) and the 159 landmark channels
are kept verbatim.

Usage
-----
    python scripts/convert_word_corpus_for_yap.py
    python scripts/convert_word_corpus_for_yap.py --out_dir data/yap_word_level --max_per_class 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from utils.dataloader_word import (  # noqa: E402
    MAX_SAMPLES_PER_CLASS,
    MIN_SAMPLES_PER_CLASS,
    WORD_LEVEL_DIR_DEFAULT,
    discover_word_classes,
)

# The canonical layout (Backend/utils/pose_augmentation.py)
LH_LO, LH_HI = 0, 63
RH_LO, RH_HI = 63, 126
LANDMARK_DIMS = 159          # 53 landmarks x 3  (the 4 flags are dropped)
J = 53                       # 21 + 21 + 4 + 7
YAP_OUT = os.path.join(BACKEND_DIR, "data", "yap_word_level")


def to_yap_pose(clip: torch.Tensor) -> np.ndarray:
    """(T, 163) -> (T, 53*3) float32: the 53 landmarks, in the corpus order.

    The corpus order IS the YAP part order (left_hand, right_hand, upper_body,
    face), so the landmarks are copied verbatim - nothing is re-indexed.
    """
    x = clip.float().numpy()
    if x.ndim == 3:                     # (1, T, 163) -> (T, 163)
        x = x[0]
    return x[:, :LANDMARK_DIMS].astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Convert the word corpus for the YAP segmentor")
    ap.add_argument("--data_dir", default=WORD_LEVEL_DIR_DEFAULT)
    ap.add_argument("--out_dir", default=YAP_OUT)
    ap.add_argument("--min_samples_per_class", type=int, default=MIN_SAMPLES_PER_CLASS)
    ap.add_argument("--max_per_class", type=int, default=MAX_SAMPLES_PER_CLASS)
    ap.add_argument("--classes", default=os.path.join(BACKEND_DIR, "models", "word_vocabulary",
                                                     "classes.json"))
    args = ap.parse_args()

    infos = discover_word_classes(args.data_dir, args.min_samples_per_class, args.max_per_class)
    print(f"[CONVERT] {len(infos)} classes from {args.data_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    total = 0
    for info in infos:
        out_dir = os.path.join(args.out_dir, info.name)
        os.makedirs(out_dir, exist_ok=True)
        for path in info.files:
            src = torch.load(path, weights_only=True)
            arr = to_yap_pose(src)
            dst = os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + ".npy")
            np.save(dst, arr)
            total += 1

    # the class list the segmentor reads
    with open(args.classes, "r", encoding="utf-8") as f:
        classes = json.load(f)
    with open(os.path.join(args.out_dir, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)

    print(f"[CONVERT] {total} clips -> {args.out_dir}")
    print(f"[CONVERT] format: (T, {J*3}) float32, the 53 landmarks of the corpus")
    print(f"[CONVERT]   0..20 left hand | 21..41 right hand | 42..45 shoulders+elbows | 46..52 face")
    print(f"[CONVERT] now train the segmentor with YAPConfig.corpus():")
    print(f"           python segmentor/train_noise_filter.py --data_dirs {args.out_dir}")


if __name__ == "__main__":
    main()
