"""
Dataset Keypoint Generator (MediaPipe Tasks API)
================================================

Walks `images/data/<CLASS>/*.jpg`, extracts 21 hand landmarks per hand, and
appends the exact 42-feature representation used by the training notebook.

Expected layout:
    images/
      data/
        1/
          0.jpg
          1.jpg
          ...
        A/
        0.jpg
        B/
        0.jpg
        ...

Output:
    keypoint.csv rows are:
    <label>, x0, y0, x1, y1, ..., x20, y20

Run:
    .venv\\Scripts\\python.exe dataset_keypoint_generation.py
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core import base_options as mp_base_options

import mediapipe as mp


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = BASE_DIR / "images" / "data"
DEFAULT_TASK_PATH = BASE_DIR / "hand_landmarker.task"
DEFAULT_OUTPUT = BASE_DIR / "keypoint.csv"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def iter_image_files(data_root: Path) -> Iterable[Path]:
    for class_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        label = class_dir.name
        if not label:
            continue

        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield image_path


def calc_landmark_list(image_shape, norm_landmarks) -> List[List[int]]:
    h, w = int(image_shape[0]), int(image_shape[1])
    points = []

    for lm in norm_landmarks:
        x = max(0, min(w - 1, int(float(lm.x) * w)))
        y = max(0, min(h - 1, int(float(lm.y) * h)))
        points.append([x, y])

    return points


def pre_process_landmark(landmark_list: List[List[int]]) -> List[float]:
    tmp = copy.deepcopy(landmark_list)

    base_x, base_y = tmp[0]
    for i in range(len(tmp)):
        tmp[i][0] -= base_x
        tmp[i][1] -= base_y

    flat = list(itertools.chain.from_iterable(tmp))
    max_value = max(map(abs, flat))

    if max_value == 0:
        return [0.0 for _ in flat]

    return [float(v) / float(max_value) for v in flat]


def build_detector(task_path: Path) -> mp_vision.HandLandmarker:
    return mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_base_options.BaseOptions(model_asset_path=str(task_path)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate ISL keypoint CSV.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--task-path", type=Path, default=DEFAULT_TASK_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--append", action="store_true", help="Append to an existing CSV.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images.")
    args = parser.parse_args()

    if not args.data_root.is_dir():
        print(f"[ERROR] Data root not found: {args.data_root}")
        print("Expected layout: images/data/<CLASS>/<image.jpg>")
        return 1

    if not args.task_path.is_file() or args.task_path.stat().st_size == 0:
        print(f"[ERROR] MediaPipe task model not found: {args.task_path}")
        return 1

    detector = None
    processed = 0
    written = 0

    try:
        detector = build_detector(args.task_path)

        mode = "a" if args.append else "w"
        args.output.parent.mkdir(parents=True, exist_ok=True)

        with open(args.output, mode, newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)

            for image_path in iter_image_files(args.data_root):
                if args.limit and processed >= args.limit:
                    break

                processed += 1
                label = image_path.parent.name

                bgr = cv2.imread(str(image_path))
                if bgr is None:
                    print(f"[WARN] unreadable image: {image_path}")
                    continue

                # Flip for selfie-style coordinate consistency.
                bgr = cv2.flip(bgr, 1)
                rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                result = detector.detect(mp_image)

                for hand_lms in getattr(result, "hand_landmarks", None) or []:
                    landmark_list = calc_landmark_list(bgr.shape, hand_lms)
                    features = pre_process_landmark(landmark_list)
                    writer.writerow([label, *features])
                    written += 1

        print(f"[DONE] images_scanned={processed} rows_written={written} output={args.output}")
        return 0

    finally:
        if detector is not None:
            try:
                detector.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())