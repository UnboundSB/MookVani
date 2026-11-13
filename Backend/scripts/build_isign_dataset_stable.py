"""
Backend/scripts/build_isign_dataset_stable.py

Builds the full iSign word-level tensor dataset from archive_extracted.

Pipeline:
  1. Scan F:\dataset\archive_extracted for word folders
  2. Filter to classes present in classes.json with >= min_clips clips
  3. Cap each class at max_samples clips (max 22 = all remaining videos)
  4. Extract 163-dim MediaPipe landmarks per frame with per-coordinate
     carry-forward stability (no zero-reset jitter)
  5. Save as (T, 163) .pt tensors in Backend/data/tensors_include50_word_level/
  6. Supports --skip-existing to resume

Usage:
    .\venv\Scripts\python.exe Backend\scripts\build_isign_dataset_stable.py \
        --max-classes 0 --max-samples 22 --min-clips 14 --jobs 8 --skip-existing

--max-classes 0 means NO class cap (extract ALL qualifying classes).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BACKEND))

from stable_extraction import extract_stable_frame, new_trackers  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ISIGN_ARCHIVE_EXTRACTED = Path(r"F:\dataset\archive_extracted")
CLASSES_JSON = BACKEND / "models" / "word_vocabulary" / "classes.json"
OUTPUT_DIR = BACKEND / "data" / "tensors_include50_word_level"

# MediaPipe model assets (in D:\mookvani\)
MODEL_DIR = Path(r"D:\mookvani")
HAND_MODEL = MODEL_DIR / "hand_landmarker.task"
POSE_MODEL = MODEL_DIR / "pose_landmarker.task"
FACE_MODEL = MODEL_DIR / "face_landmarker.task"

MAX_CLASSES_DEFAULT = 0          # 0 = no cap (all classes)
MAX_SAMPLES_PER_CLASS_DEFAULT = 22
MIN_CLIPS_DEFAULT = 14
DEFAULT_JOBS = 4


def clean_word(w: str) -> str:
    w = str(w).strip().lower()
    w = re.sub(r"^\d+\.\s*", "", w)
    w = re.sub(r"[^\w\s]", "", w)
    return w.strip()


def load_classes(path: Path = CLASSES_JSON) -> List[str]:
    with open(path, "r") as f:
        return json.load(f)


def discover_archive_words(root: Path = ISIGN_ARCHIVE_EXTRACTED) -> Dict[str, List[Path]]:
    """{clean_word_name: [video_paths, ...]} from archive_extracted."""
    word_map: Dict[str, List[Path]] = {}
    for cat_dir in sorted(root.iterdir()):
        if not cat_dir.is_dir():
            continue
        for word_dir in sorted(cat_dir.iterdir()):
            if not word_dir.is_dir():
                continue
            cn = clean_word(word_dir.name)
            videos = sorted(
                p for p in word_dir.iterdir()
                if p.is_file() and p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv")
            )
            if cn and videos:
                word_map.setdefault(cn, []).extend(videos)
    return word_map


def select_classes(word_map, classes_json, max_classes=0, max_samples=22,
                   seed=42, min_clips=14) -> Dict[str, List[Path]]:
    classes_set = set(classes_json)
    eligible = {}
    for word, videos in word_map.items():
        if word in classes_set and len(videos) >= min_clips:
            eligible[word] = videos

    print(f"Classes in classes.json found in archive with >= {min_clips} clips: {len(eligible)}")

    # Sort by video count descending (deterministic)
    sorted_words = sorted(eligible.items(), key=lambda x: (-len(x[1]), x[0]))

    if max_classes and max_classes > 0:
        sorted_words = sorted_words[:max_classes]

    selected = {}
    rng = random.Random(seed)
    for word, videos in sorted_words:
        if len(videos) > max_samples:
            selected[word] = rng.sample(videos, max_samples)
        else:
            selected[word] = videos
    return selected


# ---------------------------------------------------------------------------
# Per-video extraction worker (runs in subprocess)
# ---------------------------------------------------------------------------

def _init_worker():
    """Each worker initializes its own MediaPipe landmarkers (not shareable across processes)."""
    global _hand_lm, _pose_lm, _face_lm, _fps_default
    _fps_default = 30.0

    BaseOptions = mp.tasks.BaseOptions
    VM = mp.tasks.vision.RunningMode.VIDEO

    _hand_lm = mp.tasks.vision.HandLandmarker.create_from_options(
        mp.tasks.vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=VM, num_hands=2))
    _pose_lm = mp.tasks.vision.PoseLandmarker.create_from_options(
        mp.tasks.vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
            running_mode=VM))
    _face_lm = mp.tasks.vision.FaceLandmarker.create_from_options(
        mp.tasks.vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(FACE_MODEL)),
            running_mode=VM))


def _ensure_worker():
    """Initialize MediaPipe landmarkers in this process if not already done."""
    global _hand_lm, _pose_lm, _face_lm, _fps_default
    if '_hand_lm' not in globals():
        _init_worker()


def extract_single_video(args: Tuple[str, str]) -> Tuple[str, bool, int]:
    """
    Extract one video -> (T,163) tensor saved to out_path.
    Uses per-coordinate carry-forward stability within the video.
    """
    _ensure_worker()
    vid_path_str, out_path_str = args
    vid_path = Path(vid_path_str)
    out_path = Path(out_path_str)

    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        return vid_path_str, False, 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps) or fps <= 0:
        fps = _fps_default

    lh_t, rh_t, arms_t, face_t = new_trackers()

    frames_data = []
    frame_index = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            f_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=f_rgb)
            timestamp_ms = int(frame_index * (1000.0 / fps))

            hands = _hand_lm.detect_for_video(mp_image, timestamp_ms)
            pose = _pose_lm.detect_for_video(mp_image, timestamp_ms)
            face = _face_lm.detect_for_video(mp_image, timestamp_ms)

            frame_data = extract_stable_frame(hands, pose, face, lh_t, rh_t, arms_t, face_t)
            frames_data.append(frame_data)
            frame_index += 1
    finally:
        cap.release()

    if not frames_data:
        return vid_path_str, False, 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tensor = torch.from_numpy(np.stack(frames_data).astype(np.float32))  # (T, 163)
    torch.save(tensor, out_path)
    return vid_path_str, True, len(frames_data)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build stable iSign word-level tensor dataset")
    parser.add_argument("--archive-root", type=Path, default=ISIGN_ARCHIVE_EXTRACTED)
    parser.add_argument("--classes-json", type=Path, default=CLASSES_JSON)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-classes", type=int, default=MAX_CLASSES_DEFAULT,
                        help="Max classes (0 = no cap, extract all qualifying classes)")
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES_PER_CLASS_DEFAULT,
                        help="Max videos per class")
    parser.add_argument("--min-clips", type=int, default=MIN_CLIPS_DEFAULT,
                        help="Min clips per class to qualify")
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS,
                        help="Number of parallel workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip tensors that already exist")
    args = parser.parse_args()

    classes = load_classes(args.classes_json)
    print(f"Classes in classes.json: {len(classes)}")

    print(f"\nScanning {args.archive_root}...")
    word_map = discover_archive_words(args.archive_root)
    print(f"Total word folders found: {len(word_map)}")

    selected = select_classes(word_map, classes,
                              max_classes=args.max_classes,
                              max_samples=args.max_samples,
                              seed=args.seed,
                              min_clips=args.min_clips)
    print(f"\nSelected {len(selected)} classes:")

    # Create output dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    already_extracted = {}
    if args.skip_existing:
        for word in selected:
            d = args.output_dir / word
            if d.exists():
                n = len([f for f in d.iterdir() if f.suffix == ".pt"])
                if n > 0:
                    already_extracted[word] = n

    all_work = []
    for word, video_paths in sorted(selected.items(), key=lambda x: -len(x[1])):
        out_class_dir = args.output_dir / word
        out_class_dir.mkdir(parents=True, exist_ok=True)

        already_done = already_extracted.get(word, 0)
        if already_done >= len(video_paths):
            print(f"  [SKIP] {word}: already complete ({already_done}/{len(video_paths)})")
            continue

        print(f"  Extracting {word} ({len(video_paths)} videos, {already_done} already done)")
        for i, vid_path in enumerate(video_paths):
            out_name = f"{word}_{i+1:02d}.pt"
            out_path = out_class_dir / out_name
            if args.skip_existing and out_path.exists():
                continue
            all_work.append((str(vid_path), str(out_path)))

    print(f"\nTotal videos to extract: {len(all_work)}")

    total_extracted = 0
    total_failed = 0

    if args.jobs <= 1:
        for item in all_work:
            _, success, _ = extract_single_video(item)
            total_extracted += int(success)
            total_failed += int(not success)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init_worker) as executor:
            futures = {executor.submit(extract_single_video, item): item for item in all_work}
            for future in as_completed(futures):
                _, success, frame_count = future.result()
                if success:
                    total_extracted += 1
                else:
                    total_failed += 1
                    print(f"  [WARN] Failed: {futures[future]}")

    print(f"\n=== Extraction Complete ===")
    print(f"Classes: {len(selected)}")
    print(f"Total extracted: {total_extracted}")
    print(f"Total failed: {total_failed}")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
