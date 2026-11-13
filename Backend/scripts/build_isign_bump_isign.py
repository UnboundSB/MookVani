"""
Backend/scripts/build_isign_bump_isign.py

Bumps underfed classes in tensors_include50_word_level/ by extracting
additional videos from F:/dataset/isign_extracted/iSign-videos_v1.1/.

iSign videos are named by YouTube word_id (e.g. scJSFdl6vqk_w.mp4,
EUwbNKeZKvY--13.mp4). The iSign CSVs map word_id -> word. This script:

  1. Reads word_id -> word from both iSign CSVs (column 'word' only)
  2. Pre-computes which classes need bumping (already at max_samples = skip)
  3. Scans iSign videos in parallel (9 workers) with live progress counter
  4. Keeps only videos whose word_id maps to a word in classes.json that needs bumping
  5. Extracts 163-dim MediaPipe landmarks with per-coordinate carry-forward
  6. Saves as {word}_isign_{NN}.pt in tensors_include50_word_level/{word}/
  7. Supports --skip-existing to resume (per-file granularity)

Usage:
    .\venv\Scripts\python.exe Backend\scripts\build_isign_bump_isign.py ^
        --max-samples 22 --jobs 9 --skip-existing
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BACKEND))

ISIGN_EXTRACTED = Path(r"F:\dataset\isign_extracted")
ISIGN_VIDEOS = ISIGN_EXTRACTED / "iSign-videos_v1.1"
CLASSES_JSON = BACKEND / "models" / "word_vocabulary" / "classes.json"
OUTPUT_DIR = BACKEND / "data" / "tensors_include50_word_level"
CHECKPOINT_FILE = OUTPUT_DIR / "_bump_checkpoint.json"

MODEL_DIR = Path(r"D:\mookvani")
HAND_MODEL = MODEL_DIR / "hand_landmarker.task"
POSE_MODEL = MODEL_DIR / "pose_landmarker.task"
FACE_MODEL = MODEL_DIR / "face_landmarker.task"

MAX_SAMPLES_DEFAULT = 22
DEFAULT_JOBS = 9
MIN_CLIPS_DEFAULT = 1


def clean_word(w: str) -> str:
    w = str(w).strip().lower()
    w = re.sub(r"^\d+\.\s*", "", w)
    w = re.sub(r"[^\w\s]", "", w)
    return w.strip()


def load_classes(path: Path = CLASSES_JSON) -> List[str]:
    with open(path, "r") as f:
        return json.load(f)


def load_word_id_map(csv_dir: Path = ISIGN_EXTRACTED) -> Dict[str, str]:
    """Returns {word_id: word} from both iSign CSVs. Column 'word' only."""
    word_id_to_word: Dict[str, str] = {}
    for fname in ["word-presence-dataset_v1.1.csv", "word-description-dataset_v1.1.csv"]:
        fpath = csv_dir / fname
        if not fpath.exists():
            print(f"  [WARN] {fpath} not found, skipping")
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            r = csv.reader(f)
            next(r)  # skip header: word_id, word, sentence_id, sentence
            for row in r:
                if len(row) < 2:
                    continue
                word_id = row[0].strip()
                word = row[1].strip().lower()
                if word_id and word:
                    word_id_to_word[word_id] = word
    return word_id_to_word


def count_extracted_per_class(output_dir: Path, classes: List[str]) -> Dict[str, int]:
    """Count existing .pt files per class directory."""
    counts: Dict[str, int] = {}
    for word in classes:
        d = output_dir / word
        if d.exists():
            n = len([f for f in d.iterdir() if f.suffix == ".pt"])
            counts[word] = n
        else:
            counts[word] = 0
    return counts


# ---------------------------------------------------------------------------
# Stable per-coordinate tracker (carry-forward on zero-reset)
# ---------------------------------------------------------------------------
class CoordTracker:
    def __init__(self, n_coords: int, n_dims: int = 3):
        self.last = np.zeros((n_coords, n_dims), dtype=np.float32)

    def update(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=np.float32)
        out = coords.copy()
        zero_mask = coords == 0.0
        out[zero_mask] = self.last[zero_mask]
        nz = ~zero_mask
        self.last[nz] = coords[nz]
        return out

    def reset(self):
        self.last[:] = 0.0


def new_trackers():
    return (CoordTracker(21, 3),   # left hand
            CoordTracker(21, 3),   # right hand
            CoordTracker(4, 3),    # arms (L/R shoulder, L/R elbow)
            CoordTracker(7, 3))    # face (L-eye, R-eye, nose, L-cheek, R-cheek, L-mouth, R-mouth)


FACE_IDX = [159, 386, 4, 234, 454, 61, 291]


def _normalize_skeleton(lh, rh, arms, face_pts, lh_flag, rh_flag, arms_flag, face_flag):
    anchor = np.array([0.5, 0.5, 0.0])
    if face_flag:
        anchor = face_pts[2]
    elif arms_flag:
        anchor = (arms[0] + arms[1]) / 2.0
    elif lh_flag:
        anchor = lh[0]
    elif rh_flag:
        anchor = rh[0]

    scale = 0.3
    if arms_flag:
        pose_scale = np.linalg.norm(arms[0] - arms[1])
        if pose_scale > 0.05:
            scale = pose_scale
    elif face_flag:
        face_scale = np.linalg.norm(face_pts[3] - face_pts[4]) * 3.0
        if face_scale > 0.05:
            scale = face_scale

    lh_out = (lh - anchor) if lh_flag else np.ones((21, 3)) * np.array([-1.5 * scale, 3.0 * scale, 0.0])
    rh_out = (rh - anchor) if rh_flag else np.ones((21, 3)) * np.array([1.5 * scale, 3.0 * scale, 0.0])
    arms_out = (arms - anchor) if arms_flag else np.zeros((4, 3))
    face_out = (face_pts - anchor) if face_flag else np.zeros((7, 3))

    return lh_out / scale, rh_out / scale, arms_out / scale, face_out / scale


def extract_stable_frame(hands_result, pose_result, face_result,
                         lh_t, rh_t, arms_t, face_t):
    lh_raw = np.zeros((21, 3), dtype=np.float32)
    rh_raw = np.zeros((21, 3), dtype=np.float32)
    lh_flag = 0.0
    rh_flag = 0.0
    if hands_result and hands_result.hand_landmarks:
        for idx, handedness in enumerate(hands_result.handedness):
            label = handedness[0].category_name
            pts = np.array([[lm.x, lm.y, lm.z] for lm in hands_result.hand_landmarks[idx]],
                           dtype=np.float32)
            if label == 'Left':
                lh_raw = pts
                lh_flag = 1.0
            else:
                rh_raw = pts
                rh_flag = 1.0

    lh = lh_t.update(lh_raw) if lh_flag else lh_t.last.copy()
    rh = rh_t.update(rh_raw) if rh_flag else rh_t.last.copy()

    arms_raw = np.zeros((4, 3), dtype=np.float32)
    arms_flag = 0.0
    if pose_result and pose_result.pose_landmarks:
        p = pose_result.pose_landmarks[0]
        arms_raw = np.array([[p[11].x, p[11].y, p[11].z],
                             [p[12].x, p[12].y, p[12].z],
                             [p[13].x, p[13].y, p[13].z],
                             [p[14].x, p[14].y, p[14].z]], dtype=np.float32)
        arms_flag = 1.0
    arms = arms_t.update(arms_raw) if arms_flag else arms_t.last.copy()

    face_raw = np.zeros((7, 3), dtype=np.float32)
    face_flag = 0.0
    if face_result and face_result.face_landmarks:
        f = face_result.face_landmarks[0]
        face_raw = np.array([[f[i].x, f[i].y, f[i].z] for i in FACE_IDX], dtype=np.float32)
        face_flag = 1.0
    face = face_t.update(face_raw) if face_flag else face_t.last.copy()

    lh_out, rh_out, arms_out, face_out = _normalize_skeleton(
        lh, rh, arms, face, lh_flag, rh_flag, arms_flag, face_flag)

    return np.concatenate([
        lh_out.flatten(), rh_out.flatten(), arms_out.flatten(), face_out.flatten(),
        [lh_flag, rh_flag, arms_flag, face_flag]
    ]).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-video worker (subprocess)
# ---------------------------------------------------------------------------
_hand_lm = None
_pose_lm = None
_face_lm = None


def _init_worker():
    global _hand_lm, _pose_lm, _face_lm
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
            running_mode=VM,
            min_face_detection_confidence=0.1,
            min_face_presence_confidence=0.1,
            min_tracking_confidence=0.1,
            output_face_blendshapes=False))


def _ensure_worker():
    global _hand_lm
    if _hand_lm is None:
        _init_worker()


def extract_single_video(args: Tuple[str, str]) -> Tuple[str, bool, int]:
    _ensure_worker()
    vid_path_str, out_path_str = args
    vid_path = Path(vid_path_str)
    out_path = Path(out_path_str)

    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        return vid_path_str, False, 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps) or np.isinf(fps) or fps <= 0 or fps > 120:
        fps = 30.0

    lh_t, rh_t, arms_t, face_t = new_trackers()
    frames_data = []
    frame_index = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = frame_index

            f_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=f_rgb)

            try:
                hands = _hand_lm.detect_for_video(mp_image, timestamp_ms)
                pose = _pose_lm.detect_for_video(mp_image, timestamp_ms)
                face = _face_lm.detect_for_video(mp_image, timestamp_ms)
            except Exception:
                frames_data.append(np.zeros(163, dtype=np.float32))
                frame_index += 1
                continue

            frame_data = extract_stable_frame(hands, pose, face, lh_t, rh_t, arms_t, face_t)
            frames_data.append(frame_data)
            frame_index += 1
    finally:
        cap.release()

    if not frames_data:
        return vid_path_str, False, 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tensor = torch.from_numpy(np.stack(frames_data).astype(np.float32))
    torch.save(tensor, out_path)
    return vid_path_str, True, len(frames_data)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Bump underfed classes from iSign videos (9-core, resume-safe)")
    parser.add_argument("--isign-root", type=Path, default=ISIGN_EXTRACTED)
    parser.add_argument("--classes-json", type=Path, default=CLASSES_JSON)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES_DEFAULT,
                        help="Max videos per class (default: 22)")
    parser.add_argument("--min-clips", type=int, default=MIN_CLIPS_DEFAULT,
                        help="Min clips per class to qualify (default: 1)")
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS,
                        help="Number of parallel workers (default: 9)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip tensors that already exist (resume)")
    args = parser.parse_args()

    classes = load_classes(args.classes_json)
    classes_set = set(classes)
    print(f"Classes in classes.json: {len(classes)}")

    word_id_to_word = load_word_id_map(args.isign_root)
    print(f"Unique word_ids in iSign CSVs: {len(word_id_to_word)}")

    # Count already extracted per class
    already_extracted = count_extracted_per_class(args.output_dir, classes)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which classes need bumping
    needed: Dict[str, int] = {}
    for word in classes:
        have = already_extracted.get(word, 0)
        if have < args.max_samples:
            needed[word] = args.max_samples - have

    if not needed:
        print("\nAll classes already at max_samples. Nothing to bump.")
        return

    print(f"\nClasses needing bump ({len(needed)}):")
    for word in sorted(needed):
        print(f"  {word}: have {already_extracted.get(word, 0)}, need {needed[word]} more")

    # Scan iSign videos with progress
    print(f"\nScanning {args.isign_root / 'iSign-videos_v1.1'} for matching videos ...")
    word_to_videos: Dict[str, List[Path]] = {w: [] for w in needed}
    total_scanned = 0
    total_matched = 0
    last_report = 0

    for vf in (args.isign_root / "iSign-videos_v1.1").iterdir():
        if not vf.is_file():
            continue
        total_scanned += 1
        name = vf.stem
        word_id = name.split("--")[0] if "--" in name else name
        word = word_id_to_word.get(word_id)
        if word and word in needed:
            word_to_videos[word].append(vf)
            total_matched += 1

        # Progress report every 10,000 files
        if total_scanned - last_report >= 10000:
            print(f"  Scanned {total_scanned:,} files, matched {total_matched} so far ...")
            last_report = total_scanned

    print(f"\nScan complete: {total_scanned:,} files scanned, {total_matched} matched")

    # Build work list
    all_work: List[Tuple[str, str]] = []
    rng = random.Random(args.seed)

    for word in sorted(needed):
        videos = word_to_videos.get(word, [])
        if not videos:
            print(f"  [WARN] {word}: no iSign videos found, cannot bump")
            continue

        already_done = already_extracted.get(word, 0)
        needed_count = args.max_samples - already_done
        if len(videos) > needed_count:
            videos = rng.sample(videos, needed_count)

        print(f"  Bumping {word}: {len(videos)} videos (already have {already_done})")
        for i, vid_path in enumerate(videos):
            out_name = f"{word}_isign_{i+1:02d}.pt"
            out_path = args.output_dir / word / out_name
            if args.skip_existing and out_path.exists():
                continue
            all_work.append((str(vid_path), str(out_path)))

    print(f"\nTotal videos to extract: {len(all_work)}")

    if not all_work:
        print("Nothing to extract (all already exist or skip-existing filtered all).")
        return

    # Extract with progress
    total_extracted = 0
    total_failed = 0
    done = 0

    if args.jobs <= 1:
        for item in all_work:
            _, success, _ = extract_single_video(item)
            done += 1
            if done % 10 == 0 or done == len(all_work):
                print(f"  Progress: {done}/{len(all_work)} extracted, {total_extracted} ok, {total_failed} failed")
            total_extracted += int(success)
            total_failed += int(not success)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init_worker) as executor:
            futures = {executor.submit(extract_single_video, item): item for item in all_work}
            for future in as_completed(futures):
                item = futures[future]
                done += 1
                try:
                    _, success, _ = future.result()
                    if success:
                        total_extracted += 1
                    else:
                        total_failed += 1
                        print(f"  [WARN] Failed (no frames): {item[0]}")
                except Exception as exc:
                    total_failed += 1
                    print(f"  [WARN] Crashed: {item[0]} -> {exc}")

                if done % 10 == 0 or done == len(all_work):
                    print(f"  Progress: {done}/{len(all_work)} extracted, {total_extracted} ok, {total_failed} failed")

    print(f"\n=== Bump Extraction Complete ===")
    print(f"Total extracted: {total_extracted}")
    print(f"Total failed: {total_failed}")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
