"""
Backend/scripts/build_isign_dataset.py  (single-file, self-contained)

Full iSign word-level tensor dataset builder.  Extracts ALL classes listed in
classes.json that exist in F:\dataset\archive_extracted (with >= min_clips
clips each), capped at --max-samples videos per class.

Coordinate stability (per-coordinate carry-forward):
  Every landmark coordinate (x, y, z) is tracked independently.  When a frame
  reads exactly 0.0 (MediaPipe dropout / coord reset to zero), the previous
  non-zero value is retained.  This prevents coords from bouncing between a
  real value and 0 at infinite speed ("down then up" jitter).

Extracted regions (MediaPipe), nothing below the belt:
  * Hands full      -> 21 landmarks x 3  (left + right = 126 dims)
  * Arms above waist-> shoulders + elbows = 4 landmarks x 3  (12 dims)
  * Face/head       -> L-eye, R-eye, nose, L-cheek, R-cheek, L-mouth, R-mouth
                       = 7 landmarks x 3  (21 dims)
  * Flags           -> [lh, rh, arms, face]  (4 dims)
  Total = 163 dims per frame, saved as (T, 163) .pt tensors.

Usage (example - extract ALL classes, up to 40 videos each, 8 workers, resume):
    .\venv\Scripts\python.exe Backend\scripts\build_isign_dataset.py ^
        --max-classes 0 --max-samples 40 --min-clips 14 --jobs 8 --skip-existing

--max-classes 0  = NO class cap (extract every qualifying class).
--max-samples    = per-class video cap (raise this to feed underfed classes).
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BACKEND))

ISIGN_ARCHIVE_EXTRACTED = Path(r"F:\dataset\archive_extracted")
CLASSES_JSON = BACKEND / "models" / "word_vocabulary" / "classes.json"
OUTPUT_DIR = BACKEND / "data" / "tensors_include50_word_level"

MODEL_DIR = Path(r"D:\mookvani")
HAND_MODEL = MODEL_DIR / "hand_landmarker.task"
POSE_MODEL = MODEL_DIR / "pose_landmarker.task"
FACE_MODEL = MODEL_DIR / "face_landmarker.task"

MAX_CLASSES_DEFAULT = 0          # 0 = no cap (all qualifying classes)
MAX_SAMPLES_DEFAULT = 22         # per-class video cap (raise to feed underfed)
MIN_CLIPS_DEFAULT = 14
DEFAULT_JOBS = 4


# ===========================================================================
#  Stable per-coordinate tracker (carry-forward on zero-reset)
# ===========================================================================
class CoordTracker:
    """Holds last known good value for every coordinate independently."""

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
            CoordTracker(7, 3))    # face (eyes, nose, cheeks, mouth corners)


# MediaPipe landmark indices
FACE_IDX = [159, 386, 4, 234, 454, 61, 291]   # L-eye, R-eye, nose, L-cheek, R-cheek, L-mouth, R-mouth
POSE_ARMS_IDX = [11, 12, 13, 14]              # L-sh, R-sh, L-el, R-el


def _normalize_skeleton(lh, rh, arms, face_pts, lh_flag, rh_flag, arms_flag, face_flag):
    anchor = np.array([0.5, 0.5, 0.0])
    if face_flag:
        anchor = face_pts[2]                  # nose
    elif arms_flag:
        anchor = (arms[0] + arms[1]) / 2.0    # neck
    elif lh_flag:
        anchor = lh[0]                        # left wrist
    elif rh_flag:
        anchor = rh[0]                        # right wrist

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
    """One 163-dim frame with per-coordinate carry-forward stability."""
    # --- Hands ---
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

    # --- Arms (shoulders + elbows) ---
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

    # --- Face / head (7 pts) ---
    face_raw = np.zeros((7, 3), dtype=np.float32)
    face_flag = 0.0
    if face_result and face_result.face_landmarks:
        f = face_result.face_landmarks[0]
        face_raw = np.array([[f[i].x, f[i].y, f[i].z] for i in FACE_IDX], dtype=np.float32)
        face_flag = 1.0
    face = face_t.update(face_raw) if face_flag else face_t.last.copy()

    # --- Normalize ---
    lh_out, rh_out, arms_out, face_out = _normalize_skeleton(
        lh, rh, arms, face, lh_flag, rh_flag, arms_flag, face_flag)

    return np.concatenate([
        lh_out.flatten(), rh_out.flatten(), arms_out.flatten(), face_out.flatten(),
        [lh_flag, rh_flag, arms_flag, face_flag]
    ]).astype(np.float32)   # (163,)


# ===========================================================================
#  Discovery / selection
# ===========================================================================
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
                if p.is_file() and p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv"))
            if cn and videos:
                word_map.setdefault(cn, []).extend(videos)
    return word_map


def select_classes(word_map, classes_json, max_classes=0, max_samples=22,
                   seed=42, min_clips=14) -> Dict[str, List[Path]]:
    classes_set = set(classes_json)
    eligible = {w: v for w, v in word_map.items()
                if w in classes_set and len(v) >= min_clips}
    print(f"Classes in classes.json found in archive with >= {min_clips} clips: {len(eligible)}")

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


# ===========================================================================
#  Per-video worker (subprocess)
# ===========================================================================
_hand_lm = None
_pose_lm = None
_face_lm = None
_fps_default = 30.0


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
        fps = _fps_default

    lh_t, rh_t, arms_t, face_t = new_trackers()
    frames_data = []
    frame_index = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # MediaPipe VIDEO mode requires monotonically increasing timestamps.
            # Use frame_index directly — guaranteed to increase by exactly 1 each frame.
            timestamp_ms = frame_index

            f_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=f_rgb)

            try:
                hands = _hand_lm.detect_for_video(mp_image, timestamp_ms)
                pose = _pose_lm.detect_for_video(mp_image, timestamp_ms)
                face = _face_lm.detect_for_video(mp_image, timestamp_ms)
            except Exception:
                # If one frame's detection fails, still record a zero frame to
                # keep the tensor shape consistent rather than aborting the video.
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
    tensor = torch.from_numpy(np.stack(frames_data).astype(np.float32))  # (T, 163)
    torch.save(tensor, out_path)
    return vid_path_str, True, len(frames_data)


# ===========================================================================
#  Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Build stable iSign word-level tensor dataset (single file)")
    parser.add_argument("--archive-root", type=Path, default=ISIGN_ARCHIVE_EXTRACTED)
    parser.add_argument("--classes-json", type=Path, default=CLASSES_JSON)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-classes", type=int, default=MAX_CLASSES_DEFAULT,
                        help="Max classes (0 = no cap = extract ALL qualifying classes)")
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES_DEFAULT,
                        help="Max videos per class (raise this to feed underfed classes)")
    parser.add_argument("--min-clips", type=int, default=MIN_CLIPS_DEFAULT,
                        help="Min clips per class to qualify")
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS,
                        help="Number of parallel workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip tensors that already exist (resume)")
    args = parser.parse_args()

    classes = load_classes(args.classes_json)
    print(f"Classes in classes.json: {len(classes)}")

    print(f"\nScanning {args.archive_root} ...")
    word_map = discover_archive_words(args.archive_root)
    print(f"Total word folders found: {len(word_map)}")

    selected = select_classes(word_map, classes,
                              max_classes=args.max_classes,
                              max_samples=args.max_samples,
                              seed=args.seed,
                              min_clips=args.min_clips)
    print(f"\nSelected {len(selected)} classes (cap = {args.max_samples} videos/class):")

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
                item = futures[future]
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

    print(f"\n=== Extraction Complete ===")
    print(f"Classes: {len(selected)}")
    print(f"Total extracted: {total_extracted}")
    print(f"Total failed: {total_failed}")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
