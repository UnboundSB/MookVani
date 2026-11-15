"""
Multi-Core Sentence-Level Video Landmark Extractor
==================================================

Extracts every decoded frame from each sentence-level video using the same
MediaPipe Holistic landmark pipeline as the word- and sentence-frame data.

Source structure:
    F:\\datasets\\mookvani_dataset\\ISL_CSLRT_Corpus\\ISL_CSLRT_Corpus\\
        Videos_Sentence_Level\\
            <sentence class>\\
                <take>.mp4

Output structure:
    F:\\datasets\\mookvani_dataset\\ISL_CSLRT_Corpus\\ISL_CSLRT_Corpus\\
        extracted\\
            videos_sentence_level\\
                <sentence class>\\
                    <take>.pt

Each .pt file contains one video:

    {
        'landmarks': (T, 522) float32 tensor,
        'label': str,
        'label_id': int,
        'sentence': str,
        'take': str,
        'take_index': int,
        'source_video': str,
        'num_frames': int,
        'dims': 522,
        'fps': float,
        'source_fps': float,
        'width': int,
        'height': int,
        'frame_indices': list[int],
        'sharpen_attempts': list[int],
    }

The corpus sentence videos are 25 FPS. Every decoded frame is retained in
chronological order. No frame dropping, padding, or temporal resampling is
performed, so tensors remain variable length.
"""

import re
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Make the Backend directory importable when this file is run directly.
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from utils.landmark_extractor import (  # noqa: E402
    N_DIMS,
    extract_landmarks_from_frame,
    get_holistic_landmarker,
)


# ──────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────

DATASET_ROOT = Path(
    r"F:\datasets\mookvani_dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus"
)
SOURCE_VIDEOS_ROOT = DATASET_ROOT / "Videos_Sentence_Level"
OUTPUT_ROOT = DATASET_ROOT / "extracted" / "videos_sentence_level"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
SENTENCE_FPS = 25.0
N_WORKERS = max(1, cpu_count() - 1)

NUMBER_PATTERN = re.compile(r"(\d+)")


# ──────────────────────────────────────────────────────────────────
# TASK AND RESULT TYPES
# ──────────────────────────────────────────────────────────────────

class ExtractionTask(NamedTuple):
    label: str
    label_id: int
    take: str
    take_index: int
    video_path: str


class ExtractionResult(NamedTuple):
    label: str
    take: str
    success: bool
    num_frames: int
    error: Optional[str]


# ──────────────────────────────────────────────────────────────────
# VIDEO DISCOVERY
# ──────────────────────────────────────────────────────────────────

def natural_sort_key(path: Path) -> Tuple[Tuple[int, object], ...]:
    """Sort filenames naturally, so take 2 precedes take 10."""
    parts = NUMBER_PATTERN.split(path.name)
    key: List[Tuple[int, object]] = []
    for part in parts:
        if part.isdigit():
            key.append((1, int(part)))
        else:
            key.append((0, part.casefold()))
    return tuple(key)


def build_task_list() -> List[ExtractionTask]:
    """Discover every sentence class and video without splitting the data."""
    if not SOURCE_VIDEOS_ROOT.is_dir():
        raise FileNotFoundError(
            f"Sentence video source not found: {SOURCE_VIDEOS_ROOT}"
        )

    class_dirs = sorted(
        (path for path in SOURCE_VIDEOS_ROOT.iterdir() if path.is_dir()),
        key=lambda path: (path.name.casefold(), path.name),
    )

    tasks: List[ExtractionTask] = []
    for label_id, class_dir in enumerate(class_dirs):
        videos = sorted(
            (
                path
                for path in class_dir.iterdir()
                if path.is_file()
                and path.suffix.casefold() in VIDEO_EXTENSIONS
            ),
            key=natural_sort_key,
        )

        for take_index, video_path in enumerate(videos, start=1):
            tasks.append(
                ExtractionTask(
                    label=class_dir.name,
                    label_id=label_id,
                    take=video_path.stem,
                    take_index=take_index,
                    video_path=str(video_path),
                )
            )

    return tasks


# ──────────────────────────────────────────────────────────────────
# WORKER EXTRACTION
# ──────────────────────────────────────────────────────────────────

_worker_landmarker = None


def init_worker() -> None:
    """Initialize one MediaPipe Holistic model per worker process."""
    global _worker_landmarker
    _worker_landmarker = get_holistic_landmarker()


def extract_video(task: ExtractionTask) -> ExtractionResult:
    """Extract all frames from one sentence video."""
    capture = None
    try:
        capture = cv2.VideoCapture(task.video_path)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {task.video_path}")

        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(source_fps) or source_fps <= 0:
            raise RuntimeError(f"Invalid FPS for video: {task.video_path}")
        if not np.isclose(source_fps, SENTENCE_FPS, atol=0.01):
            raise RuntimeError(
                f"Unexpected FPS for {task.video_path}: {source_fps}; "
                f"expected {SENTENCE_FPS}"
            )

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        landmarks: List[np.ndarray] = []
        sharpen_attempts: List[int] = []
        frame_indices: List[int] = []
        frame_index = 0

        while True:
            success, frame = capture.read()
            if not success:
                break

            frame_landmarks, _, attempts = extract_landmarks_from_frame(
                frame,
                holistic_lm=_worker_landmarker,
                auto_sharpen=True,
            )
            if frame_landmarks.shape != (N_DIMS,):
                raise RuntimeError(
                    f"Unexpected landmark shape {frame_landmarks.shape} "
                    f"at frame {frame_index}; expected ({N_DIMS},)"
                )

            landmarks.append(frame_landmarks)
            sharpen_attempts.append(attempts)
            frame_indices.append(frame_index)
            frame_index += 1

        capture.release()
        capture = None

        if not landmarks:
            raise RuntimeError(f"No frames decoded from video: {task.video_path}")

        landmark_array = np.stack(landmarks, axis=0).astype(np.float32, copy=False)
        output_dir = OUTPUT_ROOT / task.label
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{task.take}.pt"

        torch.save(
            {
                "landmarks": torch.from_numpy(landmark_array),
                "label": task.label,
                "label_id": task.label_id,
                "sentence": task.label,
                "take": task.take,
                "take_index": task.take_index,
                "source_video": Path(task.video_path).name,
                "num_frames": int(landmark_array.shape[0]),
                "dims": N_DIMS,
                "fps": source_fps,
                "source_fps": source_fps,
                "width": width,
                "height": height,
                "frame_indices": frame_indices,
                "sharpen_attempts": sharpen_attempts,
            },
            output_path,
        )

        return ExtractionResult(
            label=task.label,
            take=task.take,
            success=True,
            num_frames=int(landmark_array.shape[0]),
            error=None,
        )
    except Exception as exc:
        if capture is not None:
            capture.release()
        return ExtractionResult(
            label=task.label,
            take=task.take,
            success=False,
            num_frames=0,
            error=f"{type(exc).__name__}: {exc}",
        )


# ──────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────

def extract_sentence_videos() -> None:
    """Extract all sentence videos in parallel without splitting."""
    tasks = build_task_list()
    if not tasks:
        raise RuntimeError(f"No sentence videos found under {SOURCE_VIDEOS_ROOT}")

    print(f"Sentence classes: {len({task.label for task in tasks})}")
    print(f"Videos to extract: {len(tasks)}")
    print(f"Output FPS: {SENTENCE_FPS}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Using {N_WORKERS} workers...")

    with Pool(processes=N_WORKERS, initializer=init_worker) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(extract_video, tasks, chunksize=1),
                total=len(tasks),
                desc="Extracting sentence videos",
            )
        )

    successful = [result for result in results if result.success]
    failed = [result for result in results if not result.success]

    print(f"\n{'=' * 60}")
    print(f"Extraction complete: {len(successful)}/{len(results)} videos succeeded")
    print(f"Frames extracted: {sum(result.num_frames for result in successful)}")

    if failed:
        print(f"Failed videos ({len(failed)}):")
        for result in failed:
            print(f"  {result.label}/{result.take}: {result.error}")
        print("=" * 60)
        raise RuntimeError(f"Sentence video extraction failed for {len(failed)} video(s)")

    print("=" * 60)


if __name__ == "__main__":
    extract_sentence_videos()
