"""
Multi-Core Sentence-Level Frame Landmark Extractor
==================================================

Extracts ordered sentence-level frame sequences with the same MediaPipe
Holistic pipeline used at word level.

Source structure:
    F:\\datasets\\mookvani_dataset\\ISL_CSLRT_Corpus\\ISL_CSLRT_Corpus\\
        Frames_Sentence_Level\\
            <sentence class>\\
                <take>\\
                    <ordered frame>.jpg

Output structure:
    F:\\datasets\\mookvani_dataset\\ISL_CSLRT_Corpus\\ISL_CSLRT_Corpus\\
        extracted\\
            frames_sentence_level\\
                <sentence class>\\
                    <take>.pt

Each .pt file contains one complete take:

    {
        'landmarks': (T, 522) float32 tensor,
        'label': str,
        'label_id': int,
        'sentence': str,
        'sentence_folder': str,
        'take': str,
        'take_index': int | None,
        'num_frames': int,
        'dims': 522,
        'frame_files': list[str],
        'sharpen_attempts': list[int],
    }

Frames are sorted naturally by their numeric suffix and kept in their exact
sequence. The output is not padded, truncated, resampled, or split into
train/validation directories.
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
SOURCE_FRAMES_ROOT = DATASET_ROOT / "Frames_Sentence_Level"
OUTPUT_ROOT = DATASET_ROOT / "extracted" / "frames_sentence_level"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
N_WORKERS = max(1, cpu_count() - 1)

NUMBER_PATTERN = re.compile(r"(\d+)")


# ──────────────────────────────────────────────────────────────────
# TASK AND RESULT TYPES
# ──────────────────────────────────────────────────────────────────

class ExtractionTask(NamedTuple):
    sentence_folder: str
    sentence: str
    label: str
    label_id: int
    take: str
    frame_files: Tuple[str, ...]


class ExtractionResult(NamedTuple):
    sentence_folder: str
    take: str
    success: bool
    num_frames: int
    error: Optional[str]


# ──────────────────────────────────────────────────────────────────
# FRAME DISCOVERY
# ──────────────────────────────────────────────────────────────────

def natural_sort_key(path: Path) -> Tuple[Tuple[int, object], ...]:
    """Sort filenames naturally, so frame 2 precedes frame 10."""
    parts = NUMBER_PATTERN.split(path.name)
    key: List[Tuple[int, object]] = []
    for part in parts:
        if part.isdigit():
            key.append((1, int(part)))
        else:
            key.append((0, part.casefold()))
    return tuple(key)


def collect_frame_files(take_dir: Path) -> List[Path]:
    """Collect image frames from one take directory in sequence order."""
    if not take_dir.is_dir():
        raise FileNotFoundError(f"Take directory not found: {take_dir}")

    frames = [
        path
        for path in take_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    ]
    return sorted(frames, key=natural_sort_key)


def build_task_list() -> List[ExtractionTask]:
    """Discover every sentence class and take without splitting the data."""
    if not SOURCE_FRAMES_ROOT.is_dir():
        raise FileNotFoundError(
            f"Sentence frame source not found: {SOURCE_FRAMES_ROOT}"
        )

    sentence_dirs = sorted(
        (path for path in SOURCE_FRAMES_ROOT.iterdir() if path.is_dir()),
        key=lambda path: (path.name.casefold(), path.name),
    )

    tasks: List[ExtractionTask] = []

    for label_id, sentence_dir in enumerate(sentence_dirs):
        label = sentence_dir.name
        take_dirs = sorted(
            (path for path in sentence_dir.iterdir() if path.is_dir()),
            key=lambda path: (path.name.casefold(), path.name),
        )

        if not take_dirs:
            print(f"Warning: no take directories for sentence '{sentence_dir.name}'")
            continue

        for take_dir in take_dirs:
            frame_files = collect_frame_files(take_dir)
            tasks.append(
                ExtractionTask(
                    sentence_folder=sentence_dir.name,
                    sentence=label,
                    label=label,
                    label_id=label_id,
                    take=take_dir.name,
                    frame_files=tuple(str(path) for path in frame_files),
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


def extract_take(task: ExtractionTask) -> ExtractionResult:
    """Extract and save one ordered sentence take."""
    try:
        if not task.frame_files:
            raise ValueError(f"No image frames found for take '{task.take}'")

        landmarks = np.empty((len(task.frame_files), N_DIMS), dtype=np.float32)
        sharpen_attempts: List[int] = []

        for index, frame_file in enumerate(task.frame_files):
            frame = cv2.imread(frame_file, cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Could not read frame: {frame_file}")

            frame_landmarks, _, attempts = extract_landmarks_from_frame(
                frame,
                holistic_lm=_worker_landmarker,
                auto_sharpen=True,
            )
            if frame_landmarks.shape != (N_DIMS,):
                raise RuntimeError(
                    f"Unexpected landmark shape {frame_landmarks.shape} "
                    f"for frame {frame_file}; expected ({N_DIMS},)"
                )

            landmarks[index] = frame_landmarks
            sharpen_attempts.append(attempts)

        take_index = int(task.take) if task.take.isdigit() else None
        output_dir = OUTPUT_ROOT / task.sentence_folder
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{task.take}.pt"

        torch.save(
            {
                "landmarks": torch.from_numpy(landmarks),
                "label": task.label,
                "label_id": task.label_id,
                "sentence": task.sentence,
                "sentence_folder": task.sentence_folder,
                "take": task.take,
                "take_index": take_index,
                "num_frames": int(landmarks.shape[0]),
                "dims": N_DIMS,
                "frame_files": [Path(path).name for path in task.frame_files],
                "sharpen_attempts": sharpen_attempts,
            },
            output_path,
        )

        return ExtractionResult(
            sentence_folder=task.sentence_folder,
            take=task.take,
            success=True,
            num_frames=int(landmarks.shape[0]),
            error=None,
        )
    except Exception as exc:
        return ExtractionResult(
            sentence_folder=task.sentence_folder,
            take=task.take,
            success=False,
            num_frames=0,
            error=f"{type(exc).__name__}: {exc}",
        )


# ──────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────

def extract_sentence_level() -> None:
    """Extract all sentence-level frame takes in parallel."""
    tasks = build_task_list()
    if not tasks:
        raise RuntimeError(f"No sentence takes found under {SOURCE_FRAMES_ROOT}")

    total_frames = sum(len(task.frame_files) for task in tasks)
    sentence_count = len({task.sentence_folder for task in tasks})

    print(f"Sentence classes: {sentence_count}")
    print(f"Takes to extract: {len(tasks)}")
    print(f"Total frames: {total_frames}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Using {N_WORKERS} workers...")

    with Pool(processes=N_WORKERS, initializer=init_worker) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(extract_take, tasks, chunksize=1),
                total=len(tasks),
                desc="Extracting sentence frames",
            )
        )

    successful = [result for result in results if result.success]
    failed = [result for result in results if not result.success]

    print(f"\n{'=' * 60}")
    print(f"Extraction complete: {len(successful)}/{len(results)} takes succeeded")
    print(f"Frames extracted: {sum(result.num_frames for result in successful)}")

    if failed:
        print(f"Failed takes ({len(failed)}):")
        for result in failed:
            print(
                f"  {result.sentence_folder}/{result.take}: "
                f"{result.error}"
            )
        print("=" * 60)
        raise RuntimeError(f"Sentence extraction failed for {len(failed)} take(s)")

    print("=" * 60)


if __name__ == "__main__":
    extract_sentence_level()
