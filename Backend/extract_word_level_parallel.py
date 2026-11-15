"""
Multi-Core Word-Level Frame Landmark Extractor
==============================================

Extracts word-level frame sequences with MediaPipe Holistic. Classes are
derived directly from the source directory names. The extracted dataset is
stored as one tensor per word class without a train/validation split.

Source structure:
    F:\\datasets\\mookvani_dataset\\ISL_CSLRT_Corpus\\ISL_CSLRT_Corpus\\
        Frames_Word_Level\\
            <word class>\\
                <ordered frame>.jpg

Output structure:
    F:\\datasets\\mookvani_dataset\\ISL_CSLRT_Corpus\\ISL_CSLRT_Corpus\\
        extracted\\
            frames_word_level\\
                <word class>\\
                    <word class>.pt

Each .pt file contains:

    {
        'landmarks': (120, 522) float32 tensor,
        'label': str,
        'label_id': int,
        'word': str,
        'num_frames': int,
        'source_num_frames': int,
        'dims': 522,
        'frame_files': list[str],
        'sharpen_attempts': list[int],
    }
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
SOURCE_FRAMES_ROOT = DATASET_ROOT / "Frames_Word_Level"
OUTPUT_ROOT = DATASET_ROOT / "extracted" / "frames_word_level"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_FRAMES = 120
N_WORKERS = max(1, cpu_count() - 1)

NUMBER_PATTERN = re.compile(r"(\d+)")


# ──────────────────────────────────────────────────────────────────
# TASK AND RESULT TYPES
# ──────────────────────────────────────────────────────────────────

class ExtractionTask(NamedTuple):
    label: str
    label_id: int
    source_dir: str
    frame_files: Tuple[str, ...]


class ExtractionResult(NamedTuple):
    label: str
    success: bool
    source_num_frames: int
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


def collect_frame_files(class_dir: Path) -> List[Path]:
    """Collect one word class's image frames in sequence order."""
    if not class_dir.is_dir():
        raise FileNotFoundError(f"Word class directory not found: {class_dir}")

    frames = [
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    ]
    return sorted(frames, key=natural_sort_key)


def build_task_list() -> List[ExtractionTask]:
    """Discover every word class directly from the source directories."""
    if not SOURCE_FRAMES_ROOT.is_dir():
        raise FileNotFoundError(
            f"Word-level frame source not found: {SOURCE_FRAMES_ROOT}"
        )

    class_dirs = sorted(
        (path for path in SOURCE_FRAMES_ROOT.iterdir() if path.is_dir()),
        key=lambda path: (path.name.casefold(), path.name),
    )

    tasks: List[ExtractionTask] = []
    for label_id, class_dir in enumerate(class_dirs):
        frame_files = collect_frame_files(class_dir)
        tasks.append(
            ExtractionTask(
                label=class_dir.name,
                label_id=label_id,
                source_dir=str(class_dir),
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


def extract_word_class(task: ExtractionTask) -> ExtractionResult:
    """Extract and save one word class."""
    try:
        if not task.frame_files:
            raise ValueError(f"No image frames found for class '{task.label}'")

        selected_frames = list(task.frame_files)
        if len(selected_frames) > MAX_FRAMES:
            indices = np.linspace(
                0,
                len(selected_frames) - 1,
                MAX_FRAMES,
                dtype=int,
            )
            selected_frames = [selected_frames[index] for index in indices]

        landmarks = np.zeros((MAX_FRAMES, N_DIMS), dtype=np.float32)
        sharpen_attempts: List[int] = []

        for index, frame_file in enumerate(selected_frames):
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

        output_dir = OUTPUT_ROOT / task.label
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{task.label}.pt"

        torch.save(
            {
                "landmarks": torch.from_numpy(landmarks),
                "label": task.label,
                "label_id": task.label_id,
                "word": task.label,
                "num_frames": int(landmarks.shape[0]),
                "source_num_frames": len(task.frame_files),
                "dims": N_DIMS,
                "frame_files": [Path(path).name for path in selected_frames],
                "sharpen_attempts": sharpen_attempts,
            },
            output_path,
        )

        return ExtractionResult(
            label=task.label,
            success=True,
            source_num_frames=len(task.frame_files),
            error=None,
        )
    except Exception as exc:
        return ExtractionResult(
            label=task.label,
            success=False,
            source_num_frames=len(task.frame_files),
            error=f"{type(exc).__name__}: {exc}",
        )


# ──────────────────────────────────────────────────────────────────
# VISUALIZATION
# ──────────────────────────────────────────────────────────────────

def visualize_sample_frames() -> None:
    """Display MediaPipe landmarks for a few sample word frames."""
    sample_words = ["ABUSE", "GOOD", "HELLO_HI", "THANK", "WELCOME"]

    for word in sample_words:
        class_dir = SOURCE_FRAMES_ROOT / word
        if not class_dir.exists():
            print(f"Skipping {word}: class directory not found")
            continue

        frame_files = collect_frame_files(class_dir)
        if not frame_files:
            continue

        frame_path = frame_files[0]
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            print(f"Skipping {word}: could not read {frame_path}")
            continue

        holistic_lm = get_holistic_landmarker()
        _, result, attempts = extract_landmarks_from_frame(
            frame,
            holistic_lm=holistic_lm,
            auto_sharpen=True,
        )

        from utils.landmark_extractor import draw_landmarks_on_frame

        visualization = draw_landmarks_on_frame(frame, result)
        cv2.putText(
            visualization,
            f"Word: {word}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            visualization,
            f"Sharpen attempts: {attempts}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

        cv2.imshow(f"Landmarks: {word}", cv2.resize(visualization, (960, 540)))
        cv2.waitKey(0)

    cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────

def extract_word_level() -> None:
    """Extract all word-level classes in parallel without splitting."""
    tasks = build_task_list()
    if not tasks:
        raise RuntimeError(f"No word classes found under {SOURCE_FRAMES_ROOT}")

    source_frames = sum(len(task.frame_files) for task in tasks)

    print(f"Word classes: {len(tasks)}")
    print(f"Source frames: {source_frames}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Using {N_WORKERS} workers...")

    with Pool(processes=N_WORKERS, initializer=init_worker) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(extract_word_class, tasks, chunksize=1),
                total=len(tasks),
                desc="Extracting word frames",
            )
        )

    successful = [result for result in results if result.success]
    failed = [result for result in results if not result.success]

    print(f"\n{'=' * 60}")
    print(f"Extraction complete: {len(successful)}/{len(results)} classes succeeded")
    print(f"Source frames processed: {sum(result.source_num_frames for result in successful)}")

    if failed:
        print(f"Failed classes ({len(failed)}):")
        for result in failed:
            print(f"  {result.label}: {result.error}")
        print("=" * 60)
        raise RuntimeError(f"Word extraction failed for {len(failed)} class(es)")

    print("=" * 60)


if __name__ == "__main__":
    extract_word_level()
