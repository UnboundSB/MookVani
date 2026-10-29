"""
Segmentation data builder (minimal UI, sentence-by-sentence).

- Loads sentences in order from F:\dataset\isign\word-presence-dataset_v1.1.csv
- Matches each word in the sentence to word videos under
  F:\dataset\archive_extracted\{Category}\{N}. {word}\
- Autoplays word videos one by one in a large window
- Composes the sentence video by concatenating word clips
- Plays the sentence video in slow motion; SPACE records a segmentation frame
- Appends records to segmentation.csv (sentence_hash, sentence, sentence_video,
  word_video, word, frame_index, timestamp)
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import pandas as pd

DATASET_ROOT = Path(r"F:\dataset")
WORD_VIDEOS_ROOT = DATASET_ROOT / "archive_extracted"
SENTENCES_CSV = DATASET_ROOT / "isign" / "word-presence-dataset_v1.1.csv"

COMPOSED_VIDEOS_DIR = DATASET_ROOT / "composed_sentence_videos"
SEGMENTATION_CSV = Path(__file__).resolve().parent / "segmentation.csv"

COMPOSED_FPS = 25
SLOWDOWN_FACTOR = 1.6

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}

CSV_COLUMNS = [
    "sentence_hash",
    "sentence",
    "sentence_video",
    "word_video",
    "word",
    "frame_index",
    "timestamp",
]


def hash_sentence(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


@dataclass
class WordVideo:
    word: str
    path: Path
    category: str
    index: int


@dataclass
class SentenceEntry:
    sentence_id: str
    sentence: str
    words: List[str] = field(default_factory=list)
    matched_videos: List[WordVideo] = field(default_factory=list)
    composed_path: Optional[Path] = None
    hash: str = ""

    def compute_hash(self) -> str:
        self.hash = hash_sentence(self.sentence)
        return self.hash


def _word_from_folder(name: str):
    m = re.match(r"^(\d+)\.\s+(.+)$", name.strip())
    if m:
        return m.group(2).strip().lower(), int(m.group(1))
    return name.strip().lower(), 0


def scan_word_videos(root: Path = WORD_VIDEOS_ROOT) -> Dict[str, List[WordVideo]]:
    index: Dict[str, List[WordVideo]] = {}
    if not root.is_dir():
        return index
    for cat in sorted(root.iterdir()):
        if not cat.is_dir():
            continue
        for wd in sorted(cat.iterdir()):
            if not wd.is_dir():
                continue
            word, idx = _word_from_folder(wd.name)
            for vf in sorted(wd.iterdir()):
                if vf.suffix.lower() not in {e.lower() for e in VIDEO_EXTS}:
                    continue
                index.setdefault(word, []).append(
                    WordVideo(word=word, path=vf, category=cat.name, index=idx)
                )
    return index


def load_sentences_in_order(csv_path: Path = SENTENCES_CSV) -> List[SentenceEntry]:
    if not csv_path.is_file():
        return []
    df = pd.read_csv(csv_path)
    needed = {"word", "sentence_id", "sentence"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    grouped: Dict[str, SentenceEntry] = {}
    for _, row in df.iterrows():
        sid = str(row["sentence_id"]).strip()
        sentence = str(row["sentence"]).strip()
        word = str(row["word"]).strip()
        if sid not in grouped:
            ent = SentenceEntry(sentence_id=sid, sentence=sentence)
            ent.compute_hash()
            grouped[sid] = ent
        ent = grouped[sid]
        if word and word.lower() not in [w.lower() for w in ent.words]:
            ent.words.append(word)

    return sorted(grouped.values(), key=lambda e: e.sentence_id)


def match_words_to_videos(sentence: str, index: Dict[str, List[WordVideo]]) -> List[WordVideo]:
    tokens = re.findall(r"\b[\w'-]+\b", sentence.lower())
    matched: List[WordVideo] = []
    seen = set()
    for tok in tokens:
        if tok in seen:
            continue
        seen.add(tok)
        cands = index.get(tok)
        if not cands:
            continue
        matched.append(cands[0])
    return matched


def compose_sentence_video(word_videos: List[WordVideo], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not word_videos:
        raise ValueError("No word videos to compose")

    cw = ch = 0
    for wv in word_videos:
        cap = cv2.VideoCapture(str(wv.path))
        if cap.isOpened():
            cw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ch = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if cw > 0 and ch > 0:
                break
    if cw == 0 or ch == 0:
        cw, ch = 640, 480

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, COMPOSED_FPS, (cw, ch))
    try:
        for wv in word_videos:
            cap = cv2.VideoCapture(str(wv.path))
            if not cap.isOpened():
                continue
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame.shape[:2] != (ch, cw):
                    frame = cv2.resize(frame, (cw, ch))
                writer.write(frame)
            cap.release()
    finally:
        writer.release()
    return output_path


class SegmentationDB:
    def __init__(self, path: Path = SEGMENTATION_CSV):
        self.path = path
        self._ensure()

    def _ensure(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.is_file():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()

    def append(self, sentence_hash, sentence, sentence_video, word_video, word, frame_index, timestamp):
        row = {
            "sentence_hash": sentence_hash,
            "sentence": sentence,
            "sentence_video": sentence_video,
            "word_video": word_video,
            "word": word,
            "frame_index": int(frame_index),
            "timestamp": float(timestamp),
        }
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(row)

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.path)


# ---------------------------------------------------------------------------
# Checkpoint / auto-resume
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = Path(__file__).resolve().parent / "checkpoint.json"


def load_checkpoint() -> Optional[dict]:
    """Return saved checkpoint dict, or None if none exists."""
    if not CHECKPOINT_PATH.is_file():
        return None
    try:
        return json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_checkpoint(sentence_index: int, sentence_id: str, stage: str) -> None:
    """Persist progress so the app can resume after a stop / restart."""
    data = {
        "sentence_index": int(sentence_index),
        "sentence_id": str(sentence_id),
        "stage": stage,
        "updated_at": datetime.now().isoformat(),
    }
    CHECKPOINT_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def clear_checkpoint() -> None:
    """Remove checkpoint file (use when starting a fresh full pass)."""
    if CHECKPOINT_PATH.is_file():
        CHECKPOINT_PATH.unlink()
