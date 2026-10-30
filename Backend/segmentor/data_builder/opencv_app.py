"""
OpenCV desktop segmentation app — no Streamlit, real-time keyboard capture.

Reads sentences.csv and word_videos.csv.

Flow:
  1. Pick a sentence from sentences.csv
  2. Show word videos one-by-one (auto-play)
  3. User: SPACE = ready, R = rewatch word sequence
  4. Play sentence video in slow-mo; tap SPACE at each word boundary
  5. Each tap recorded to segmentation.csv
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ────────────────────────── paths ──────────────────────────
APP_DIR = Path(__file__).resolve().parent
SENTENCES_CSV = APP_DIR / "sentences.csv"
WORD_VIDEOS_CSV = APP_DIR / "word_videos.csv"
SEGMENTATION_CSV = APP_DIR / "segmentation.csv"
CHECKPOINT_JSON = APP_DIR / "checkpoint.json"
COMPOSED_DIR = Path(r"F:\dataset\composed_sentence_videos")

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".MP4", ".MOV"}
SLOWDOWN = 1.6
FULLSCREEN = True

# ────────────────────────── CSV schema ──────────────────────────
SEG_COLUMNS = [
    "sentence_hash", "sentence", "sentence_id", "sentence_video",
    "word", "word_video", "frame_index", "timestamp", "accepted", "marked_at",
]


def hash_sentence(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def sentence_words(sentence: str) -> List[str]:
    return re.findall(r"\b[\w'-]+\b", sentence.lower())


# ────────────────────────── load CSVs ──────────────────────────
def load_sentences() -> List[dict]:
    sentences = []
    if not SENTENCES_CSV.is_file():
        print(f"[ERROR] {SENTENCES_CSV} not found!")
        return sentences
    with open(SENTENCES_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sentences.append(row)
    print(f"[INFO] Loaded {len(sentences)} sentences")
    return sentences


def load_word_videos() -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    if not WORD_VIDEOS_CSV.is_file():
        print(f"[WARN] {WORD_VIDEOS_CSV} not found.")
        return grouped
    with open(WORD_VIDEOS_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            grouped.setdefault(row["sentence_id"], []).append(row)
    print(f"[INFO] Word videos for {len(grouped)} sentences")
    return grouped


# ────────────────────────── segmentation DB ──────────────────────────
class SegmentationDB:
    def __init__(self, path: Path = SEGMENTATION_CSV):
        self.path = path
        self._ensure()

    def _ensure(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.is_file():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=SEG_COLUMNS).writeheader()

    def append(self, sentence_hash, sentence, sentence_id, sentence_video,
               word, word_video, frame_index, timestamp, accepted):
        row = {
            "sentence_hash": sentence_hash,
            "sentence": sentence,
            "sentence_id": sentence_id,
            "sentence_video": sentence_video,
            "word": word,
            "word_video": word_video,
            "frame_index": int(frame_index),
            "timestamp": float(timestamp),
            "accepted": int(accepted),
            "marked_at": datetime.now().isoformat(),
        }
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=SEG_COLUMNS).writerow(row)

    def checkpoint(self, sentence_index: int, sentence_id: str, stage: str):
        data = {
            "sentence_index": int(sentence_index),
            "sentence_id": str(sentence_id),
            "stage": stage,
            "updated_at": datetime.now().isoformat(),
        }
        CHECKPOINT_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load_checkpoint(self) -> Optional[dict]:
        if not CHECKPOINT_JSON.is_file():
            return None
        try:
            return json.loads(CHECKPOINT_JSON.read_text(encoding="utf-8"))
        except Exception:
            return None


# ────────────────────────── compose sentence video ──────────────────────────
def ensure_sentence_video(sentence_id: str, sentence: str, word_videos: List[dict],
                          existing_video: str) -> str:
    if existing_video and Path(existing_video).is_file():
        return existing_video

    COMPOSED_DIR.mkdir(parents=True, exist_ok=True)
    composed_path = COMPOSED_DIR / f"{sentence_id}.mp4"

    if composed_path.is_file():
        return str(composed_path)

    if not word_videos:
        print(f"[COMPOSE] No word videos for: {sentence}")
        return ""

    print(f"[COMPOSE] Building sentence video: {sentence[:60]}...")
    print(f"[COMPOSE] Using {len(word_videos)} word clips")

    canvas_w = canvas_h = 0
    for wv in word_videos[:3]:  # check first 3 for size
        cap = cv2.VideoCapture(wv["word_video_path"])
        if cap.isOpened():
            canvas_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            canvas_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if canvas_w > 0 and canvas_h > 0:
                break
    if canvas_w == 0 or canvas_h == 0:
        canvas_w, canvas_h = 640, 480

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(composed_path), fourcc, 25.0, (canvas_w, canvas_h))

    try:
        for i, wv in enumerate(word_videos):
            cap = cv2.VideoCapture(wv["word_video_path"])
            if not cap.isOpened():
                print(f"[COMPOSE] Skip unreadable: {wv['word_video_path']}")
                continue
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame.shape[:2] != (canvas_h, canvas_w):
                    frame = cv2.resize(frame, (canvas_w, canvas_h))
                writer.write(frame)
            cap.release()
            if (i + 1) % 5 == 0:
                print(f"[COMPOSE] {i + 1}/{len(word_videos)} clips done")
    finally:
        writer.release()

    if composed_path.is_file():
        print(f"[COMPOSE] Done: {composed_path.name}")
        return str(composed_path)
    return ""


# ────────────────────────── OpenCV helpers ──────────────────────────
def play_video_fullscreen(path: str, window_name: str, slowdown: float = 1.0,
                          on_key=None) -> Optional[int]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    delay = max(1, int((1.0 / fps) * slowdown * 1000))

    if FULLSCREEN:
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

    last_key = None
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(delay) & 0xFF
        last_key = key
        frame_count += 1
        if on_key and not on_key(key):
            break
        # Safety: stop after very long videos (prevent infinite loop)
        if frame_count > 10000:
            print("[WARN] Video too long, stopping playback")
            break

    cap.release()
    return last_key


def draw_overlay(window_name: str, title: str, lines: List[str]):
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    y = 100
    cv2.putText(frame, title, (60, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    y += 80
    for line in lines:
        color = (255, 255, 255)
        if line.startswith("SPACE"):
            color = (0, 255, 0)
        elif line.startswith("R"):
            color = (0, 255, 255)
        elif line.startswith("Q"):
            color = (0, 0, 255)
        cv2.putText(frame, line, (60, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        y += 60
    if FULLSCREEN:
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
    cv2.imshow(window_name, frame)


def wait_for_key(window_name: str, valid_keys: Tuple[int, ...], timeout_ms: int = 0) -> Optional[int]:
    while True:
        key = cv2.waitKey(timeout_ms) & 0xFF
        if key == 255:
            if timeout_ms > 0:
                return None  # timeout
            continue
        if key in valid_keys:
            return key


# ────────────────────────── main app ──────────────────────────
class SegmentationApp:
    def __init__(self):
        self.sentences = load_sentences()
        self.word_videos = load_word_videos()
        self.db = SegmentationDB()
        self.window = "MookVani Segmentor"
        self.current_pos = 0
        self.stage = "select"

    def run(self):
        print(f"[INFO] {len(self.sentences)} sentences total")
        print(f"[INFO] Output: {self.db.path}")
        print("[INFO] Controls: UP/DOWN navigate, ENTER/SPACE select, SPACE mark, R redo, Q quit")

        ckpt = self.db.load_checkpoint()
        if ckpt:
            try:
                self.current_pos = int(ckpt.get("sentence_index", 0))
                self.stage = ckpt.get("stage", "words")
                print(f"[RESUME] sentence #{self.current_pos}, stage={self.stage}")
            except Exception:
                self.current_pos = 0
                self.stage = "select"

        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, 1280, 720)

        while True:
            if self.stage == "select":
                self._stage_select()
            elif self.stage == "words":
                self._stage_words()
            elif self.stage == "sentence":
                self._stage_sentence()
            elif self.stage == "done":
                self._stage_done()

    # ── select ──
    def _stage_select(self):
        while True:
            s = self.sentences[self.current_pos]
            draw_overlay(
                self.window,
                "SELECT SENTENCE",
                [
                    f"{self.current_pos + 1} / {len(self.sentences)}",
                    s["sentence"],
                    f"[{'real video' if s['sentence_video'] else 'composable'}]",
                    "",
                    "UP/DOWN = navigate",
                    "ENTER / SPACE = select",
                    "Q = quit",
                ],
            )
            key = cv2.waitKey(100) & 0xFF
            if key in (ord('q'), ord('Q')):
                cv2.destroyAllWindows()
                print("[INFO] Quit by user")
                sys.exit(0)
            elif key in (ord('j'), 82):
                self.current_pos = (self.current_pos + 1) % len(self.sentences)
            elif key in (ord('k'), 81):
                self.current_pos = (self.current_pos - 1) % len(self.sentences)
            elif key in (13, 32):
                print(f"[INFO] Selected: {s['sentence']}")
                self.stage = "words"
                self.db.checkpoint(self.current_pos, s["sentence_id"], "words")
                return

    # ── words ──
    def _stage_words(self):
        ent = self.sentences[self.current_pos]
        words = self.word_videos.get(ent["sentence_id"], [])

        print(f"\n=== WORD SEQUENCE: {ent['sentence']} ===")
        if words:
            for i, w in enumerate(words):
                print(f"  {i+1}. {w['word']} -> {Path(w['word_video_path']).name} ({w['source']})")
        else:
            print("  (no matching word videos)")

        if not words:
            draw_overlay(
                self.window,
                "NO WORD VIDEOS",
                [ent["sentence"], "", "SPACE = skip to next", "Q = quit"],
            )
            key = wait_for_key(self.window, (ord(' '), ord('q'), ord('Q'), 27))
            if key in (ord('q'), ord('Q'), 27) or key is None:
                cv2.destroyAllWindows()
                print("[INFO] Quit by user")
                sys.exit(0)
            self.current_pos = (self.current_pos + 1) % len(self.sentences)
            self.db.checkpoint(self.current_pos, self.sentences[self.current_pos]["sentence_id"], "words")
            return

        while True:
            for i, wv in enumerate(words):
                label = f"{i+1}/{len(words)}: {wv['word']}"
                print(f"[PLAY] {label}")
                last_key = play_video_fullscreen(
                    wv["word_video_path"], self.window, slowdown=1.0,
                    on_key=lambda k: self._word_video_key(k)
                )
                if last_key in (ord('r'), ord('R')):
                    break
                elif last_key == 27:
                    cv2.destroyAllWindows()
                    print("[INFO] Quit by user")
                    sys.exit(0)

            key = self._ask_ready_or_repeat(ent["sentence"])
            if key == ord(' '):
                self.stage = "sentence"
                self.db.checkpoint(self.current_pos, ent["sentence_id"], "sentence")
                return
            elif key in (ord('r'), ord('R')):
                continue
            elif key == 27 or key is None:
                cv2.destroyAllWindows()
                print("[INFO] Quit by user")
                sys.exit(0)

    def _word_video_key(self, key: int) -> bool:
        if key in (ord('r'), ord('R')):
            return False
        if key == 27:
            cv2.destroyAllWindows()
            print("[INFO] Quit by user")
            sys.exit(0)
        return True

    def _ask_ready_or_repeat(self, sentence_text: str) -> Optional[int]:
        draw_overlay(
            self.window,
            "WORD SEQUENCE COMPLETE",
            [
                f"Sentence: {sentence_text}",
                "",
                "SPACE = Ready -> sentence video",
                "R = Rewatch word sequence",
                "Q = Quit",
            ],
        )
        return wait_for_key(self.window, (ord(' '), ord('r'), ord('R'), ord('q'), ord('Q'), 27))

    # ── sentence ──
    def _stage_sentence(self):
        ent = self.sentences[self.current_pos]
        words = self.word_videos.get(ent["sentence_id"], [])

        print(f"\n=== SENTENCE VIDEO: {ent['sentence']} ===")

        sentence_video = ensure_sentence_video(
            ent["sentence_id"], ent["sentence"], words, ent["sentence_video"]
        )

        if not sentence_video:
            print(f"[WARN] No sentence video for: {ent['sentence']}")
            self._advance_next()
            return

        print(f"[PLAY] {sentence_video}")

        cap = cv2.VideoCapture(sentence_video)
        if not cap.isOpened():
            print(f"[WARN] Cannot open: {sentence_video}")
            self._advance_next()
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        delay = max(1, int((1.0 / fps) * SLOWDOWN * 1000))

        word_idx = 0

        if FULLSCREEN:
            cv2.namedWindow(self.window, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(self.window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window, 1280, 720)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            fidx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            ts = fidx / fps

            current_word = words[word_idx]["word"] if word_idx < len(words) else ""
            current_word_video = words[word_idx]["word_video_path"] if word_idx < len(words) else ""

            cv2.putText(frame, f"Sentence: {ent['sentence']}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
            cv2.putText(frame, f"Mark word: {current_word}", (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
            cv2.putText(frame, f"Frame {fidx}/{total}  {ts:.2f}s", (30, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "SPACE = mark boundary  N = next word  R = redo  Q = quit", (30, 660),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)

            cv2.imshow(self.window, frame)
            key = cv2.waitKey(delay) & 0xFF

            if key == ord(' '):
                if word_idx < len(words):
                    wv = words[word_idx]
                    self.db.append(
                        sentence_hash=hash_sentence(ent["sentence"]),
                        sentence=ent["sentence"],
                        sentence_id=ent["sentence_id"],
                        sentence_video=sentence_video,
                        word=wv["word"],
                        word_video=wv["word_video_path"],
                        frame_index=fidx,
                        timestamp=ts,
                        accepted=1,
                    )
                    print(f"[MARK] frame {fidx} ({ts:.2f}s) -> '{wv['word']}'")
                word_idx = min(word_idx + 1, len(words) - 1) if words else 0
            elif key in (ord('n'), ord('N')):
                word_idx = min(word_idx + 1, len(words) - 1) if words else 0
            elif key in (ord('r'), ord('R')):
                cap.release()
                self.stage = "words"
                self.db.checkpoint(self.current_pos, ent["sentence_id"], "words")
                return
            elif key in (ord('q'), ord('Q'), 27):
                cap.release()
                cv2.destroyAllWindows()
                print("[INFO] Quit by user")
                sys.exit(0)

            if words and word_idx >= len(words):
                break

        cap.release()
        self.stage = "done"
        self.db.checkpoint(self.current_pos, ent["sentence_id"], "done")
        print(f"[DONE] Sentence complete.")

    # ── done ──
    def _stage_done(self):
        ent = self.sentences[self.current_pos]
        draw_overlay(
            self.window,
            "SENTENCE COMPLETE",
            [
                ent["sentence"],
                "",
                "SPACE = Next sentence",
                "R = Redo this sentence",
                "Q = Quit",
            ],
        )
        key = wait_for_key(self.window, (ord(' '), ord('r'), ord('R'), ord('q'), ord('Q'), 27))
        if key == ord(' '):
            self.current_pos = (self.current_pos + 1) % len(self.sentences)
            self.stage = "words"
            self.db.checkpoint(self.current_pos, self.sentences[self.current_pos]["sentence_id"], "words")
        elif key in (ord('r'), ord('R')):
            self.stage = "words"
            self.db.checkpoint(self.current_pos, ent["sentence_id"], "words")
        elif key in (ord('q'), ord('Q'), 27) or key is None:
            cv2.destroyAllWindows()
            print("[INFO] Quit by user")
            sys.exit(0)

    def _advance_next(self):
        self.current_pos = (self.current_pos + 1) % len(self.sentences)
        self.stage = "words"
        self.db.checkpoint(self.current_pos, self.sentences[self.current_pos]["sentence_id"], "words")


def main():
    print("=" * 60)
    print("MookVani Sentence Segmentor — OpenCV Desktop App")
    print("=" * 60)
    app = SegmentationApp()
    app.run()


if __name__ == "__main__":
    main()
