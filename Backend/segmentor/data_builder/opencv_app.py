"""
MookVani Sentence Segmentor — OpenCV Desktop App
=================================================

Reads sentences.csv and word_videos.csv. For each sentence:
  1. Shows matched word videos individually (no blank frames, no stitching)
  2. User presses SPACE when ready
  3. Plays the FULL sentence recording:
     - ISL_CSLRT: plays the real sentence video file
     - ISIGN: plays matched word videos sequentially (each as its own clip)
  4. During full playback, SPACE marks segmentation boundary → segmentation.csv

Controls:
  SPACE = mark segmentation boundary at current word
  N     = skip to next word boundary
  R     = redo this sentence
  Q     = quit
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
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

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".MP4", ".MOV"}
SLOWDOWN = 1.6
FULLSCREEN = False

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

    def reject_sentence(self, sentence_id: str):
        """Remove all segmentation rows for a given sentence_id (reject/discard)."""
        if not self.path.is_file():
            return
        with open(self.path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = [row for row in reader if row.get("sentence_id") != sentence_id]
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

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


# ────────────────────────── OpenCV helpers ──────────────────────────
def play_video_fullscreen(path: str, window_name: str, slowdown: float = 1.0,
                          on_key=None) -> Optional[int]:
    """Play a single video file. Returns last key pressed."""
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
                return None
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
        self.marks_in_current_sentence = 0

    def run(self):
        print(f"[INFO] {len(self.sentences)} sentences total")
        print(f"[INFO] Output: {self.db.path}")
        print("[INFO] Controls: UP/DOWN navigate, ENTER/SPACE select, SPACE mark boundary, R redo, Q quit")

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
                    f"[{'real video' if s['sentence_video'] else 'word videos'}]",
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
            elif key in (ord('j'), 82):  # down
                self.current_pos = (self.current_pos + 1) % len(self.sentences)
            elif key in (ord('k'), 81):  # up
                self.current_pos = (self.current_pos - 1) % len(self.sentences)
            elif key in (13, 32):  # ENTER or SPACE
                print(f"[INFO] Selected: {s['sentence']}")
                self.stage = "words"
                self.db.checkpoint(self.current_pos, s["sentence_id"], "words")
                return

    # ── words ──
    def _stage_words(self):
        ent = self.sentences[self.current_pos]
        sentence = ent["sentence"]
        all_words = sentence_words(sentence)

        word_entries = self.word_videos.get(ent["sentence_id"], [])
        video_map = {}
        for wv in word_entries:
            video_map[wv["word"]] = wv["word_video_path"]

        # Only include words that HAVE videos — skip missing words entirely (no blank frames)
        matched_entries = []
        for w in all_words:
            if w in video_map:
                matched_entries.append({
                    "word": w,
                    "word_video_path": video_map[w],
                })

        if not matched_entries:
            print(f"[SKIP] No word videos for sentence: {sentence}")
            self.current_pos = (self.current_pos + 1) % len(self.sentences)
            self.db.checkpoint(self.current_pos, self.sentences[self.current_pos]["sentence_id"], "words")
            return

        print(f"\n=== WORD SEQUENCE (matched words only): {ent['sentence']} ===")
        for i, e in enumerate(matched_entries):
            print(f"  {i+1}. {e['word']} -> {e['word_video_path']}")

        while True:
            for i, entry in enumerate(matched_entries):
                label = f"{i+1}/{len(matched_entries)}: {entry['word']}"
                print(f"[PLAY] {label} (video)")
                last_key = play_video_fullscreen(
                    entry["word_video_path"], self.window, slowdown=1.0,
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
                "SPACE = Ready -> full sentence playback",
                "R = Rewatch word sequence",
                "Q = Quit",
            ],
        )
        return wait_for_key(self.window, (ord(' '), ord('r'), ord('R'), ord('q'), ord('Q'), 27))

    # ── sentence ──
    def _stage_sentence(self):
        ent = self.sentences[self.current_pos]
        self.marks_in_current_sentence = 0
        sentence = ent["sentence"]
        all_words = sentence_words(sentence)

        word_entries = self.word_videos.get(ent["sentence_id"], [])
        video_map = {}
        for wv in word_entries:
            video_map[wv["word"]] = wv["word_video_path"]

        # Only include words that HAVE videos
        entries = []
        for w in all_words:
            if w in video_map:
                entries.append({
                    "word": w,
                    "word_video_path": video_map[w],
                })

        if not entries:
            print(f"[SKIP] No word videos for sentence: {sentence}")
            self.current_pos = (self.current_pos + 1) % len(self.sentences)
            self.db.checkpoint(self.current_pos, self.sentences[self.current_pos]["sentence_id"], "words")
            return

        print(f"\n=== FULL SENTENCE PLAYBACK: {ent['sentence']} ===")
        for i, e in enumerate(entries):
            print(f"  {i+1}. {e['word']} -> {e['word_video_path']}")

        is_composed = ent.get("is_composed", "1") == "1"

        if not is_composed:
            # ── ISL_CSLRT: gather all video takes for this sentence ──
            sentence_id = ent["sentence_id"]
            takes = [
                s for s in self.sentences
                if s["sentence_id"] == sentence_id and s.get("sentence_video", "").strip()
                and os.path.isfile(s.get("sentence_video", "").strip())
            ]
            if not takes:
                print(f"[WARN] No real sentence video found: {sentence}")
                self._advance_next()
                return
            # Find current take index
            current_take_idx = 0
            for i, t in enumerate(takes):
                if t["sentence_video"] == ent["sentence_video"]:
                    current_take_idx = i
                    break
            print(f"[PLAY] Real sentence video (take {current_take_idx + 1}/{len(takes)}): {takes[current_take_idx]['sentence_video']}")
            self._play_full_with_marking(ent, entries, takes, current_take_idx, is_real=True)
        else:
            # ── ISIGN: play matched word videos sequentially (NOT stitched) ──
            print(f"[PLAY] Word videos sequentially (no stitching)")
            self._play_words_sequential_with_marking(ent, entries)

    def _play_full_with_marking(self, ent: dict, entries: list, takes: list,
                                take_idx: int, is_real: bool):
        """Play a full video (real sentence video) with SPACE boundary marking.
        Supports switching takes with T key."""
        take = takes[take_idx]
        video_path = take["sentence_video"]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[WARN] Cannot open video: {video_path}")
            self._advance_next()
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        delay = max(1, int((1.0 / fps) * SLOWDOWN * 1000))

        # Build word→frame map (divide duration across matched words)
        word_frame_map = []
        n_words = len(entries)
        if n_words > 0:
            frames_per_word = total // n_words
            current_frame = 0
            for entry in entries:
                word_frame_map.append((
                    entry["word"],
                    current_frame,
                    current_frame + frames_per_word,
                    entry["word_video_path"],
                ))
                current_frame += frames_per_word

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

            current_word = ""
            current_word_video = ""
            for w, start_f, end_f, vpath in word_frame_map:
                if start_f <= fidx < end_f:
                    current_word = w
                    current_word_video = vpath if vpath else ""
                    break

            cv2.putText(frame, f"Sentence: {ent['sentence']}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
            cv2.putText(frame, f"Mark word: {current_word}", (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
            cv2.putText(frame, f"Take {take_idx + 1}/{len(takes)}  Frame {fidx}/{total}  {ts:.2f}s", (30, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "SPACE = mark boundary  S = skip this word  T = next take  N = next word  R = redo  Q = quit", (30, 660),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)

            cv2.imshow(self.window, frame)
            key = cv2.waitKey(delay) & 0xFF

            if key == ord(' '):
                self.db.append(
                    sentence_hash=hash_sentence(ent["sentence"]),
                    sentence=ent["sentence"],
                    sentence_id=ent["sentence_id"],
                    sentence_video=video_path,
                    word=current_word,
                    word_video=current_word_video,
                    frame_index=fidx,
                    timestamp=ts,
                    accepted=1,
                )
                self.marks_in_current_sentence += 1
                print(f"[MARK] frame {fidx} ({ts:.2f}s) -> '{current_word}'")
            elif key in (ord('s'), ord('S')):
                # Skip this word boundary — jump to next word's start frame
                print(f"[SKIP] frame {fidx} ({ts:.2f}s) -> '{current_word}' not marked")
                # Find current word's index and jump to next word's start frame
                current_word_idx = None
                for i, (w, sf, ef, vp) in enumerate(word_frame_map):
                    if w == current_word:
                        current_word_idx = i
                        break
                if current_word_idx is not None and current_word_idx + 1 < len(word_frame_map):
                    next_start_frame = word_frame_map[current_word_idx + 1][1]
                    cap.set(cv2.CAP_PROP_POS_FRAMES, next_start_frame)
                    print(f"[SKIP] Jumped to next word: {word_frame_map[current_word_idx + 1][0]} at frame {next_start_frame}")
            elif key == ord('t') or key == ord('T'):
                # Next take for same sentence
                cap.release()
                next_idx = (take_idx + 1) % len(takes)
                print(f"[NEXT TAKE] Switching to take {next_idx + 1}/{len(takes)}")
                self._play_full_with_marking(ent, entries, takes, next_idx, is_real=True)
                return
            elif key in (ord('n'), ord('N')):
                pass
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

        cap.release()
        if self.marks_in_current_sentence == 0:
            print(f"[DONE] No words marked — auto-advancing to next sentence.")
            self._advance_next()
            return
        self.stage = "done"
        self.db.checkpoint(self.current_pos, ent["sentence_id"], "done")
        print(f"[DONE] Sentence complete.")

    def _play_words_sequential_with_marking(self, ent: dict, entries: list):
        """Play matched word videos sequentially (NOT stitched) with SPACE boundary marking.
        For ISIGN sentences that have no real sentence video."""
        window = self.window

        if FULLSCREEN:
            cv2.namedWindow(window, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, 1280, 720)

        # Track cumulative frame index across all word videos
        global_frame_idx = 0
        global_timestamp = 0.0

        for entry_idx, entry in enumerate(entries):
            word = entry["word"]
            word_video_path = entry["word_video_path"]

            cap = cv2.VideoCapture(word_video_path)
            if not cap.isOpened():
                print(f"[WARN] Cannot open word video: {word_video_path}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 25.0
            delay = max(1, int((1.0 / fps) * SLOWDOWN * 1000))
            word_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            word_frame_idx = 0
            skipped = False
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                current_ts = global_timestamp + (word_frame_idx / fps)

                cv2.putText(frame, f"Sentence: {ent['sentence']}", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
                cv2.putText(frame, f"Mark word: {word}", (30, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
                cv2.putText(frame, f"Word {entry_idx + 1}/{len(entries)}  Frame {word_frame_idx}/{word_total}  {current_ts:.2f}s", (30, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, "SPACE = mark boundary  S = skip this word  N = next word  R = redo  Q = quit", (30, 660),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)

                cv2.imshow(window, frame)
                key = cv2.waitKey(delay) & 0xFF

                if key == ord(' '):
                    self.db.append(
                        sentence_hash=hash_sentence(ent["sentence"]),
                        sentence=ent["sentence"],
                        sentence_id=ent["sentence_id"],
                        sentence_video=word_video_path,
                        word=word,
                        word_video=word_video_path,
                        frame_index=global_frame_idx,
                        timestamp=current_ts,
                        accepted=1,
                    )
                    self.marks_in_current_sentence += 1
                    print(f"[MARK] frame {global_frame_idx} ({current_ts:.2f}s) -> '{word}'")
                elif key in (ord('s'), ord('S')):
                    # Skip this word — break out of current word video loop, move to next word
                    print(f"[SKIP] frame {global_frame_idx} ({current_ts:.2f}s) -> '{word}' not marked")
                    skipped = True
                    break
                elif key in (ord('n'), ord('N')):
                    pass
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

                word_frame_idx += 1
                global_frame_idx += 1

            cap.release()
            # Only add actual frames played to timestamp (if skipped early, don't add full duration)
            frames_played = word_frame_idx if skipped else word_total
            global_timestamp += frames_played / fps

        if self.marks_in_current_sentence == 0:
            print(f"[DONE] No words marked — auto-advancing to next sentence.")
            self._advance_next()
            return
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
                "D = Reject this sentence (discard marks)",
                "R = Redo this sentence",
                "Q = Quit",
            ],
        )
        key = wait_for_key(self.window, (ord(' '), ord('d'), ord('D'), ord('r'), ord('R'), ord('q'), ord('Q'), 27))
        if key == ord(' '):
            self.current_pos = (self.current_pos + 1) % len(self.sentences)
            self.stage = "words"
            self.db.checkpoint(self.current_pos, self.sentences[self.current_pos]["sentence_id"], "words")
        elif key in (ord('d'), ord('D')):
            self.db.reject_sentence(ent["sentence_id"])
            print(f"[REJECT] Discarded marks for: {ent['sentence']}")
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
