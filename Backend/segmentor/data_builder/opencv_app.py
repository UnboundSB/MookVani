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

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

APP_DIR = Path(__file__).resolve().parent
SENTENCES_CSV = APP_DIR / "sentences.csv"
WORD_VIDEOS_CSV = APP_DIR / "word_videos.csv"
SEGMENTATION_CSV = APP_DIR / "segmentation.csv"
CHECKPOINT_JSON = APP_DIR / "checkpoint.json"

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".MP4", ".MOV"}
SLOWDOWN = 1.6
FULLSCREEN = False

SEG_COLUMNS = [
    "sentence_hash", "sentence", "sentence_id", "sentence_video",
    "word", "word_video", "frame_index", "timestamp", "accepted", "marked_at",
]


def hash_sentence(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def sentence_words(sentence: str) -> List[str]:
    return re.findall(r"\b[\w'-]+\b", sentence.lower())


def load_sentences() -> List[dict]:
    sentences = []
    if not SENTENCES_CSV.is_file():
        print(f"[ERROR] {SENTENCES_CSV} not found!")
        return sentences
    with open(SENTENCES_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sentences.append(row)
    print(f"[INFO] Loaded {len(sentences)} sentences", flush=True)
    return sentences


def load_word_videos() -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    if not WORD_VIDEOS_CSV.is_file():
        print(f"[WARN] {WORD_VIDEOS_CSV} not found.")
        return grouped
    with open(WORD_VIDEOS_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            grouped.setdefault(row["sentence_id"], []).append(row)
    print(f"[INFO] Word videos for {len(grouped)} sentences", flush=True)
    return grouped


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


def draw_overlay(window_name: str, title: str, lines: List[str]):
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    y = 100
    cv2.putText(frame, title, (60, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    y += 80
    for line in lines:
        color = (255, 255, 255)
        if line.startswith("SPACE") or line.startswith("M"):
            color = (0, 255, 0)
        elif line.startswith("R"):
            color = (0, 255, 255)
        elif line.startswith("D"):
            color = (0, 0, 255)
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


class SegmentationApp:
    def __init__(self):
        self.sentences = load_sentences()
        self.word_videos = load_word_videos()
        self.db = SegmentationDB()
        self.window = "MookVani Segmentor"
        self.current_pos = 0
        self.stage = "select"
        self.marks_in_current_sentence = 0
        self.last_m_time = 0.0  # for M-key debounce

    def run(self):
        print(f"[INFO] {len(self.sentences)} sentences total")
        print(f"[INFO] Output: {self.db.path}")
        print("[INFO] Controls: ENTER/SPACE select, R=preview words, M=mark boundary, R=redo, D=discard, Q=quit")

        ckpt = self.db.load_checkpoint()
        if ckpt:
            try:
                self.current_pos = int(ckpt.get("sentence_index", 0))
                self.stage = ckpt.get("stage", "select")
                print(f"[RESUME] sentence #{self.current_pos}, stage={self.stage}")
            except Exception:
                self.current_pos = 0
                self.stage = "select"

        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, 1280, 720)

        while True:
            if self.stage == "select":
                self._stage_select()
            elif self.stage == "preview_words":
                self._stage_preview_words()
            elif self.stage == "sentence":
                self._stage_sentence()
            elif self.stage == "done":
                self._stage_done()

    def _stage_select(self):
        s = self.sentences[self.current_pos]
        draw_overlay(
            self.window,
            "SELECT SENTENCE",
            [
                f"{self.current_pos + 1} / {len(self.sentences)}",
                s["sentence"],
                f"[{'real video' if s['sentence_video'] else 'word videos'}]",
                "",
                "ENTER / SPACE = play sentence",
                "R = preview word videos from archive",
                "Q = quit",
            ],
        )
        key = cv2.waitKey(100) & 0xFF
        if key in (ord('q'), ord('Q')):
            cv2.destroyAllWindows()
            print("[INFO] Quit by user")
            sys.exit(0)
        elif key in (13, 32):
            print(f"[INFO] Playing: {s['sentence']}")
            self.marks_in_current_sentence = 0
            self.stage = "sentence"
            self.db.checkpoint(self.current_pos, s["sentence_id"], "sentence")
        elif key in (ord('r'), ord('R')):
            self.stage = "preview_words"
            self.db.checkpoint(self.current_pos, s["sentence_id"], "preview_words")

    def _stage_preview_words(self):
        """Preview word videos from archive/ISIGN before committing to sentence."""
        ent = self.sentences[self.current_pos]
        sentence = ent["sentence"]
        all_words = sentence_words(sentence)

        word_entries = self.word_videos.get(ent["sentence_id"], [])
        video_map = {}
        for wv in word_entries:
            video_map[wv["word"]] = wv["word_video_path"]

        matched_entries = []
        seen = set()
        for w in all_words:
            if w in video_map and w not in seen:
                matched_entries.append({
                    "word": w,
                    "word_video_path": video_map[w],
                })
                seen.add(w)

        if not matched_entries:
            print(f"[SKIP] No word videos for sentence: {sentence}")
            self.stage = "select"
            self.db.checkpoint(self.current_pos, ent["sentence_id"], "select")
            return

        print(f"\n=== WORD VIDEO PREVIEW: {ent['sentence']} ===")
        for i, e in enumerate(matched_entries):
            print(f"  {i+1}. {e['word']} -> {e['word_video_path']}")

        # Play each word video sequentially
        for entry in matched_entries:
            word = entry["word"]
            word_video_path = entry["word_video_path"]

            cap = cv2.VideoCapture(word_video_path)
            if not cap.isOpened():
                print(f"[WARN] Cannot open: {word_video_path}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 25.0
            frame_interval = (1.0 / fps) * SLOWDOWN
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            while cap.isOpened():
                key = cv2.waitKey(1) & 0xFF
                ret, frame = cap.read()
                if not ret:
                    break

                cv2.putText(frame, f"PREVIEW: {sentence}", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
                cv2.putText(frame, f"Frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}/{total}",
                            (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, "SPACE = next word  Q = back to selection",
                            (30, 660), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
                cv2.imshow(self.window, frame)

                if key in (ord(' '), 32):
                    break  # next word
                elif key in (ord('q'), ord('Q'), 27):
                    cap.release()
                    cv2.destroyAllWindows()
                    print("[INFO] Quit by user")
                    sys.exit(0)

                time.sleep(frame_interval)

            cap.release()

        # After preview, go back to selection
        draw_overlay(
            self.window,
            "PREVIEW COMPLETE",
            [
                f"Sentence: {sentence}",
                f"Words previewed: {len(matched_entries)}",
                "",
                "SPACE = play this sentence",
                "R = preview again",
                "Q = back to selection",
            ],
        )
        key = wait_for_key(self.window, (ord(' '), ord('r'), ord('R'), ord('q'), ord('Q'), 27))
        if key == ord(' '):
            self.marks_in_current_sentence = 0
            self.stage = "sentence"
            self.db.checkpoint(self.current_pos, ent["sentence_id"], "sentence")
        elif key in (ord('r'), ord('R')):
            self.stage = "preview_words"
            self.db.checkpoint(self.current_pos, ent["sentence_id"], "preview_words")
        else:
            self.stage = "select"
            self.db.checkpoint(self.current_pos, ent["sentence_id"], "select")

    def _stage_sentence(self):
        ent = self.sentences[self.current_pos]
        sentence = ent["sentence"]
        all_words = sentence_words(sentence)

        word_entries = self.word_videos.get(ent["sentence_id"], [])
        video_map = {}
        for wv in word_entries:
            video_map[wv["word"]] = wv["word_video_path"]

        entries = []
        seen = set()
        for w in all_words:
            if w in video_map and w not in seen:
                entries.append({
                    "word": w,
                    "word_video_path": video_map[w],
                })
                seen.add(w)

        if not entries:
            print(f"[SKIP] No word videos for sentence: {sentence}")
            self.current_pos = (self.current_pos + 1) % len(self.sentences)
            self.db.checkpoint(self.current_pos, self.sentences[self.current_pos]["sentence_id"], "words")
            return

        print(f"\n=== FULL SENTENCE PLAYBACK: {ent['sentence']} ===")
        for i, e in enumerate(entries):
            print(f"  {i+1}. {e['word']} -> {e['word_video_path']}")

        is_composed = ent.get("is_composed", "1") == "1"
        real_video_path = ent.get("sentence_video", "").strip()

        if not is_composed and real_video_path and os.path.isfile(real_video_path):
            print(f"[PLAY] Real sentence video: {real_video_path}")
            self._play_full_video(ent, entries, real_video_path)
        else:
            print(f"[PLAY] Word videos sequentially (no stitching)")
            self._play_words_sequential(ent, entries)

    def _play_full_video(self, ent: dict, entries: list, video_path: str):
        """Play a single full video file.
        M = mark boundary at current frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[WARN] Cannot open video: {video_path}")
            self._advance_next()
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = (1.0 / fps) * SLOWDOWN

        while cap.isOpened():
            key = cv2.waitKey(1) & 0xFF

            ret, frame = cap.read()
            if not ret:
                break
            fidx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            ts = fidx / fps

            if key in (ord('m'), ord('M')):
                now = time.time()
                if now - self.last_m_time < 0.5:
                    continue
                self.last_m_time = now
                self.db.append(
                    sentence_hash=hash_sentence(ent["sentence"]),
                    sentence=ent["sentence"],
                    sentence_id=ent["sentence_id"],
                    sentence_video=video_path,
                    word="",
                    word_video="",
                    frame_index=fidx,
                    timestamp=ts,
                    accepted=1,
                )
                self.marks_in_current_sentence += 1
                print(f"[MARK] frame {fidx} ({ts:.2f}s) boundary")
                continue

            elif key in (ord('r'), ord('R')):
                cap.release()
                self.stage = "select"
                self.db.checkpoint(self.current_pos, ent["sentence_id"], "select")
                return
            elif key in (ord('d'), ord('D')):
                cap.release()
                self.db.reject_sentence(ent["sentence_id"])
                print(f"[REJECT] Discarded marks for: {ent['sentence']}")
                self.current_pos = (self.current_pos + 1) % len(self.sentences)
                self.db.checkpoint(self.current_pos, self.sentences[self.current_pos]["sentence_id"], "select")
                return
            elif key in (ord('q'), ord('Q'), 27):
                cap.release()
                cv2.destroyAllWindows()
                print("[INFO] Quit by user")
                sys.exit(0)

            # Minimal overlay - just frame/time
            cv2.putText(frame, f"Sentence: {ent['sentence']}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame {fidx}/{total_frames}  {ts:.2f}s", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Marks: {self.marks_in_current_sentence}", (30, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, "M = mark boundary  R = redo  D = discard  Q = quit", (30, 680),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)

            cv2.imshow(self.window, frame)

            time.sleep(frame_interval)

        cap.release()
        if self.marks_in_current_sentence == 0:
            print(f"[DONE] No boundaries marked — auto-advancing to next sentence.")
            self._advance_next()
            return
        print(f"[DONE] Sentence complete. Advancing to next.")
        self._advance_next()

    def _play_words_sequential(self, ent: dict, entries: list):
        """Play matched word videos sequentially (NOT stitched).
        M = mark boundary at current frame
        """
        global_frame_idx = 0
        global_timestamp = 0.0

        try:
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
                frame_interval = (1.0 / fps) * SLOWDOWN
                word_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                word_frame_idx = 0

                while cap.isOpened():
                    key = cv2.waitKey(1) & 0xFF

                    ret, frame = cap.read()
                    if not ret:
                        break
                    word_frame_idx += 1
                    global_frame_idx += 1
                    current_ts = global_timestamp + (word_frame_idx / fps)

                    if key in (ord('m'), ord('M')):
                        now = time.time()
                        if now - self.last_m_time < 0.5:
                            continue
                        self.last_m_time = now
                        self.db.append(
                            sentence_hash=hash_sentence(ent["sentence"]),
                            sentence=ent["sentence"],
                            sentence_id=ent["sentence_id"],
                            sentence_video=word_video_path,
                            word="",
                            word_video="",
                            frame_index=global_frame_idx,
                            timestamp=current_ts,
                            accepted=1,
                        )
                        self.marks_in_current_sentence += 1
                        print(f"[MARK] frame {global_frame_idx} ({current_ts:.2f}s) boundary")
                        continue

                    elif key in (ord('r'), ord('R')):
                        cap.release()
                        self.stage = "select"
                        self.db.checkpoint(self.current_pos, ent["sentence_id"], "select")
                        return
                    elif key in (ord('d'), ord('D')):
                        cap.release()
                        self.db.reject_sentence(ent["sentence_id"])
                        print(f"[REJECT] Discarded marks for: {ent['sentence']}")
                        self.current_pos = (self.current_pos + 1) % len(self.sentences)
                        self.db.checkpoint(self.current_pos, self.sentences[self.current_pos]["sentence_id"], "select")
                        return
                    elif key in (ord('q'), ord('Q'), 27):
                        cap.release()
                        cv2.destroyAllWindows()
                        print("[INFO] Quit by user")
                        sys.exit(0)

                    # Minimal overlay
                    cv2.putText(frame, f"Sentence: {ent['sentence']}", (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.putText(frame, f"Word clip: {entry_idx + 1}/{len(entries)}  Frame {word_frame_idx}/{word_total}  {current_ts:.2f}s", (30, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Marks: {self.marks_in_current_sentence}", (30, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(frame, "M = mark boundary  R = redo  D = discard  Q = quit", (30, 680),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)

                    cv2.imshow(self.window, frame)

                    time.sleep(frame_interval)

                cap.release()
                global_timestamp += word_total / fps

            if self.marks_in_current_sentence == 0:
                print(f"[DONE] No boundaries marked — auto-advancing to next sentence.")
                self._advance_next()
                return
            print(f"[DONE] Sentence complete. Advancing to next.")
            self._advance_next()

        except Exception as e:
            print(f"[ERROR] {e}")
            self._advance_next()

    def _stage_done(self):
        """Show completion screen for current sentence."""
        ent = self.sentences[self.current_pos]
        draw_overlay(
            self.window,
            "SENTENCE COMPLETE",
            [
                ent["sentence"],
                "",
                "SPACE = Next sentence",
                "R = Redo this sentence",
                "D = Discard this sentence",
                "Q = Quit",
            ],
        )
        key = wait_for_key(self.window, (ord(' '), ord('r'), ord('R'), ord('d'), ord('D'), ord('q'), ord('Q'), 27))
        if key == ord(' '):
            self.current_pos = (self.current_pos + 1) % len(self.sentences)
            self.stage = "words"
            self.db.checkpoint(self.current_pos, self.sentences[self.current_pos]["sentence_id"], "words")
        elif key in (ord('r'), ord('R')):
            self.stage = "words"
            self.db.checkpoint(self.current_pos, ent["sentence_id"], "words")
        elif key in (ord('d'), ord('D')):
            self.db.reject_sentence(ent["sentence_id"])
            print(f"[REJECT] Discarded marks for: {ent['sentence']}")
            self.current_pos = (self.current_pos + 1) % len(self.sentences)
            self.stage = "words"
            self.db.checkpoint(self.current_pos, self.sentences[self.current_pos]["sentence_id"], "words")
        elif key in (ord('q'), ord('Q'), 27) or key is None:
            cv2.destroyAllWindows()
            print("[INFO] Quit by user")
            sys.exit(0)

    def _advance_next(self):
        self.current_pos = (self.current_pos + 1) % len(self.sentences)
        self.stage = "select"
        self.db.checkpoint(self.current_pos, self.sentences[self.current_pos]["sentence_id"], "select")


def main():
    print("=" * 60)
    print("MookVani Sentence Segmentor — OpenCV Desktop App")
    print("=" * 60)
    app = SegmentationApp()
    app.run()


if __name__ == "__main__":
    main()
