"""
ISL session state
=================

Caches predicted characters while a hand is visible, waits for a no-hand gap,
then converts the cached letters into a corrected phrase using SymSpell and
sends the phrase to the local speech worker.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

from .spelling import SpellCorrector
from .speech import SpeechEngine


IDLE_SECONDS = float(os.environ.get("ISL_FINALIZE_IDLE_SECONDS", "1.0"))


class ISLSession:
    def __init__(
        self,
        engine,
        spelling: Optional[SpellCorrector] = None,
        speech: Optional[SpeechEngine] = None,
        idle_seconds: float = IDLE_SECONDS,
    ) -> None:
        self.engine = engine
        self.spelling = spelling or SpellCorrector()
        self.speech = speech
        self.idle_seconds = max(0.2, float(idle_seconds))

        self.letters: List[str] = []
        self.sentence_parts: List[str] = []
        self.last_corrected_word = ""
        self.last_spoken_text = ""
        self.last_hand_time = time.time()
        self.last_letter_append_time = 0.0

    # ------------------------------------------------------------------
    # State operations
    # ------------------------------------------------------------------
    def _append_letter(self, label: str) -> None:
        if not label or label == "-":
            return

        if self.letters and self.letters[-1] == label:
            return

        self.letters.append(label)
        self.last_letter_append_time = time.time()
        print(f"[SESSION] letter cached: {label} | buffer={''.join(self.letters)}")

    def _reset_engine_votes(self) -> None:
        try:
            self.engine.window_votes.clear()
            self.engine.window_confs.clear()
            self.engine.current_label = "-"
            self.engine.current_confidence = 0.0
            self.engine.last_update_t = time.time()
        except Exception:
            pass

    def finalize(self, force: bool = False) -> Dict:
        """Convert cached letters into corrected speech input."""
        if not self.letters:
            return {"finalized": False}

        raw = "".join(self.letters)
        corrected = self.spelling.correct(raw) if self.spelling else raw.upper()

        self.letters.clear()
        self.last_corrected_word = corrected

        if corrected:
            self.sentence_parts.append(corrected)
        else:
            return {"finalized": False, "raw": raw}

        sentence = " ".join(self.sentence_parts)
        self.last_spoken_text = sentence

        # Frontend React TTS handles audible speech.

        return {
            "finalized": True,
            "raw": raw,
            "corrected_word": corrected,
            "sentence": sentence,
            "speech_text": self.last_spoken_text,
            "parts_count": len(self.sentence_parts),
        }

    def process(self, result: Dict) -> Dict:
        """Update cache and finalize when hands disappear for the idle window."""
        now = time.time()
        hands_count = int(result.get("hands_count", 0) or 0)

        if hands_count > 0:
            self.last_hand_time = now

            if result.get("prediction_updated"):
                label = str(result.get("prediction", "-"))
                self._append_letter(label)

        finalized = None
        if now - self.last_hand_time >= self.idle_seconds:
            finalized = self.finalize()

        return {
            "letters": list(self.letters),
            "word": "".join(self.letters),
            "sentence": " ".join(self.sentence_parts),
            "corrected_word": self.last_corrected_word,
            "hands_count": hands_count,
            "idle_seconds": self.idle_seconds,
            "time_since_hand": round(now - self.last_hand_time, 3),
            "finalized": finalized or {"finalized": False},
        }

    def flush_spoken(self, count: Optional[int] = None) -> Dict:
        """
        Flush only the words already spoken.

        Any words finalized while speech was still playing are preserved.
        """
        if count is None:
            count = len(self.sentence_parts)

        safe_count = max(0, min(int(count), len(self.sentence_parts)))

        if safe_count:
            del self.sentence_parts[:safe_count]

        if not self.sentence_parts:
            self.last_corrected_word = ""
            self.last_spoken_text = ""
        else:
            self.last_spoken_text = " ".join(self.sentence_parts)

        return self.snapshot()

    def reset(self) -> Dict:
        self.letters.clear()
        self.sentence_parts.clear()
        self.last_corrected_word = ""
        self.last_spoken_text = ""
        self.last_hand_time = time.time()
        self.last_letter_append_time = 0.0
        self._reset_engine_votes()

        return self.snapshot()

    def snapshot(self) -> Dict:
        return {
            "letters": list(self.letters),
            "word": "".join(self.letters),
            "sentence": " ".join(self.sentence_parts),
            "corrected_word": self.last_corrected_word,
            "last_spoken_text": self.last_spoken_text,
            "time_since_hand": round(time.time() - self.last_hand_time, 3),
            "idle_seconds": self.idle_seconds,
        }