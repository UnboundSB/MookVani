"""
Local text-to-speech worker
===========================

Runs TTS on a background thread so `/predict/*` requests never block waiting
for audio playback to finish.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from typing import Optional


SPEECH_ENABLED = os.environ.get("ISL_SPEECH_ENABLED", "1") == "1"
SPEECH_RATE = int(os.environ.get("ISL_SPEECH_RATE", "175"))
SPEECH_VOLUME = float(os.environ.get("ISL_SPEECH_VOLUME", "0.9"))
SPEECH_MODE = os.environ.get("ISL_SPEECH_MODE", "sentence")  # sentence|word


class SpeechEngine:
    def __init__(
        self,
        enabled: bool = SPEECH_ENABLED,
        rate: int = SPEECH_RATE,
        volume: float = SPEECH_VOLUME,
    ) -> None:
        self.enabled = bool(enabled)
        self.rate = rate
        self.volume = volume
        self.mode = SPEECH_MODE

        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop_sent = False

        self.last_spoken = ""
        self.last_spoken_at = 0.0
        self.queue_depth = 0
        self.error = ""

    def start(self) -> None:
        if not self.enabled:
            print("[SPEECH] Disabled by configuration.")
            return

        if self._thread and self._thread.is_alive():
            return

        self._stop_sent = False
        self._thread = threading.Thread(
            target=self._worker,
            name="ISLSpeechWorker",
            daemon=True,
        )
        self._thread.start()

    def speak(self, text: str) -> None:
        if not self.enabled or not text.strip():
            return

        self.queue_depth = self._queue.qsize() + 1
        self._queue.put(text.strip())

    def speak_word_or_sentence(self, word: str, sentence: str) -> None:
        text = sentence if self.mode == "sentence" and sentence else word
        self.speak(text)

    def stop(self) -> None:
        if not self._thread:
            return

        if not self._stop_sent:
            self._stop_sent = True
            self._queue.put(None)

    def _worker(self) -> None:
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)

            while True:
                text = self._queue.get()
                self.queue_depth = max(0, self._queue.qsize())

                if text is None:
                    break

                engine.say(text)
                engine.runAndWait()
                self.last_spoken = text
                self.last_spoken_at = time.time()

        except Exception as exc:
            self.enabled = False
            self.error = str(exc)
            print(f"[WARN] Speech disabled: {exc}")

    def state(self) -> dict:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "rate": self.rate,
            "volume": self.volume,
            "last_spoken": self.last_spoken,
            "last_spoken_at": self.last_spoken_at,
            "queue_depth": self.queue_depth,
            "error": self.error,
        }