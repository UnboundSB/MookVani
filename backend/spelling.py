"""
SymSpell post-processor
=======================

Breaks a cached run of ISL letters into words and applies lightweight spelling
correction before the result is spoken.
"""

from __future__ import annotations

import os
import re
import urllib.request
from pathlib import Path
from typing import Optional


DEFAULT_DICTIONARY_URL = (
    "https://raw.githubusercontent.com/wolfgarbe/SymSpell/master/SymSpell/"
    "frequency_dictionary_en_82_765.txt"
)
DEFAULT_DICTIONARY_NAME = "frequency_dictionary_en_82_765.txt"


class SpellCorrector:
    """Thread-safe symspell wrapper with local dictionary caching."""

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        dictionary_url: str = DEFAULT_DICTIONARY_URL,
        auto_download: bool = True,
    ) -> None:
        self.data_dir = data_dir or Path(__file__).resolve().parent / "data"
        self.dictionary_path = self.data_dir / DEFAULT_DICTIONARY_NAME
        self.dictionary_url = dictionary_url
        self.auto_download = auto_download

        self.enabled = False
        self.load_error = ""
        self._sym_spell = None

        self._load()

    def _load(self) -> None:
        try:
            from symspellpy import SymSpell

            if not self.dictionary_path.is_file():
                if not self.auto_download:
                    raise FileNotFoundError(str(self.dictionary_path))
                self._download_dictionary()

            self._sym_spell = SymSpell(
                max_dictionary_edit_distance=2,
                prefix_length=7
            )
            self._sym_spell.load_dictionary(
                str(self.dictionary_path),
                term_index=0,
                count_index=1,
            )

            self.enabled = True
            print(f"[SPELL] Dictionary ready: {self.dictionary_path.name}")
        except Exception as exc:
            self.enabled = False
            self.load_error = str(exc)
            print(f"[WARN] SymSpell disabled: {exc}")

    def _download_dictionary(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        print(f"[SPELL] Downloading dictionary: {self.dictionary_url}")
        urllib.request.urlretrieve(self.dictionary_url, str(self.dictionary_path))

    @staticmethod
    def _clean_letters(text: str) -> str:
        allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        return "".join(ch for ch in text.upper() if ch in allowed)

    def _correct_alpha_run(self, run: str) -> str:
        """Correct one alphabetic run without touching digits."""
        if len(run) <= 1:
            return run

        try:
            segmented = self._sym_spell.word_segmentation(run.lower())
            corrected = segmented.corrected_string.strip()
            return corrected.upper() if corrected else run
        except Exception as exc:
            self.load_error = str(exc)
            return run

    def correct(self, text: str) -> str:
        """
        Return corrected word/phrase for a cached letter run.

        Digit runs are preserved as-is and are never sent to SymSpell. Only
        alphabetic runs are segmented/corrected.
        """
        cleaned = self._clean_letters(text)

        if not cleaned:
            return ""

        if not self.enabled or self._sym_spell is None:
            return cleaned

        tokens = re.findall(r"[A-Z]+|[0-9]+", cleaned)
        corrected_tokens: list[str] = []

        for token in tokens:
            if token.isdigit():
                corrected_tokens.append(token)
            else:
                corrected_tokens.append(self._correct_alpha_run(token))

        return " ".join(part for part in corrected_tokens if part)

    def state(self) -> dict:
        return {
            "enabled": self.enabled,
            "dictionary": str(self.dictionary_path),
            "dictionary_exists": self.dictionary_path.is_file(),
            "error": self.load_error,
        }