"""Feature extraction proxy forwarding to canonical implementation in scripts.extraction."""

import os
import sys

_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from extraction import (
    _normalize_skeleton,
    extract_frame_features,
    extract_frame_features_video_mode,
    clean_word
)

__all__ = [
    "_normalize_skeleton",
    "extract_frame_features",
    "extract_frame_features_video_mode",
    "clean_word"
]
