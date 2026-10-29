"""Sentence-word segmentation data builder pipeline."""

from .builder import (
    DATASET_ROOT,
    SENTENCES_CSV,
    WORD_VIDEOS_ROOT,
    SEGMENTATION_CSV,
    scan_word_videos,
    load_sentences,
    match_words_to_videos,
    compose_sentence_video,
    hash_sentence,
    SegmentationDB,
)

__all__ = [
    "DATASET_ROOT",
    "SENTENCES_CSV",
    "WORD_VIDEOS_ROOT",
    "SEGMENTATION_CSV",
    "scan_word_videos",
    "load_sentences",
    "match_words_to_videos",
    "compose_sentence_video",
    "hash_sentence",
    "SegmentationDB",
]
