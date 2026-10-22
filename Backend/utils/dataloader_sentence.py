"""
Sentence-level dataloader: continuous sentences + the word clips we train on.
=============================================================================

This loader feeds the CTC sentence models with BOTH sources of supervision:

1. **The continuous sentence tensors** (``Backend/data/tensors_sentence_level``,
   described by ``index.json``).  Each clip is a ``(T, 163)`` MediaPipe
   landmark tensor of a human signing a whole sentence, and its target is the
   ordered list of the words of that sentence.
2. **The word clips we trained on** (the very same selection the word-level
   loader uses: the top-most frequent classes with a 15-clip budget).  Every
   word clip is treated as a one-word "sentence", which keeps the CTC head
   anchored to the isolated-sign geometry and prevents the blank-collapse that
   a purely continuous corpus produces.

Both sources go through the SAME augmentation set
(:mod:`utils.pose_augmentation`):

  * **mirror** - X inverted and every left/right pair swapped (hands, arms,
    eyes, cheeks AND the mouth corners), so the clip reads as a left-handed
    signer instead of an upside-down / backwards pose;
  * **tilt** - the skeleton rotates around the NECK (never around the origin,
    the corpus is anchor relative), so the signer leans;
  * **left / right shift** - a bounded translation expressed in shoulder
    widths, so the coordinates can never be pushed out of frame;
  * **speed up / slow down** - the clip is resampled in time;
  * **temporal shift / SpecAugment** - the sign starts earlier or later, and a
    few frames are cut out.

Geometric frame
---------------
The sentence corpus was extracted with a different anchor than the word corpus
(neck at ``(0.01, 0.61, -0.52)`` vs ``(3.59, 2.72, -1.20)``, a displacement of
more than four shoulder widths).  ``harmonize=True`` moves the sentence clips
into the word-corporus frame, which is the frame the backbone was pretrained
in; the word clips need no correction.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from utils.dataloader_word import (
        FILL_MISSING_DEFAULT,
        MAX_SAMPLES_PER_CLASS,
        MIN_SAMPLES_PER_CLASS,
        NOTHING_CLASS_NAME,
        WORD_LEVEL_DIR_DEFAULT,
        discover_word_classes,
        split_word_classes,
    )
    from utils.pose_augmentation import (
        AugmentConfig,
        DEFAULT_AUGMENT,
        NECK_OFFSET,
        NECK_OFFSET_WORD,
        augment_pose_sequence,
        as_sequence,
        fill_missing_temporal,
        harmonize_features,
    )
except ImportError:  # pragma: no cover
    from dataloader_word import (  # type: ignore
        FILL_MISSING_DEFAULT,
        MAX_SAMPLES_PER_CLASS,
        MIN_SAMPLES_PER_CLASS,
        NOTHING_CLASS_NAME,
        WORD_LEVEL_DIR_DEFAULT,
        discover_word_classes,
        split_word_classes,
    )
    from pose_augmentation import (  # type: ignore
        AugmentConfig,
        DEFAULT_AUGMENT,
        NECK_OFFSET,
        NECK_OFFSET_WORD,
        augment_pose_sequence,
        as_sequence,
        fill_missing_temporal,
        harmonize_features,
    )

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SENTENCE_DIR_DEFAULT = os.path.join(BACKEND_DIR, "data", "tensors_sentence_level")
INDEX_FILE_NAME = "index.json"

DEFAULT_MAX_LEN = 220        # frames
DEFAULT_MAX_WORDS = 18       # CTC target length


# --------------------------------------------------------------------------- #
# Index handling
# --------------------------------------------------------------------------- #
def load_sentence_index(sentence_dir: str = SENTENCE_DIR_DEFAULT) -> List[dict]:
    """Read the ``index.json`` manifest of the continuous sentence corpus."""
    index_file = os.path.join(sentence_dir, INDEX_FILE_NAME)
    if not os.path.exists(index_file):
        raise FileNotFoundError(
            f"Missing {index_file} - run scripts/extract_real_sentences.py first."
        )
    with open(index_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        payload = payload.get("items", payload.get("clips", []))
    return list(payload)


def remap_sentence_words(
    items: Sequence[dict],
    class_to_idx: Dict[str, int],
    max_out_scope: int = 5,
    max_words: int = DEFAULT_MAX_WORDS,
    drop_sentence_on_unknown: bool = False,
) -> Tuple[List[dict], dict]:
    """Re-index the sentence targets onto the CURRENT vocabulary.

    ``index.json`` stores the word indices of the 262-class vocabulary that was
    in use when the corpus was extracted.  The word loader now trains on a
    different (smaller, frequency filtered) class list, so the sentence targets
    have to be re-indexed BY NAME.

    A word that is not part of the current vocabulary is treated exactly like
    an out-of-scope word: it is removed from the CTC target (the model cannot
    be scored on a sign it does not know).  With
    ``drop_sentence_on_unknown=True`` the whole sentence is dropped instead.

    A sentence is dropped when
      * it has more than ``max_out_scope`` out-of-scope words (the sentence
        contains a lot of sign the model cannot be scored on), or
      * nothing is left of it after the re-indexing.

    Returns the surviving items and a small report.
    """
    kept: List[dict] = []
    dropped_unknown = 0
    dropped_out_of_scope = 0
    dropped_empty = 0
    unknown_words = 0

    for item in items:
        out_scope = item.get("out_scope_words") or []
        if max_out_scope >= 0 and len(out_scope) > max_out_scope:
            dropped_out_of_scope += 1
            continue

        words = item.get("in_scope_words") or []
        indices: List[int] = []
        kept_words: List[str] = []
        unknown = False
        for word in words:
            idx = class_to_idx.get(word)
            if idx is None:
                unknown = True
                unknown_words += 1
                if drop_sentence_on_unknown:
                    break
                continue
            indices.append(idx)
            kept_words.append(word)
        if unknown and drop_sentence_on_unknown:
            dropped_unknown += 1
            continue
        if not indices:
            dropped_empty += 1
            continue

        new_item = dict(item)
        new_item["target_words"] = kept_words[:max_words]
        new_item["target_indices"] = indices[:max_words]
        # The historic key is rewritten in place, so every downstream consumer
        # keeps working with the CURRENT vocabulary.
        new_item["in_scope_indices"] = new_item["target_indices"]
        kept.append(new_item)

    report = {
        "total": len(items),
        "kept": len(kept),
        "dropped_unknown_word": dropped_unknown,
        "dropped_out_of_scope": dropped_out_of_scope,
        "dropped_empty": dropped_empty,
        "unknown_words": unknown_words,
    }
    return kept, report


def load_sentence_items(
    sentence_dir: str,
    class_to_idx: Dict[str, int],
    max_out_scope: int = 5,
    max_words: int = DEFAULT_MAX_WORDS,
    require_file: bool = True,
) -> Tuple[List[dict], dict]:
    """Load the continuous corpus and re-index its targets onto ``class_to_idx``."""
    items = load_sentence_index(sentence_dir)
    kept, report = remap_sentence_words(
        items, class_to_idx, max_out_scope=max_out_scope, max_words=max_words
    )
    if require_file:
        kept = [it for it in kept if os.path.exists(it.get("tensor_path", ""))]
    return kept, report


def split_sentence_items(
    items: Sequence[dict],
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """Zero-leakage split of the continuous corpus.

    The split is done on the SENTENCE (not on the clip), so the same sentence
    text never appears in both partitions.
    """
    rng = random.Random(seed)
    groups: Dict[Tuple[str, str], List[dict]] = {}
    for item in items:
        key = (item.get("dataset", ""), item.get("sentence", ""))
        groups.setdefault(key, []).append(item)

    keys = list(groups.keys())
    rng.shuffle(keys)
    n_val = max(1, int(round(len(keys) * val_fraction)))

    train_items: List[dict] = []
    val_items: List[dict] = []
    for i, key in enumerate(keys):
        if i < n_val:
            val_items.extend(groups[key])
        else:
            train_items.extend(groups[key])

    rng.shuffle(train_items)
    return train_items, val_items


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class ISLSentenceTensorDataset(Dataset):
    """Continuous sentences + isolated word clips, ready for CTC.

    Every item is ``(clip, length, target_indices)`` with the clip shaped
    ``(T, 163)`` and the target shaped ``(L,)`` (``L`` = number of words).
    """

    def __init__(
        self,
        sentence_items: Sequence[dict] = (),
        word_items: Sequence[Tuple[str, int]] = (),
        is_train: bool = True,
        max_len: int = DEFAULT_MAX_LEN,
        augment_config: Optional[AugmentConfig] = None,
        harmonize: bool = True,
        fill_missing: str = FILL_MISSING_DEFAULT,
    ):
        self.sentence_items = list(sentence_items)
        self.word_items = list(word_items)
        self.is_train = is_train
        self.max_len = max_len
        self.augment_config = augment_config
        self.harmonize = harmonize
        self.fill_missing = fill_missing

    def __len__(self) -> int:
        return len(self.sentence_items) + len(self.word_items)

    # -- helpers ------------------------------------------------------- #
    def _prepare(self, clip: torch.Tensor, harmonize_offset) -> torch.Tensor:
        clip = as_sequence(clip)
        if self.harmonize:
            # The word corpus IS the reference frame, so its clips are only
            # cleared of the dead face block (offset 0).  The continuous corpus
            # was extracted with a different anchor and needs the real
            # correction (NECK_OFFSET).
            clip = harmonize_features(clip, zero_face=True, offset=harmonize_offset)
        # Fill the drop-out holes with the last valid value: a hand that
        # MediaPipe lost for three frames would otherwise read as a jump to
        # the origin, which is neither a real movement nor a useful signal.
        # The fill runs AFTER the harmonisation and it sets the presence flag
        # of the filled frames, so the zero convention of the corpus is not
        # restored afterwards.
        if self.fill_missing and self.fill_missing != "none":
            clip = fill_missing_temporal(clip, mode=self.fill_missing)
        if self.is_train and self.augment_config is not None:
            clip = augment_pose_sequence(clip, self.augment_config)
        if clip.shape[0] > self.max_len:
            clip = clip[: self.max_len]
        return clip

    # -- items --------------------------------------------------------- #
    def __getitem__(self, idx: int):
        if idx < len(self.sentence_items):
            item = self.sentence_items[idx]
            clip = self._prepare(
                torch.load(item["tensor_path"], weights_only=True), NECK_OFFSET
            )
            target = torch.tensor(item["target_indices"], dtype=torch.long)
            return clip, clip.shape[0], target

        path, label = self.word_items[idx - len(self.sentence_items)]
        clip = self._prepare(
            torch.load(path, weights_only=True), NECK_OFFSET_WORD
        )
        target = torch.tensor([label], dtype=torch.long)
        return clip, clip.shape[0], target


def collate_sentence_batch(batch):
    """CTC collation: ``(inputs, input_lengths, targets, target_lengths)``.

    The targets are concatenated (the classic CTC layout) and the number of
    words of every sample is returned separately.
    """
    xs, lengths, labels = zip(*batch)
    B, maxT = len(xs), max(int(l) for l in lengths)
    xp = torch.zeros(B, maxT, xs[0].shape[-1], dtype=torch.float32)
    for i, x in enumerate(xs):
        xp[i, : int(lengths[i])] = x
    non_empty = [l for l in labels if len(l) > 0]
    targets = torch.cat(non_empty) if non_empty else torch.tensor([], dtype=torch.long)
    return (
        xp,
        torch.tensor([int(l) for l in lengths], dtype=torch.long),
        targets,
        torch.tensor([len(l) for l in labels], dtype=torch.long),
    )


# --------------------------------------------------------------------------- #
# The loader factory
# --------------------------------------------------------------------------- #
def get_sentence_dataloaders(
    sentence_dir: str = SENTENCE_DIR_DEFAULT,
    word_dir: str = WORD_LEVEL_DIR_DEFAULT,
    classes: Optional[Sequence[str]] = None,
    batch_size: int = 8,
    min_samples_per_class: int = MIN_SAMPLES_PER_CLASS,
    max_samples_per_class: int = MAX_SAMPLES_PER_CLASS,
    include_word_clips: bool = True,
    word_clip_ratio: float = 1.0,
    val_fraction: float = 0.1,
    max_len: int = DEFAULT_MAX_LEN,
    max_words: int = DEFAULT_MAX_WORDS,
    max_out_scope: int = 5,
    augment_config: Optional[AugmentConfig] = None,
    harmonize: bool = True,
    fill_missing: str = FILL_MISSING_DEFAULT,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
    save_meta_path: Optional[str] = None,
):
    """Build the sentence-level train / val dataloaders.

    ``classes`` defaults to the selection of the word-level loader, so both
    loaders always agree on the vocabulary (and therefore on the CTC blank id
    and on the head size).

    Returns
    -------
    ``(train_loader, val_loader, class_to_idx, idx_to_class, classes, report)``
    """
    if classes is None:
        infos = discover_word_classes(word_dir, min_samples_per_class, max_samples_per_class)
        classes = [info.name for info in infos]
    classes = list(classes)

    class_to_idx = {name: i for i, name in enumerate(classes)}
    idx_to_class = {i: name for i, name in enumerate(classes)}

    # ---- 1. the continuous corpus ------------------------------------- #
    index_items = load_sentence_index(sentence_dir)
    kept_items, report = remap_sentence_words(
        index_items, class_to_idx, max_out_scope=max_out_scope, max_words=max_words
    )
    train_sent, val_sent = split_sentence_items(kept_items, val_fraction=val_fraction, seed=seed)

    # ---- 2. the word clips we trained on ------------------------------ #
    train_word: List[Tuple[str, int]] = []
    val_word: List[Tuple[str, int]] = []
    if include_word_clips:
        infos = discover_word_classes(word_dir, min_samples_per_class, max_samples_per_class)
        tr, va = split_word_classes(infos, val_fraction=max(val_fraction, 0.2), seed=seed)
        train_word = tr
        val_word = va
        if word_clip_ratio != 1.0:
            n_tr = int(len(tr) * word_clip_ratio)
            n_va = int(len(va) * word_clip_ratio)
            train_word, val_word = tr[:n_tr], va[:n_va]

    train_dataset = ISLSentenceTensorDataset(
        train_sent, train_word, is_train=True, max_len=max_len,
        augment_config=augment_config if augment_config is not None else DEFAULT_AUGMENT,
        harmonize=harmonize, fill_missing=fill_missing,
    )
    val_dataset = ISLSentenceTensorDataset(
        val_sent, val_word, is_train=False, max_len=max_len,
        augment_config=None, harmonize=harmonize, fill_missing=fill_missing,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_sentence_batch, num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_sentence_batch, num_workers=num_workers, pin_memory=pin_memory,
    )

    print("=" * 78)
    print("ISL SENTENCE-LEVEL DATALOADER (continuous sentences + trained word clips)")
    print("=" * 78)
    print(f"Sentence corpus:              {sentence_dir}")
    print(f"Word corpus:                  {word_dir}")
    print(f"Vocabulary size:              {len(classes)} (CTC blank = {len(classes)})")
    print(f"Continuous clips on disk:     {report['total']}")
    print(f"  kept:                       {report['kept']}")
    print(f"  dropped (unknown word):     {report['dropped_unknown_word']}")
    print(f"  target words dropped:       {report['unknown_words']}")
    print(f"  dropped (out of scope):     {report['dropped_out_of_scope']}")
    print(f"  dropped (empty):            {report['dropped_empty']}")
    print(f"Word clips added (train/val): {len(train_word)} / {len(val_word)}")
    print(f"Train samples:                {len(train_dataset)}")
    print(f"Val samples:                  {len(val_dataset)}")
    print(f"Augmentation:                 mirror / tilt / shift / speed / temporal")
    print(f"Hole filling:                {fill_missing}")
    print("=" * 78)

    if save_meta_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_meta_path)), exist_ok=True)
        with open(save_meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "classes": classes,
                "num_classes": len(classes),
                "ctc_blank_id": len(classes),
                "class_to_idx": class_to_idx,
                "report": report,
            }, f, indent=2)
        print(f"Saved sentence class metadata to: {save_meta_path}")

    return train_loader, val_loader, class_to_idx, idx_to_class, classes, report
