"""
Word-level language model for the ISL sentence decoders.
=======================================================

The CTC models are trained on a corpus of 1 579 continuous sentences with only
2 359 word instances, i.e. 1.5 words per sentence, and the distribution is
extremely skewed (``i`` 490, ``you`` 381, everything else below 25).  A purely
acoustic decoder has therefore nothing to disambiguate a short, frequent
sequence: it can only pick the most likely word at every frame.

A word-level language model gives the decoder a second opinion:

    score(words) = log P_acoustic(words) + alpha * log P_LM(words) + beta * |words|

``alpha`` (the LM weight) and ``beta`` (the insertion penalty) are the two
knobs that decide how much the language is allowed to override the acoustics.
They have to be TUNED on a validation split: a model that trusts the language
too much starts to produce fluent sentences that are wrong.

The model is a smoothed unigram + bigram:

    P(w)          = (count(w) + k) / (total + k * V)
    P(w2 | w1)    = (count(w1, w2) + k) / (count(w1) + k * V)

with a back-off to the unigram when a bigram was never seen.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence


class WordLM:
    """A smoothed unigram + bigram language model over the sign vocabulary."""

    def __init__(self, unigram: Optional[Counter] = None, bigram: Optional[Counter] = None,
                 vocab_size: int = 0, add_k: float = 0.5):
        self.unigram: Counter = unigram or Counter()
        self.bigram: Counter = bigram or Counter()
        self.bigram_first: Counter = Counter()
        self.vocab_size = int(vocab_size)
        self.add_k = float(add_k)
        self._total = sum(self.unigram.values())

    # ------------------------------------------------------------------ #
    @classmethod
    def from_sentences(cls, sentences: Iterable[Sequence[str]], vocab_size: int = 0,
                       add_k: float = 0.5) -> "WordLM":
        """Build the model from a corpus of token sequences."""
        uni: Counter = Counter()
        bi: Counter = Counter()
        for words in sentences:
            words = list(words)
            uni.update(words)
            for a, b in zip(words, words[1:]):
                bi[(a, b)] += 1
        return cls(uni, bi, vocab_size=vocab_size or len(uni), add_k=add_k)

    # ------------------------------------------------------------------ #
    def log_unigram(self, word: str) -> float:
        v = max(self.vocab_size, 1)
        p = (self.unigram.get(word, 0) + self.add_k) / (self._total + self.add_k * v)
        return math.log(max(p, 1e-12))

    def log_bigram(self, prev: Optional[str], word: str) -> float:
        """log P(word | prev) with a back-off to the unigram."""
        v = max(self.vocab_size, 1)
        if prev is None:
            return self.log_unigram(word)
        denom = self.unigram.get(prev, 0) + self.add_k * v
        num = self.bigram.get((prev, word), 0) + self.add_k
        if self.bigram.get((prev, word), 0) == 0 and self.unigram.get(prev, 0) == 0:
            return self.log_unigram(word)
        return math.log(max(num / max(denom, 1e-12), 1e-12))

    # ------------------------------------------------------------------ #
    def score(self, words: Sequence[str]) -> float:
        """The log-probability of a word sequence (unigram + bigram)."""
        if not words:
            return 0.0
        total = 0.0
        prev = None
        for w in words:
            total += self.log_bigram(prev, w)
            prev = w
        return total

    def score_ids(self, ids: Sequence[int], id_to_word: Sequence[str]) -> float:
        return self.score([id_to_word[i] for i in ids if 0 <= i < len(id_to_word)])

    # ------------------------------------------------------------------ #
    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "unigram": dict(self.unigram),
            "bigram": {f"{a}\t{b}": c for (a, b), c in self.bigram.items()},
            "vocab_size": self.vocab_size,
            "add_k": self.add_k,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "WordLM":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        uni = Counter(payload.get("unigram", {}))
        bi = Counter()
        for key, count in payload.get("bigram", {}).items():
            a, _, b = key.partition("\t")
            bi[(a, b)] = count
        return cls(uni, bi, vocab_size=payload.get("vocab_size", len(uni)),
                   add_k=payload.get("add_k", 0.5))


def build_word_lm_from_index(
    index_file: str,
    classes: Optional[Sequence[str]] = None,
    val_fraction: float = 0.1,
    seed: int = 42,
    add_k: float = 0.5,
) -> WordLM:
    """Build the language model from the sentence corpus (its TRAINING split).

    The validation sentences are excluded, otherwise the LM would be scored on
    data it has already seen and the tuning of ``alpha`` would be meaningless.
    """
    import random

    with open(index_file, "r", encoding="utf-8") as f:
        items = json.load(f)

    # zero-leakage split by sentence (the same rule the trainers use)
    groups: Dict[str, List[dict]] = defaultdict(list)
    for it in items:
        groups[(it.get("sentence", "") or "").strip().lower()].append(it)
    keys = sorted(groups.keys())
    random.Random(seed).shuffle(keys)
    n_val = max(1, int(round(len(keys) * val_fraction)))
    train_keys = set(keys[n_val:])

    sentences = []
    for key, group in groups.items():
        if key not in train_keys:
            continue
        for it in group:
            words = it.get("in_scope_words") or []
            if words:
                sentences.append(words)

    return WordLM.from_sentences(sentences, vocab_size=len(classes or []), add_k=add_k)
