"""Build the word-level language model of the continuous corpus.

The language model is estimated on the TRAINING split of the sentence corpus
only (the validation sentences stay unseen, so the tuning of the LM weight
stays honest), and it is what the prefix beam search rescores its hypotheses
with.

    python scripts/build_word_lm.py
    ->  Backend/models/word_vocabulary/word_lm.json
"""
import os
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from utils.dataloader_word import canonical_classes  # noqa: E402
from utils.word_lm import build_word_lm_from_index  # noqa: E402

INDEX = os.path.join(BACKEND_DIR, "data", "tensors_sentence_level", "index.json")
OUT = os.path.join(BACKEND_DIR, "models", "word_vocabulary", "word_lm.json")


def main():
    classes = canonical_classes()
    lm = build_word_lm_from_index(INDEX, classes=classes)
    total = sum(lm.unigram.values())
    print(f"[LM] {len(lm.unigram)} word types, {total} tokens")
    top = lm.unigram.most_common(8)
    print("[LM] most frequent: " + ", ".join(f"{w}={c}" for w, c in top))
    print(f"[LM] the top-2 hold {100*(top[0][1]+top[1][1])/max(1,total):.1f}% of the tokens - "
          f"that is why the decoder uses the BIGRAM only (the word order), never the "
          f"unigram prior.")
    lm.save(OUT)
    print(f"[LM] saved -> {OUT}")


if __name__ == "__main__":
    main()
