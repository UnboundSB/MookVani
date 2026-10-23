# Directory Structure: Backend/models/word_vocabulary

The **published vocabulary** of the project: the ordered list of the word classes
every trainer, every evaluation script and the live inference path work with.

It is generated from the word corpus with the shared selection rule of
`Backend/utils/dataloader_word.py`:

* a class with fewer than **14** clips is eliminated from the vocabulary
  (its clips stay on disk — the pretrainer and the other models still use them),
* every surviving class contributes at most **20** clips,
* the classes are ordered by descending clip count (the "top most frequent" classes).

For the current corpus (262 class directories / 4 273 clips) this yields **228
classes / 3 941 clips** (3135 train + 806 val), and the eliminated ones are the 34
rarest classes (4, 8, 9 and 10 clips).

Regenerate it with:

```python
from utils.dataloader_word import refresh_canonical_classes
refresh_canonical_classes()
```

## Files
- `classes.json`: Plain JSON list of the 228 class names, in the canonical order. The class index of a name is its position in that list; the CTC blank of a sentence model is `len(classes)`.
- `split.json`: **The canonical train/val manifest.** For every class it pins the exact training clips and the exact validation clips. Every phase of the word pipeline reads this file (through `utils/splits.py`), so a clip that is training data in one phase is never validation data in another. Its fingerprint (`76586776f574e698` for the current corpus) is recorded in the metadata of every phase. Regenerate it with `python scripts/save_phase4_split.py --canonical` and verify it with `python scripts/audit_split_leakage.py`.
- `word_lm.json`: The word-level **language model** of the continuous corpus (a smoothed unigram + bigram over 178 word types / 2 122 tokens, estimated on the TRAINING split only). It is what the prefix beam search rescores its hypotheses with (`--beam_alpha`); the decoder uses the BIGRAM only, because the unigram prior of that corpus is the pronoun skew that causes the collapse. Regenerate it with `python scripts/build_word_lm.py`.

## Immediate Subdirectories
- *(None)*
