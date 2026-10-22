# Directory Structure: Backend/models/word_vocabulary

The **published vocabulary** of the project: the ordered list of the word classes
every trainer, every evaluation script and the live inference path work with.

It is generated from the word corpus with the shared selection rule of
`Backend/utils/dataloader_word.py`:

* a class with fewer than **14** clips is eliminated from the vocabulary
  (its clips stay on disk — the pretrainer and the other models still use them),
* every surviving class contributes at most **15** clips,
* the classes are ordered by descending clip count (the "top most frequent" classes).

For the current corpus (262 class directories / 4 273 clips) this yields **228
classes** out of 262, and the eliminated ones are the 34 rarest classes
(10, 9, 8 and 4 clips).

Regenerate it with:

```python
from utils.dataloader_word import refresh_canonical_classes
refresh_canonical_classes()
```

## Files
- `classes.json`: Plain JSON list of the 228 class names, in the canonical order. The class index of a name is its position in that list; the CTC blank of a sentence model is `len(classes)`.

## Immediate Subdirectories
- *(None)*
