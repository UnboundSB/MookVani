# Directory Structure: Backend/utils

Shared utilities: the word-level and sentence-level dataloaders, the single
canonical implementation of the pose augmentations, gloss normalization and
evaluation metrics.

## Files
- `__init__.py`: Package entry point re-exporting the dataloaders, the augmentation primitives and the metrics helpers.
- `checkpoint_utils.py`: Checkpoint helpers that survive a change of the vocabulary. `load_flexible()` drops the shape-mismatched entries (the class heads) before loading, so a checkpoint trained on the old 262-class list can still warm-start the backbone of a model with the current class list instead of dying with a `RuntimeError: size mismatch`.
- `dataloader_word.py`: **Word-level dataloader.** Loads the TOP-MOST FREQUENT classes of the word corpus: a class with fewer than 14 clips is eliminated from the loader (its files stay on disk), every surviving class contributes at most 15 clips, and the classes are ordered by descending clip count. Adds the synthetic `[NOTHING]` idle class (as the last label) and exposes the published vocabulary (`models/word_vocabulary/classes.json`).
- `dataloader_sentence.py`: **Sentence-level dataloader.** Loads the continuous sentence tensors (via `index.json`, re-indexed onto the current vocabulary BY NAME) AND the word clips the word model was trained on, and applies the canonical augmentation set (mirror / tilt / bounded left-right shift / speed up or slow down / temporal shift / SpecAugment) plus the temporal hole filling.
- `gloss_utils.py`: Tokenization, text normalization, and gloss-to-index mapping utilities.
- `metrics.py`: Evaluation metrics implementation (Word Error Rate, Top-k accuracy, F1 score).
- `pose_augmentation.py`: **The single source of truth for the pose augmentations.** Canonical 163-dim layout, the missing-modality convention (an undetected modality owns a zero block), the cross-corpus geometric harmonisation, and the augmentation primitives: `mirror_pose` (every left/right pair swapped, mouth corners included, around the signer's own axis), `tilt_pose` (around the neck), `shift_pose` (bounded left/right shift), `time_stretch` (speed up / slow down), `temporal_shift`, `temporal_mask`, `scale_pose`, plus `fill_missing_temporal` / `TemporalPoseTracker` (carry the last valid value forward instead of dropping a landmark to zero).
- `structure.md`: Directory structure documentation for utils.

## Immediate Subdirectories
- *(None)*
