# Directory Structure: Backend/scripts

Utility and processing scripts for data extraction, augmentation, dataset loaders, and evaluations.

The word-level dataloaders now live in `Backend/utils/` (`dataloader_word.py`,
`dataloader_sentence.py`, `pose_augmentation.py`); this folder holds only the
extraction / diagnostic / plotting utilities.

## Files
- `__init__.py`: Package initialization for the scripts module (feature extraction + gloss utilities).
- `audit_data_leakage.py`: Verifies split hygiene and checks for train-test data leakage.
- `audit_split_leakage.py`: **Proves that no phase leaks across the others.** Checks that the canonical split manifest is leak-free (no clip in both partitions, every class stratified, every file filed under its own class), that EVERY phase that reads the corpus resolves the same manifest fingerprint, that the legacy Phase-4 file agrees with it and that the corpus still matches it.
- `benchmark_expansion.py`: Controlled comparison of the temporal-expansion architectures (baseline / compressed / expanded-before / expanded-after) on the same data, seed, loss and epoch budget: parameters, latency, epoch time and the WER per epoch.
- `augment.py`: Spatial and temporal augmentation methods for sign keypoints and videos.
- `augment_extract_sentence.py`: Augmentation and feature extraction pipeline for sentence sequences.
- `benchmark_expansion.py`: Controlled comparison of the temporal-expansion architectures (baseline / compressed / expanded-before / expanded-after) on the same data, seed, loss and epoch budget: parameters, latency, epoch time and the WER per epoch.
- `build_word_lm.py`: Builds the word-level language model of the continuous corpus (the training split only) into `models/word_vocabulary/word_lm.json`. It is what the prefix beam search rescores its hypotheses with.
- `convert_word_corpus_for_yap.py`: **Adapts the word corpus to the format the YAP segmentor expects.** The segmentor was written against `.npy` clips of `(T, J*3)` with J=42 (the two MediaPipe hands), while our corpus is `.pt` tensors of `(T, 163)` (hands + 4 arm points + 7 face points + 4 flags) — without this the segmentor finds no data at all and trains on an empty dataset. It writes `data/yap_word_level/<class>/<clip>.npy` with the 53 corpus landmarks copied verbatim (the hands, the 2 shoulders + the 2 elbows and the 7 face points) - exactly the layout of `YAPConfig.corpus()`.
- `check_ckpt.py`: Diagnostic utility to inspect PyTorch checkpoint keys and weights.
- `check_collapse.py`: Diagnostic utility auditing CTC blank collapse, burst durations, and emission health.
- `check_real_sentence_collapse.py`: Audits real continuous human sign video predictions, burst length, and blank ratios. Applies the same feature pipeline as the training (`harmonize_features` + `_reclamp`), because the three tensor corpora use three different anchors and a harmonisation-trained model otherwise looks collapsed on raw tensors.
- `check_real_sentence_phase2.py`: Zero-leakage Phase 2 audit script evaluating continuous checkpoints strictly on the hold-out validation partition.
- `check_sizes.py`: Quick verification script for checking tensor shape and sequence lengths.
- `debug_augmentation.py`: Debug script for validating keypoint augmentation correctness.
- `extract_images.py`: Feature extraction pipeline for image frame sequences (e.g., ISL_CSLRT Frames_Word_Level).
- `extract_real_sentences.py`: Extracts real continuous sentence videos into canonical 163-dim MediaPipe tensors with selective in-scope relabeling and multi-core parallel processing.
- `extract_video.py`: Unified video feature extraction pipeline for isolated word and continuous sentence recordings.
- `extraction.py`: Canonical 163-dim MediaPipe landmark feature extraction, skeleton normalization, and sanitization.
- `generate_labeled_confusion_matrix.py`: Generates labeled confusion matrix plots with class names.
- `generate_synthetic_poses.py`: Generates interpolated synthetic pose sequences for data augmentation.
- `live_progress.py`: Terminal progress monitor for long-running batch extraction jobs.
- `organize_directories.py`: Reorganizes model checkpoints, metrics, and plots into designated folders.
- `plot_model_report.py`: Builds the **full plotting report** of a trained model from what the trainers write: the loss / accuracy / F1 / WER curves, the learning rate, the confusion matrix (with its readable text report) and the per-class metrics. `--model_dir models/phase6_word_model --predictions models/phase6_word_model/predictions.json`.
- `plot_confusion_matrix.py`: Script to plot and save confusion matrix figures.
- `plotter.py`: Sentence dataset vocabulary coverage and distribution plotter analyzing ISL-CSLRT and iSign datasets. `match_sentence_tokens()` returns the in-scope words in SIGNING ORDER — multi-word glosses are matched in place (the previous version pushed every compound phrase to the front, which mis-ordered the CTC target of 18 of the 1579 extracted clips).
- `save_phase4_split.py`: Writes the **canonical train/val manifest** (`models/word_vocabulary/split.json`) that EVERY phase of the word pipeline reads, plus the legacy Phase-4 file consumed by Phase 5 and Phase 6. The split is per class and file-level, so a clip can never be in both partitions. `--canonical` writes only the manifest.
- `structure.md`: Directory structure documentation for scripts.
- `test_eval.py`: Evaluation script testing model inferences against validation sets.
- `test_phase6_word_model.py`: Diagnostic audit script verifying Phase 6 word model classification accuracy on raw tensors without blanks.
- `track_extraction_progress.py`: Real-time progress monitor displaying extraction percentage, dataset breakdown, and storage metrics.

## Immediate Subdirectories
- `isign_pipeline/`: Dedicated processing scripts for unpacking and extracting iSign dataset videos.
