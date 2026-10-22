# Directory Structure: Backend/eda

Exploratory Data Analysis (EDA) module for dataset inspection, format verification, and visual distribution generation.

## Files
- `analyze_tensor_dataset.py`: **Tensor Dataset Integrity & MediaPipe Collapse Audit.** Scans the finished 163-dim MediaPipe landmark tensors (word level and sentence level) for upstream extraction collapse: dead (all-zero) frames, frozen/duplicated frames, NaN/Inf, out-of-range coordinates, per-modality presence rates, flag/geometry inconsistencies, raw-zero vs resting-pose encoding of an undetected modality, constant samples, motion energy, class balance, exact-duplicate leakage, the **geometric frame** each corpus is expressed in (the neck = shoulder midpoint is used as the reference point, which reveals that the three corpora were extracted with three different anchors), and a nearest-centroid/logistic **discriminability probe** (accuracy vs chance, between/within-class scatter ratio, PCA effective rank). Writes a JSON report plus collapse and separability plots, and a cross-corpus domain-shift report.
- `check_dataset.py`: Validates dataset folder organization and directory integrity.
- `check_file_type.py`: Verifies file types and video encoding formats across dataset sources.
- `check_progress.py`: Monitors progress for batch extraction and feature processing jobs.
- `check_word_collision.py`: Checks for overlapping vocabulary and collisions across gloss sets.
- `extract_dataset.py`: Extracts raw video and metadata archives for the primary dataset.
- `extract_isign_only.py`: Unpacks and filters subsets specifically from the iSign dataset.
- `generate_isl_cslrt_eda.py`: Standalone EDA script generating comprehensive distribution plots for ISL-CSLRT (word and sentence levels).
- `inspect_eda.py`: Console-based data inspector printing high-level dataset metrics.
- `sentence_class_frequency_eda.py`: Analyzes distribution and frequencies of trained vocabulary classes in continuous sentences. Reads the published vocabulary (`models/word_vocabulary/classes.json`).
- `sentence_class_frequencies.json`: Stored JSON frequency metrics and vocabulary coverage data.
- `structure.md`: Directory structure documentation for eda.
- `tensor_collapse_audit_tensors_include50_word_level.json`: Audit report for the word-level tensor corpus (262 classes / 4,273 samples / 275,009 frames).
- `tensor_collapse_audit_tensors_sentence_level.json`: Audit report for the continuous sentence tensor corpus (1,579 samples / 387,665 frames).
- `tensor_collapse_audit_cross_corpus.json`: Cross-corpus domain-shift report (modality presence, missing-modality encoding, geometric frame).

## Immediate Subdirectories
- `plots/`: Output directory containing generated EDA charts, distribution graphs, and overview dashboards (e.g., `sentence_active_words_frequency.png`, `sentence_all_262_class_frequencies.png`, `sentence_word_frequencies.png`, `tensor_collapse_audit_*.png`, `tensor_separability_*.png`).
