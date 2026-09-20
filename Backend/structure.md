# Directory Structure: MookVani Backend

## Core Packages
- `data/`: Contains scripts for dataset management, feature extraction, and augmentation.
  - `datasets.py`: PyTorch Datasets and DataLoaders for word and sentence-level modeling.
  - `extract_word.py`: Script to extract MediaPipe 163-dimensional tensors for word-level videos.
  - `extract_sentence.py`: Script to extract tensors for full sentence videos.
  - `augment.py`: Contains affine, translation, and noise augmentations for landmarks.
  - `utils.py`: Utility functions for feature extraction.
- `models/`: Contains neural network architectures.
  - `isl_conformer.py`: ISL_Conformer definition using `MultiStreamEmbedding`, `EfficientNet1D`, and `ConformerBlock`.
- `training/`: Contains PyTorch training loops.
  - `train_word.py`: Training script for the static word classification model (CrossEntropy).
  - `train_sentence.py`: Training script for the sequence-to-sequence sentence model (CTC Loss).
- `utils/`: Evaluation and decoding utilities.
  - `metrics.py`: Computes WER (Word Error Rate) and exact match accuracy.
  - `gloss_utils.py`: Maps sentence ID sequences to word-level integer sequences.

## Top-Level Files
- `requirements.txt`: Python dependencies.
- `venv/`: Local Python virtual environment.
