# MookVani

MookVani is an Indian Sign Language (ISL) continuous sign language recognition project. It uses a canonical 163-dimensional feature extraction pipeline via MediaPipe and a customized Conformer architecture (ISL_Conformer).

## Structure
- `Backend/models`: Contains the neural network models.
- `Backend/data`: Contains the datasets processing modules, augmentations, and extraction scripts.
- `Backend/extract_features.py`: MediaPipe keypoint extraction scripts.
- `Backend/train_word.py`: Training script for the word-level baseline.
- `Backend/train_sentence.py`: Training script for sentence-level sequences (CTC).
- `Backend/evaluate.py`: Evaluation metrics (WER / Edit Distance).

## Setup
1. Create a Python environment (`python -m venv venv`)
2. Install requirements (`pip install -r Backend/requirements.txt`)
3. Download MediaPipe models and extract features using `Backend/extract_features.py`

## Features
- Multi-stream conformer model for 3D landmark combinations (Hands, Pose, Face)
- Scale normalization and skeletal data augmentations
- Word error rate analysis via edit distance computations
