# Directory Structure: Backend

Core backend engine containing ML models, feature extractors, training loops, and inference services.

## Files
- `live_inference.py`: Real-time webcam and video inference script for live sign language translation.
- `merge_isign_to_dataset.py`: Merges and aligns iSign dataset annotations into main dataset directories.
- `requirements.txt`: Python package dependencies required for backend execution and training.
- `run_pipeline.py`: Pipeline orchestration script to trigger feature extraction and model training.
- `run_train_synthetic.py`: Script to launch training jobs on synthetic sentence datasets.
- `structure.md`: Directory structure documentation for Backend.

## Immediate Subdirectories
- `eda/`: Exploratory data analysis scripts and dataset inspection routines.
- `features/`: Feature extraction logic and MediaPipe holistic landmark processing.
- `models/`: Neural network architectures (Conformer, ISL models) and saved checkpoint weights.
- `scripts/`: Data augmentation, extraction pipelines, directory management, and evaluation utilities.
- `training/`: Model training routines, fine-tuning scripts, and validation evaluation scripts.
- `utils/`: Reusable utilities for gloss mapping, metric calculations, and formatting.
