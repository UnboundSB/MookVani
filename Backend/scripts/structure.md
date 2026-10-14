# Directory Structure: Backend/scripts

Utility and processing scripts for data extraction, augmentation, dataset loaders, and evaluations.

## Files
- `__init__.py`: Package initialization for the scripts module.
- `audit_data_leakage.py`: Verifies split hygiene and checks for train-test data leakage.
- `augment.py`: Spatial and temporal augmentation methods for sign keypoints and videos.
- `augment_extract_sentence.py`: Augmentation and feature extraction pipeline for sentence sequences.
- `check_ckpt.py`: Diagnostic utility to inspect PyTorch checkpoint keys and weights.
- `check_sizes.py`: Quick verification script for checking tensor shape and sequence lengths.
- `dataset.py`: PyTorch Dataset and DataLoader classes for sign language sequences.
- `debug_augmentation.py`: Debug script for validating keypoint augmentation correctness.
- `extract_images.py`: Feature extraction pipeline for image frame sequences (e.g., ISL_CSLRT Frames_Word_Level).
- `extract_video.py`: Unified video feature extraction pipeline for isolated word and continuous sentence recordings.
- `extraction.py`: Canonical 163-dim MediaPipe landmark feature extraction, skeleton normalization, and sanitization.
- `generate_labeled_confusion_matrix.py`: Generates labeled confusion matrix plots with class names.
- `generate_synthetic_poses.py`: Generates interpolated synthetic pose sequences for data augmentation.
- `live_progress.py`: Terminal progress monitor for long-running batch extraction jobs.
- `organize_directories.py`: Reorganizes model checkpoints, metrics, and plots into designated folders.
- `plot_confusion_matrix.py`: Script to plot and save confusion matrix figures.
- `structure.md`: Directory structure documentation for scripts.
- `tensor_augmentation.py`: Direct tensor augmentations including temporal jitter and keypoint dropout.
- `test_eval.py`: Evaluation script testing model inferences against validation sets.
- `utils.py`: Shared utilities for path handling, file I/O, and data conversions.

## Immediate Subdirectories
- `isign_pipeline/`: Dedicated processing scripts for unpacking and extracting iSign dataset videos.
