# Directory Structure: Backend/models/phase4_word_model

Phase 4 full-FPS fine-tuned model. No frame clipping; random time-stretch/mirror/tilt
bounded shift augmentation. 50 epochs: transformer frozen for first 30 (CosineAnnealingWarmRestarts
on RNN+CNN+head), then all layers train for the final 20 (CosineAnnealing) + a plateau escape.

Saves best/final checkpoints, classes.json, history.json, classification report, and plots/.
Also writes `split_paths.json` (the canonical train/val manifest consumed by phases 5 and 6).

## Files
- `best_phase4_model.pth`: Checkpoint with highest validation accuracy.
- `final_phase4_model.pth`: Final trained weights at completion of training epochs.
- `classes.json`: List of word vocabulary classes.
- `history.json`: Per-epoch metrics history (loss, accuracy, F1/precision/recall, LR per group).
- `classification_report.txt`: Full classification report per sign class.
- `predictions.json`: Validation (true, predicted) pairs.
- `split_paths.json`: Canonical train/val manifest (fingerprint shared with phases 5-6).
- `structure.md`: Directory structure documentation.

## Immediate Subdirectories
- `plots/`: Contains evaluation figures (loss_acc.png, metrics.png, lr_schedule.png, confusion_matrix.png).
