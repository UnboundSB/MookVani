# Directory Structure: Backend/models/phase6_word_model

Phase 6 high-precision squeeze model breaking out of local minima with Lookahead,
decoupled AdamW, Annealed LR combined with ReduceLROnPlateau, Model EMA, and SpecAugment
frame masking. Gentle mixup regularization. Reuses the exact Phase 4/5 split.

Saves best standard, best EMA, final checkpoints, classes.json, history.json,
classification report, and plots/.

## Files
- `best_phase6_model.pth`: Best standard model weights.
- `ema_phase6_model.pth`: Best EMA shadow model weights.
- `final_phase6_model.pth`: Final model weights at completion of training epochs.
- `classes.json`: List of word vocabulary classes.
- `history.json`: Per-epoch metrics history (train/val loss, accuracy, F1/precision/recall, LR per group, plateau epochs).
- `classification_report.txt`: Full classification report per sign class (standard + EMA).
- `predictions.json`: Validation (true, predicted) pairs.
- `structure.md`: Directory structure documentation.

## Immediate Subdirectories
- `plots/`: Contains evaluation figures (loss_acc.png, metrics.png, lr_schedule.png, ema_vs_standard.png, confusion_matrix_best.png, confusion_matrix_ema.png).
