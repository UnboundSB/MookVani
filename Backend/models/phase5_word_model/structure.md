# Directory Structure: Backend/models/phase5_word_model

Phase 5 maximum accuracy squeeze model reusing the exact Phase 4 split (the canonical
manifest, so it cannot go stale). All layers unfrozen from epoch 1, OneCycleLR with SWA.

Saves best/final/swa checkpoints, classes.json, history.json, classification report, and plots/.

## Files
- `best_phase5_model.pth`: Checkpoint with highest validation accuracy.
- `final_phase5_model.pth`: Final trained weights at completion of training epochs.
- `swa_phase5_model.pth`: Stochastic Weight Averaged model (tail epochs).
- `classes.json`: List of word vocabulary classes.
- `history.json`: Per-epoch metrics history (loss, accuracy, F1/precision/recall, LR per group, SWA val acc).
- `classification_report.txt`: Full classification report per sign class (best + SWA).
- `predictions.json`: Validation (true, predicted) pairs.
- `structure.md`: Directory structure documentation.

## Immediate Subdirectories
- `plots/`: Contains evaluation figures (loss_acc.png, metrics.png, lr_schedule.png, swa_vs_best.png, confusion_matrix.png).
