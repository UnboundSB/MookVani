# Directory Structure: Backend/models/pretrained_phase_2_model

Phase 2 checkpoint (BiGRU→Transformer→TCNN, curriculum FPS), class definitions,
metrics history, and plots for the word-level Reconformer fine-tuned from the
pretrained Transformer+TCNN backbone.

## Files
- `best_phase2_model.pth`: Checkpoint with highest validation accuracy.
- `final_phase2_model.pth`: Final trained weights at completion of training epochs.
- `classes.json`: List of word vocabulary classes (228 classes from the canonical split).
- `history.json`: Per-epoch metrics history (loss, accuracy, precision, recall, F1-score).
- `classification_report.txt`: Full precision, recall, and F1-score classification report per sign class.
- `predictions.json`: Validation (true, predicted) pairs for the confusion matrix.
- `split_paths.json`: Canonical train/val split manifest (shared with phases 3-6).
- `structure.md`: Directory structure documentation.

## Immediate Subdirectories
- `plots/`: Contains evaluation figures (acc_vs_loss.png, confusion_matrix.png, metrics.png, fps_schedule.png).
