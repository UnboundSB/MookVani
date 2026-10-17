# Directory Structure: Backend/models/islsentence_synthetic_phase2

Model checkpoints, training history, and diagnostics for Phase 2 unfrozen continuous sentence recognition.

## Files
- `classes.json`: Vocabulary gloss mapping and CTC blank token index.
- `history.json`: Epoch-by-epoch loss, Word Error Rate (WER), exact match accuracy, F1, and learning rates.
- `phase2_sentence_best.pth`: Best checkpoint model weights achieving the lowest validation WER.
- `phase2_sentence_ema_best.pth`: Exponential Moving Average (EMA) shadow model weights.
- `phase2_sentence_checkpoint.pth`: Complete training state (optimizer, scheduler, epoch, EMA).
- `phase2_sentence_final.pth`: Final model weights after completing all training epochs.
- `structure.md`: Directory structure documentation for this phase directory.

## Immediate Subdirectories
- `plots/`: Metric dashboards, loss curves, and WER progression charts.
