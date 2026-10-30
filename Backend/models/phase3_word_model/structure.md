# Directory Structure: Backend/models/phase3_word_model

Phase 3 fine-tuned word model (classes.json + history/classification report/plots written at the end of a run).
Fine-tunes from the Phase-2 checkpoint, replaces the classifier head for the current
class count, freezes the backbone for 5 epochs then unfreezes with differential LR.

**Currently empty apart from its class list — it has to be re-trained, because the
vocabulary changed.**

## Files
- `best_phase3_model.pth`: Checkpoint with highest validation accuracy (when trained).
- `final_phase3_model.pth`: Final trained weights at completion of training epochs.
- `classes.json`: List of word vocabulary classes.
- `history.json`: Per-epoch metrics history.
- `classification_report.txt`: Classification report per sign class.
- `predictions.json`: Validation (true, predicted) pairs (when trained).
- `structure.md`: Directory structure documentation.

## Immediate Subdirectories
- `plots/`: Contains evaluation figures (acc_vs_loss.png, metrics.png, fps_schedule.png, confusion_matrix.png).
