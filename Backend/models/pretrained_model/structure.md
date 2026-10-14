# Directory Structure: Backend/models/pretrained_model

Model weights, class vocabulary, classification metrics, and training figures for the Transformer + TCNN MediaPipe pose pretrainer.

## Files
- `best_pretrained_model.pth`: Checkpoint with highest validation accuracy.
- `final_pretrained_model.pth`: Final trained weights at completion of training epochs.
- `classes.json`: List of all 114 sign vocabulary classes.
- `history.json`: Per-epoch metrics history (loss, accuracy, precision, recall, F1-score).
- `classification_report.txt`: Full precision, recall, and F1-score classification report per sign class.
- `structure.md`: Directory structure documentation.

## Immediate Subdirectories
- `plots/`: Contains evaluation figures (acc_vs_loss.png, confusion_matrix.png, metrics.png).
