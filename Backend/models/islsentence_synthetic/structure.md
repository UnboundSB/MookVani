# Directory Structure: Backend/models/islsentence_synthetic

Model weights, training logs, and evaluations for synthetic sentence recognition.

## Files
- `classes.json`: Vocabulary gloss mapping for synthetic sentence sequences.
- `history.json`: Training history tracking CTC loss and Word Error Rate (WER) per epoch.
- `reconformer_synthetic_sentence_best.pth`: Model weights achieving the lowest validation loss / WER.
- `reconformer_synthetic_sentence_checkpoint.pth`: Mid-training checkpoint preserving optimizer and epoch states.
- `reconformer_synthetic_sentence_final.pth`: Final trained weights at completion of training.
- `structure.md`: Directory structure documentation for islsentence_synthetic.

## Immediate Subdirectories
- `plots/`: Metric curves and loss visualizations from synthetic sentence training runs.
