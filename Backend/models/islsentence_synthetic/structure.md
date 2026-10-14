# Directory Structure: Backend/models/islsentence_synthetic

Model weights, training logs, and evaluations for synthetic sentence recognition.

## Files
- `classes.json`: Vocabulary gloss mapping for synthetic sentence sequences.
- `history.json`: Training history tracking CTC loss and Word Error Rate (WER) per epoch.
- `phase1_sentence_best.pth`: Phase 1 CTC sentence model weights achieving lowest validation WER over frozen Phase 6 backbone.
- `phase1_sentence_checkpoint.pth`: Mid-training checkpoint preserving optimizer, CTC head, and epoch states.
- `phase1_sentence_final.pth`: Final trained weights at completion of 10-epoch Phase 1 sentence training.
- `reconformer_synthetic_sentence_best.pth`: Legacy model weights achieving the lowest validation loss / WER.
- `reconformer_synthetic_sentence_checkpoint.pth`: Legacy mid-training checkpoint preserving optimizer and epoch states.
- `reconformer_synthetic_sentence_final.pth`: Legacy final trained weights at completion of training.
- `structure.md`: Directory structure documentation for islsentence_synthetic.

## Immediate Subdirectories
- `plots/`: Metric curves and loss visualizations from synthetic sentence training runs.
