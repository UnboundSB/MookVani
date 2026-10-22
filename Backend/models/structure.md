# Directory Structure: Backend/models

Neural network architectures (Conformer, ISL models) and organized model checkpoint repositories.

## Files
- `__init__.py`: Package initialization for the models module.
- `conformer.py`: Core PyTorch Conformer layer, multi-head self-attention, and convolution module definitions.
- `isl_conformer.py`: Domain-tailored Indian Sign Language Conformer architecture.
- `isl_sentence_model.py`: End-to-end continuous sentence recognition model using CTC loss.
- `structure.md`: Directory structure documentation for models.

## Immediate Subdirectories
- `pretrained_model/`: Weights, class definitions, metrics history, and plots for the Transformer + TCNN pose classifier.
- `pretrained_phase_2_model/`: Phase 2 checkpoint (BiGRU→Transformer→TCNN, curriculum FPS), classes.json, history.json, classification report, and plots.
- `phase3_word_model/`: Phase 3 fine-tuned word model (classes.json + history/classification report/plots written at the end of a run). **Currently empty apart from its class list — it has to be re-trained, because the vocabulary changed.**
- `phase4_word_model/`: Phase 4 full-FPS fine-tuned model. No frame clipping; random time-stretch/mirror/tilt augmentation. 50 epochs: transformer frozen for first 30, then full training. Saves best/final checkpoints, classes.json, history.json, classification report, and plots/.
- `phase5_word_model/`: Phase 5 maximum accuracy squeeze model reusing exact Phase 4 split. Saves best/final/swa checkpoints, classes.json, history.json, classification report, and plots/.
- `phase6_word_model/`: Phase 6 high-precision squeeze model breaking out of local minima with Lookahead, decoupled AdamW, Annealed + ReduceLROnPlateau scheduler, Model EMA, and SpecAugment frame masking. Saves best standard, best EMA, final checkpoints, classes.json, history.json, classification report, and plots/.
- `word_vocabulary/`: **The published vocabulary** (`classes.json`) — the 228 top-most frequent word classes every trainer and the live inference path use.
- `islsentence_synthetic/`: Checkpoints, class definitions, and training history for synthetic sentence models.
- `sentence_ctc/`: Checkpoints and artifacts for CTC continuous sentence recognition.
- `word_reconformer_135/`: Weights, class mappings, and plots for the 135-class word model.
- `word_reconformer_229/`: Weights, class mappings, and metrics for the 229-class balanced word model.
- `word_reconformer_264/`: Final checkpoints and classification report for the 264-class word model (the warm start of the word trainer).
- `sentence_level/`: Legacy sentence-level checkpoint repository.
- `word_level/`: Legacy word-level checkpoint repository.
- `plots/`: Diagnostic training plots and architecture diagnostic visualizations.
