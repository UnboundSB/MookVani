# Directory Structure: Backend/models

Neural network architectures (Conformer, ISL models) and organized model checkpoint repositories.

## Files
- `__init__.py`: Package initialization for the models module.
- `conformer.py`: Core PyTorch Conformer layer, multi-head self-attention, and convolution module definitions.
- `isl_conformer.py`: Domain-tailored Indian Sign Language Conformer architecture.
- `isl_sentence_model.py`: End-to-end continuous sentence recognition model using CTC loss.
- `structure.md`: Directory structure documentation for models.

## Immediate Subdirectories
- `pretrained_model/`: Weights, class definitions, metrics history, and plots for the Transformer + TCNN pose classifier. Has `structure.md`.
- `pretrained_phase_2_model/`: Phase 2 checkpoint (BiGRU→Transformer→TCNN, curriculum FPS), classes.json, history.json, classification report, and plots. **No structure.md.**
- `phase3_word_model/`: Phase 3 fine-tuned word model (classes.json + history/classification report/plots written at the end of a run). **Currently empty apart from its class list — it has to be re-trained, because the vocabulary changed. No structure.md.**
- `phase4_word_model/`: Phase 4 full-FPS fine-tuned model. No frame clipping; random time-stretch/mirror/tilt augmentation. 50 epochs: transformer frozen for first 30, then full training. Saves best/final checkpoints, classes.json, history.json, classification report, and plots/. **No structure.md.**
- `phase5_word_model/`: Phase 5 maximum accuracy squeeze model reusing exact Phase 4 split. Saves best/final/swa checkpoints, classes.json, history.json, classification report, and plots/. **No structure.md.**
- `phase6_word_model/`: Phase 6 high-precision squeeze model breaking out of local minima with Lookahead, decoupled AdamW, Annealed + ReduceLROnPlateau scheduler, Model EMA, and SpecAugment frame masking. Saves best standard, best EMA, final checkpoints, classes.json, history.json, classification report, and plots/. **No structure.md.**
- `word_vocabulary/`: **The published vocabulary** (`classes.json`) — the 228 top-most frequent word classes every trainer and the live inference path use. Has `structure.md`.
- `islsentence_synthetic/`: Checkpoints, class definitions, and training history for synthetic sentence models. Has `structure.md`.
- `islsentence_synthetic_phase2/`: Phase 2+ unfrozen sentence model checkpoints and history. Has `structure.md`.
- `isl_conformer_ctc/`: ISL Hybrid Conformer-CTC continuous sign recognition model (stages 1-4). Has `structure.md`.
- `isl_real_sentences/`: Real continuous human sign recognition model (fine-tuned from synthetic Conformer-CTC). Has `structure.md`.
- `isl_real_sentences_phase2/`: Phase 2 real continuous sentence model. Has `structure.md`.
- `isl_real_final/`: Final regularized continuous model. Has `structure.md`.
- `isl_real_sentences_mix/`: Real sentence model trained with the three-source mix. **No structure.md.**
- `sentence_ctc/`: Checkpoints and artifacts for CTC continuous sentence recognition. Has `structure.md`.
- `word_reconformer_135/`: Weights, class mappings, and plots for the 135-class word model. Has `structure.md`.
- `word_reconformer_229/`: Weights, class mappings, and metrics for the 229-class balanced word model. Has `structure.md`.
- `word_reconformer_264/`: Final checkpoints and classification report for the 264-class word model. Has `structure.md`.
- `sentence_level/`: Legacy sentence-level checkpoint repository. **No structure.md.**
- `word_level/`: Legacy word-level checkpoint repository. **No structure.md.**
- `plots/`: Diagnostic training plots and architecture diagnostic visualizations.
