# Directory Structure: Backend/training

Model training and evaluation scripts for word-level and sentence-level sign recognition.

## Files
- `evaluate_fresh_word_model.py`: Evaluates newly trained word Conformer models on validation splits.
- `evaluate_reconformer.py`: Generic evaluation script for ReConformer model architectures.
- `evaluate_synthetic_sentences.py`: Calculates Word Error Rate (WER) and loss metrics on synthetic sentence data.
- `evaluate_word_135.py`: Evaluates the 135-class isolated word model.
- `finetune_word.py`: Fine-tunes pre-trained word models with customized hyperparameters.
- `generate_training_plot.py`: Visualizes training and validation curves from metric histories.
- `phase2_trainer.py`: Phase 2 fine-tuning script — loads pretrained Transformer+TCNN backbone into the ISLSentenceReconformer (BiGRU→Transformer→TCNN) for word-level classification with curriculum FPS (1→25fps). Saves to `models/pretrained_phase_2_model/`.
- `phase3_trainer.py`: Phase 3 word-level fine-tuning from Phase-2 checkpoint. Replaces head for new class count, freezes backbone for 5 epochs then unfreezes with differential LR, curriculum FPS, augmentation after split, 14-sample cap per class. Saves to `models/phase3_word_model/`.
- `phase4_trainer.py`: Phase 4 full-FPS fine-tuning from Phase-3 checkpoint. No frame clipping. Augmentation: random time-stretch, horizontal flip, tilt (applied after train/val split). 50 epochs: transformer frozen for first 30 (CosineAnnealingWarmRestarts on RNN+CNN+head), then all layers train for final 20 (CosineAnnealing). Saves to `models/phase4_word_model/`.
- `phase5_trainer.py`: Phase 5 maximum accuracy squeeze reusing exact Phase 4 split (loaded from `split_paths.json`). All layers unfrozen, OneCycleLR with SWA. Saves to `models/phase5_word_model/`.
- `phase6_trainer.py`: Phase 6 high-precision squeeze trainer reusing exact Phase 4/5 split. Lookahead wrapping decoupled AdamW, Annealed LR combined with ReduceLROnPlateau, Model EMA, SpecAugment temporal frame masking, and gentle mixup. Saves to `models/phase6_word_model/`.
- `pretrainer.py`: Transformer + Temporal CNN (TCNN) pretraining script training on MediaPipe pose sequences with median length normalization and geometric augmentations.
- `structure.md`: Directory structure documentation for training.
- `train_reconformer_word.py`: Primary training script for ReConformer word-level models.
- `train_synthetic.py`: Trains sign recognition models using synthetic pose sequences.
- `train_synthetic_sentences.py`: Trains continuous sentence models using Connectionist Temporal Classification (CTC).
- `train_weak_classes.py`: Implements class re-weighting and focused training on underrepresented sign classes.
- `train_word_135.py`: Specialized training routine for the 135-class isolated sign vocabulary.

## Immediate Subdirectories
- *(None)*
