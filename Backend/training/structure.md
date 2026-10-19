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
- `train_conformer_ctc.py`: Next-generation continuous sentence trainer with ISL Conformer-CTC (MHA + Depthwise Conv, no global pooling), 4-stage curriculum (single-word, 2-word, N-word progressive, unfrozen polish), and Blank Connectivity Regularization.
- `train_real_sentences.py`: Real continuous human sign sentence fine-tuner training Conformer-CTC on real continuous video streams with selective in-scope relabeling.
- `train_real_sentences_phase2.py`: Phase 2 continuous sentence fine-tuner incorporating CosineAnnealingWarmRestarts, SpecAugment temporal cutout, and class-balanced focal CTC loss to escape the 77% local minimum. Saves to `models/isl_real_sentences_phase2/`.
- `train_real_final.py`: Master regularized continuous trainer warm-starting from Phase 2 real-sentence weights with calibrated CTC duration deficit, SpecAugment temporal cutout, and zero-leakage prompt splits. Saves to `models/isl_real_final/`.
- `train_reconformer_word.py`: Primary training script for ReConformer word-level models.
- `train_synthetic.py`: Trains sign recognition models using synthetic pose sequences.
- `train_synthetic_sentences.py`: Phase 1 continuous sentence trainer using CTC loss. Generates synthetic sentences (1-10 words, no repetition, smooth co-articulation, uniform sentence-level flip/shift/tilt augmentations), loads all uncapped class samples, and trains the CTC head over a frozen Phase 6 backbone.
- `train_synthetic_sentences_phase2.py`: Phase 2+ unfrozen sentence trainer with dynamic phase auto-detection and cycling (automatically detects latest Phase K, initializes from its weights, and trains Phase K+1 into `islsentence_synthetic_phase{K+1}/`). Implements discriminative learning rates, Model EMA, Cosine Annealing, and regularized CTC loss.
- `train_weak_classes.py`: Implements class re-weighting and focused training on underrepresented sign classes.
- `train_word_135.py`: Specialized training routine for the 135-class isolated sign vocabulary.

## Immediate Subdirectories
- *(None)*
