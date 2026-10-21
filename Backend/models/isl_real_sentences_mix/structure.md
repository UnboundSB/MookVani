# Directory Structure: Backend/models/isl_real_sentences

Real continuous human sign recognition model (fine-tuned from the synthetic Conformer-CTC by `training/train_real_sentences.py`).

## Files
- `real_sentence_best.pth`: Best-val-WER weights (plain state dict).
- `real_sentence_ema_best.pth`: Best EMA shadow weights.
- `real_sentence_swa.pth`: Stochastic-weight-averaged weights over the tail epochs.
- `real_sentence_checkpoint.pth`: Latest full training state (model, EMA, optimizer, history) for `--resume`.
- `history.json`: Per-epoch metrics (CTC loss, WER, S/D/I, blank %, escape count).

## Immediate Subdirectories
- `plots/`: Training curves (`real_sentence_training_curves.png`).
