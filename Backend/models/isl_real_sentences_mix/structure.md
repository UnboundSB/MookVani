# Directory Structure: Backend/models/isl_real_sentences_mix

Real continuous human sign recognition model trained with the three-source mix
(isolated word clips + synthetic sentences + real continuous sentences) via
`training/train_real_sentences.py --mix 1`.

The mixture balances the pronoun-skewed continuous corpus with uniform word and
synthetic sources, lifting the effective support of the label prior.

## Files
- `real_sentence_best.pth`: Best-val-WER weights (plain state dict).
- `real_sentence_ema_best.pth`: Best EMA shadow weights.
- `real_sentence_checkpoint.pth`: Latest full training state (model, EMA, optimizer, history) for `--resume`.
- `history.json`: Per-epoch metrics (CTC loss, WER, S/D/I, blank %, escape count).
- `structure.md`: Directory structure documentation.

## Immediate Subdirectories
- `plots/`: Training curves (`real_sentence_training_curves.png`).
