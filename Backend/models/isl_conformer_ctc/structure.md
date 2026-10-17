# Directory Structure: Backend/models/isl_conformer_ctc

ISL Hybrid Conformer-CTC continuous sign recognition model.
Pretrained Phase 6 Backbone (InputProj + BiGRU + Transformer + TCNN) + Conformer Contextualizer + CTC Regularization.

## Files
- `classes.json`: Vocabulary and CTC blank ID.
- `stageN_best.pth`: Best standard model weights per stage.
- `stageN_ema_best.pth`: Best EMA model weights per stage.
- `stageN_checkpoint.pth`: Full training state per stage.
- `stageN_final.pth`: Final weights at end of each stage.
- `stageN_history.json`: Per-epoch metrics for each stage.
- `structure.md`: This file.

## Subdirectories
- `plots/`: Metric dashboards and curves per stage.

## Stages
- Stage 1 (20 ep): Single-word CTC warm-up (backbone frozen, trains CTC alignment)
- Stage 2 (25 ep): 2-word short sentences (backbone unfrozen with differential LR)
- Stage 3 (40 ep): N-word progressive curriculum (1→10 words)
- Stage 4 (30 ep): Full unfrozen polish + EMA
