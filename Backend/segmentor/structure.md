# Directory Structure: Backend/segmentor

YAP (You Adapt Progressively) segmentor development workspace. Contains the canonical
`yap_conformer.py`, a noise augmentation module, and the two-stage training pipeline
for a noise-robust gesture segmenter built on YAP + YAPpingConformer.

## Files

- `yap_conformer.py`: Copy of the protected root `yap_conformer.py`. ONE FILE containing
  PoseCleaner, FeatureExpander, CausalConvStem, MemoryLayer, HiPPOLegS, SegmentTracker,
  YAPCore, YAP, YAPStream, all losses, YAPpingConformer, and smoke tests. Run directly
  to verify: `python yap_conformer.py` → ALL TESTS PASSED.
- `noise_aug.py`: MediaPipe noise augmentation for Stage 1 training. Applies jitter,
  per-joint dropout, whole-part dropout, hand swap, depth noise, speed warp, and idle
  padding to clean word-level clips to produce realistic corrupted versions.
- `train_noise_filter.py`: **Stage 1** — Noise filtration training. Two YAPpingConformer
  instances loaded simultaneously: teacher (clean path) and student (noisy path). Student
  is trained to produce outputs consistent with the teacher via KL divergence on
  presence/end heads and MSE on entity vectors. Teacher is updated as EMA of student.
  Anti-collapse: word-label cross-entropy on clean path entities. Saves `yap_noise_robust.pth`.
- `train_segmentor.py`: **Stage 2** — Synthetic sentence segmentation. Loads Stage 1
  weights and fine-tunes on synthetically stitched sentences with known boundaries.
  Presence head: focal BCE. End head: Gaussian-bump BCE. Deep supervision: layer_conf_loss.
  Conformer: CTC on entity stream. Regularisers: soft_count_loss, min_gap_penalty,
  dip_prior_loss. Uses the same YAPStream queue path (offline equivalent) so live == offline.
  Saves `yap_seg_best.pth` for Stage 3 real-sentence fine-tuning.
- `structure.md`: This file.

## Immediate Subdirectories

- `models/`: Training outputs. `yap_stage1/` — Stage 1 noise-robust weights (`yap_noise_robust.pth`). `yap_stage2/` — Stage 2 segmentation weights (`yap_seg_best.pth`).
