# Directory Structure: Backend/segmentor

YAP (You Adapt Progressively) gesture-segmentor workspace: the `yap_conformer.py`
model, a MediaPipe noise augmenter, and the two-stage training pipeline for a
noise-robust segmenter built on YAP + YAPpingConformer.

## Files

- `yap_conformer.py`: The whole model in ONE file — PoseCleaner, FeatureExpander,
  CausalConvStem, MemoryLayer, HiPPOLegS pooling, SegmentTracker, YAPCore, YAP,
  YAPStream, all losses, YAPpingConformer, plus its own smoke tests.
  Verify with `python yap_conformer.py` → ALL TESTS PASSED.
  Math fixes applied: the pooling is now the **exact Legendre projection** of the
  segment (was a wrong HiPPO operator: 50–130% off the analytic projection), the
  palm frame is image-plane only so MediaPipe's noisy `z` cannot corrupt the
  orientation or hide the edge-on condition, `min_gap_penalty` checks every
  frame, `dip_prior_loss` normalises per sample, the memory rate is clamped, and
  `timestamps_from_fps()` was added for data stored without timing.
- `noise_aug.py`: MediaPipe noise augmentation for Stage 1 — jitter, per-joint and
  whole-part dropout, hand swap, depth noise, speed warp and idle padding applied
  to clean word-level clips.
- `train_noise_filter.py`: **Stage 1** — noise-robustness training. Teacher (clean
  path) and student (noisy path) YAPpingConformer in lockstep; the student is
  trained to match the teacher (KL on the presence/end heads, MSE on the entity
  vectors) while also predicting the word label to prevent collapse. Teacher is an
  EMA of the student. Saves `yap_noise_robust.pth`.
- `train_segmentor.py`: **Stage 2** — segmentation training on synthetically
  stitched sentences with known boundaries. Focal-BCE presence head, Gaussian-bump
  end head, deep supervision, CTC on the entity stream, and the soft-count /
  min-gap / dip-prior regularisers. Saves `yap_seg_best.pth`.
- `structure.md`: This file.

## Immediate Subdirectories

- `__pycache__`: Python bytecode cache (generated, ignored).
