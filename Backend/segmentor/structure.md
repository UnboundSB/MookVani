# Directory Structure: Backend/segmentor

YAP (You Adapt Progressively) gesture-segmentor workspace: the `yap_conformer.py`
model, a MediaPipe noise augmenter, and the two-stage training pipeline for a
noise-robust segmenter built on YAP + YAPpingConformer.

## The training order
**Stage 1 (`train_noise_filter.py`) has to be trained FIRST** — stage 2 initialises
from its weights, and stage 3 (the real-sentence fine-tune) initialises from
stage 2's.

| stage | script | needs | produces |
|---|---|---|---|
| 1 — noise filtration | `train_noise_filter.py` | the word corpus in the **YAP format** (`.npy`, `(T, 42*3)`, the two hands) | `yap_noise_robust.pth` |
| 2 — synthetic segmentation | `train_segmentor.py` | stage 1's weights (`--init`) + the word corpus | `yap_seg_best.pth` |
| 3 — real-sentence fine-tune | `../training/train_real_sentences.py --init` | stage 2's weights | the live sentence model |

**The body layout was the blocker.** YAP was written against the FULL
MediaPipe point sets, but our corpus stores a subset of them:

    MediaPipe                 our corpus (163 dims)              YAP before
    ------------------------  ---------------------------  -----------------------
    HandLandmarker  21        left hand  21  (0:63)          left_hand  21
    HandLandmarker  21        right hand 21  (63:126)        right_hand 21
    PoseLandmarker  33        shoulders+elbows 4 (126:138)    upper_body  9
                              (p[11], p[12], p[13], p[14])   (shoulders, elbows,
                                                                WRISTS, HIPS, nose)
    FaceLandmarker  478       7 face points (138:159)         face       28
                              (f[159], f[386], f[4], f[234],  (brows, eyes, mouth,
                               f[454], f[61], f[291])          nose tip, chin)

So the corpus has **53** landmarks, the default YAP wanted 42 and the holistic
preset wanted 79 - neither matched.  `YAPConfig.corpus()` was added: the two
hands + the 4 body points + the 7 face points, in the corpus order (which IS
the YAP part order), with the two shoulders as the body frame.  The trainers
use it by default (`--layout corpus|holistic|hands`).

**The corpus format is the second blocker.** The segmentor reads `.npy` files of
`(T, J*3)` with `J = 42` (the two MediaPipe hands — the default YAP config),
while our word corpus is `.pt` tensors of `(T, 163)` (hands + 4 arm points +
7 face points + 4 presence flags). `load_clips()` scans for `*.npy` and would
find **nothing** → stage 1 would train on an empty dataset. Convert it first:

```powershell
python scripts/convert_word_corpus_for_yap.py
python segmentor/train_noise_filter.py --data_dirs Backend/data/yap_word_level
```

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
  to clean word-level clips. **Every magnitude is scaled by a global `scale` factor**,
  so the corruption INTENSITY can be drawn at random for every batch: the student
  then learns to be robust across the whole range of noise levels instead of only at
  the nominal one (`--noise_scale_min/--noise_scale_max/--noise_prob/--jitter_std`).
- `train_noise_filter.py`: **Stage 1** — noise-robustness training. Teacher (clean
  path) and student (noisy path) YAPpingConformer in lockstep. The canonical
  augmentation (mirror / tilt / shift / speed) is applied ONCE to the clean clip, so
  the teacher and the student see the SAME augmented clip and the only difference
  between them is the noise (otherwise the consistency loss would punish the student
  for the augmentation instead of for the noise). The student is
  trained to match the teacher (KL on the presence/end heads, MSE on the entity
  vectors) while also predicting the word label to prevent collapse. Teacher is an
  EMA of the student. Saves `yap_noise_robust.pth`.
- `train_segmentor.py`: **Stage 2** — segmentation training on synthetically
  stitched sentences with known boundaries. Focal-BCE presence head, Gaussian-bump
  end head, deep supervision, CTC on the entity stream, and the soft-count /
  min-gap / dip-prior regularisers. The canonical augmentation is applied to the
  synthetic sentences, but only its SPATIAL part (mirror / tilt / shift / scale):
  the temporal augmentations would move or delete frames and the ground-truth
  gesture boundaries would stop being the truth. Saves `yap_seg_best.pth`.
- `structure.md`: This file.

## Immediate Subdirectories

- `__pycache__`: Python bytecode cache (generated, ignored).
