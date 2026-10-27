# Directory Structure: Backend/segmentor

YAP (You Adapt Progressively) gesture-segmentor workspace: the `yap_conformer.py`
model, a MediaPipe noise augmenter, a per-word dataset builder, and the two-stage
training pipeline for a noise-robust segmenter built on YAP + YAPpingConformer.

## The training order

**Stage 1 (`train_noise_filter.py`) has to be trained FIRST** — stage 2 initialises
from its weights, and stage 3 (the real-sentence fine-tune) initialises from
stage 2's.

| stage | script | needs | produces |
|---|---|---|---|
| 0 — word dataset | `build_word_dataset.py` | `segmentor/models/yap_stage1/yap_noise_robust.pth` | `segmentor/data/<word>/<take>/` + `mapping.json` |
| 1 — noise filtration | `train_noise_filter.py` | the word corpus in the **YAP format** (`.npy`, `(T, 42*3)`, the two hands) | `yap_noise_robust.pth` |
| 2A — word-entity matching | `train_segmentor.py --phase A` | stage 1's weights + `segmentor/data/` (processed.pt targets) | `yap_phaseA_best.pth` + `word_head_phaseA_final.pth` |
| 2B — synthetic segmentation | `train_segmentor.py --phase B` | stage 2A's weights + `.npy` word corpus | `yap_seg_best.pth` |
| 3 — real-sentence fine-tune | `../training/train_real_sentences.py --init` | stage 2's weights | the live sentence model |

**Phase A must come before Phase B.** Phase A trains the filteration + expansion +
entity→word mapping on isolated word clips with `processed.pt` targets (cosine
similarity + word CE). Phase B then trains segmentation + Conformer + CTC on
stitched synthetic sentences. The Conformer is **not** touched in Phase A.

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
                              (f[159], f[386], f[4], f[234],   (brows, eyes, mouth,
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
python -c "
import torch, numpy as np
from pathlib import Path
root = Path('data/tensors_include50_word_level')
for p in root.rglob('*.pt'):
    npy = p.with_suffix('.npy')
    if not npy.exists():
        np.save(npy, torch.load(p, weights_only=True).numpy().astype(np.float32))
"
python segmentor/train_noise_filter.py --data_dirs Backend/data/tensors_include50_word_level
```

## Files

- `yap_conformer.py`: The whole model in ONE file — PoseCleaner, FeatureExpander,
  CausalConvStem, MemoryLayer, HiPPOLegS pooling, SegmentTracker, YAPCore, YAP,
  YAPStream, all losses, YAPpingConformer, plus its own smoke tests.
  Verify with `python yap_conformer.py` → ALL TESTS PASSED.
  Known issues (in the committed version): the HiPPO-LegS `A` matrix diagonal
  uses `n+1` instead of the reference `-(2n+1)` (pool deviates from the analytic
  Legendre projection by 50–130%), the palm frame uses the noisy 3D `z`
  coordinate (13.4% of real ISL-CSLRT hand frames are edge-on but the fade-out
  never fires), and `min_gap_penalty` only checks `L-k+1` windows. These were
  fixed in a working branch but the file is currently at the committed revision.
- `noise_aug.py`: MediaPipe noise augmentation for Stage 1 — jitter, per-joint and
  whole-part dropout, hand swap, depth noise, speed warp and idle padding applied
  to clean word-level clips. **Every magnitude is scaled by a global `scale` factor**, so the corruption INTENSITY can be drawn at random for every batch.
- `train_noise_filter.py`: **Stage 1** — noise-robustness training. Teacher (clean
  path) and student (noisy path) YAPpingConformer in lockstep. The canonical
  augmentation (mirror / tilt / shift / speed) is applied ONCE to the clean clip, so
  the teacher and the student see the SAME augmented clip and the only difference
  between them is the noise. The student is trained to match the teacher (KL on the
  presence/end heads, MSE on the entity vectors) while also predicting the word
  label to prevent collapse. Teacher is an EMA of the student. Saves `yap_noise_robust.pth`.
  NOTE: loads `.npy` files; run the `.pt → .npy` conversion above first.
- `train_segmentor.py`: **Stage 2** — two phases:
  - **Phase A** (`--phase A`): word-clip entity matching. Runs each clip in
    `segmentor/data/<word>/<take>/` through YAP → one entity → word classification
    head + cosine similarity to the `processed.pt` target. Trains YAP's MemoryLayer,
    presence/end heads, lift, readout, ctx_vec, and the word head. **No Conformer.**
    Saves `yap_phaseA_best.pth` + `word_head_phaseA_final.pth`.
  - **Phase B** (`--phase B`): synthetic sentence segmentation + CTC. Stitched
    sentences with ground-truth spans; focal-BCE presence, Gaussian-bump end,
    deep supervision, soft-count / min-gap / dip-prior regularisers, and CTC on
    the entity stream. Trains the Conformer + word head. Saves `yap_seg_best.pth`.
    NOTE: loads `.npy` files; run the `.pt → .npy` conversion first.
- `build_word_dataset.py`: Builds the per-word processed dataset that Phase A
  trains on. Phase 1 (instant): creates `segmentor/data/<word>/<take>/word.pt`
  for every clip + `mapping.json`. Phase 2 (slow, ~4s/clip on CUDA, resumable):
  runs YAP Stage 1 front end on each clip → `processed.pt` entity vector. Includes
  `fill_missing_coords()` — prev/next linear interpolation for missing joints
  (max gap `max_gap_frames`, default 15).
- `verify_no_collapse.py`: Standalone audit script — runs YAP on 30 real ISL-CSLRT
  clips and checks that the output is not NULL/zero-collapsed. Verified: output
  has meaningful variance (cleaned pose std 2.41, zero-joint % matches raw input).
- `structure.md`: This file.

## Immediate Subdirectories

- `models/`: Training outputs. `yap_stage1/` — Stage 1 noise-robust weights
  (`yap_noise_robust.pth`). `yap_stage2/` — Stage 2 weights
  (`yap_phaseA_best.pth`, `yap_seg_best.pth`, `yap_seg_ckpt.pth`).
- `data/`: Per-word processed dataset built by `build_word_dataset.py`.
  `<word>/<take>/word.pt` + `processed.pt`, plus `mapping.json`. Generated, not
  committed.
- `__pycache__`: Python bytecode cache (generated, ignored).
