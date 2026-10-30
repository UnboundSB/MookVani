# Directory Structure: Backend/segmentor

YAP (You Adapt Progressively) gesture-segmentor workspace: the `yap_conformer.py`
model, a MediaPipe noise augmenter, per-word dataset builders, and the training
pipeline for a noise-robust gesture segmentor built on YAP + YAPpingConformer.

## CRITICAL: YAP is a gesture SEGMENTOR, not a word classifier

`yap_conformer.py` contains **two models in one file**:

1. **YAP** (the segmentor front-end) — watches a live stream of poses, decides
   WHERE one gesture starts and ends ("segmenting"), and squashes each segment
   into ONE fixed-size vector (an "entity"). Outputs per-pose `presence`
   (is a gesture happening?) and `end` (does a gesture end here?) logits.
   YAP does NOT recognise words — it only finds boundaries and packages segments.
   YAP has **no word-classification head and no CTC output**.

2. **YAPpingConformer** — wraps YAP, runs the entity stream through a Conformer
   (transformer + conv blocks), and produces per-entity `word_logits` (class 0
   = blank, classes 1..N = words). This is the CTC sentence model.

**The Conformer/word-classification part is NOT YAP.** If you load YAP and try
to train it on CTC word-classification tasks, it will fail because YAP has no
word head — it only has presence/end heads. **Always use `YAPpingConformer` for
CTC / word-training tasks.** YAP alone is only for segmentation (Stage 1 noise
filtration, Stage 2A entity matching, Stage 2B boundary detection).

## The training order

**Stage 1 (`train_noise_filter.py`) has to be trained FIRST** — it teaches YAP's
presence head to distinguish real ISL word signs from noise/garbage, with every
other layer frozen. Stage 2 then trains boundary/end detection, and stage 3
fine-tunes on real sentences.

| stage | script | needs | produces |
|---|---|---|---|
| 0 — word dataset | `build_word_dataset.py` | `segmentor/models/yap_stage1/yap_noise_robust.pth` | `segmentor/data/<word>/<take>/` + `mapping.json` |
| 1A — presence only | `presence_trainer.py` | word clips + on-the-fly junk | `yap_presence_best.pth` (presence head only) |
| 1 — noise filtration | `train_noise_filter.py` | word clips + noise clips | `yap_noise_robust.pth` |
| 2A — word-entity matching | `train_segmentor.py --phase A` | stage 1's weights + `segmentor/data/` | `yap_phaseA_best.pth` + `word_head_phaseA_final.pth` |
| 2B — synthetic segmentation | `train_segmentor.py --phase B` | stage 2A's weights + `.npy` word corpus | `yap_seg_best.pth` |
| 3 — real-sentence fine-tune | `../training/train_real_sentences.py --init` | stage 2's weights | the live sentence model |

**Phase A must come before Phase B.** Phase A trains entity→word mapping on
isolated word clips. Phase B trains segmentation + Conformer + CTC on stitched
synthetic sentences.

## Files

- `yap_conformer.py`: **The whole model in ONE file** — PoseCleaner, FeatureExpander,
  CausalConvStem, MemoryLayer, HiPPOLegS pooling, SegmentTracker, YAPCore, YAP,
  YAPStream, YAPpingConformer, all losses, plus smoke tests.
  Verify with `python yap_conformer.py` → ALL TESTS PASSED.
  **Two models in this file:**
  - `YAP` = gesture segmentor (presence + end heads only, NO word classifier).
  - `YAPpingConformer` = YAP + Conformer + word classification head (CTC sentence model).
  **Do not train YAP on CTC tasks — it has no word head. Use `YAPpingConformer`.**
- `presence_trainer.py`: **Standalone presence trainer.** Trains ONLY
  `yap.core.presence_head` (everything else frozen). Mixes word clips
  (presence=1 everywhere) with on-the-fly random junk clips (presence=0
  everywhere, 60–180 frames of garbage coordinates). Teaches YAP: "is this a
  real sign or just noise?" Saves `yap_presence_best.pth`.
- `train_noise_filter.py`: **Stage 1 noise filtration.** Can run in two modes:
  (a) default — teacher/student consistency on presence + end heads with
  augmentation + noise; (b) presence-only — freezes everything except the
  presence head, trains on word vs junk clips. Saves `yap_noise_robust.pth`.
  Loads `.pt` word tensors directly (no `.npy` conversion needed).
- `noise_aug.py`: MediaPipe noise augmentation — jitter, per-joint and
  whole-part dropout, hand swap, depth noise, speed warp and idle padding.
- `train_segmentor.py`: **Stage 2** — two phases:
  - **Phase A** (`--phase A`): word-clip entity matching. Word classification
    head + cosine similarity to `processed.pt` target. **No Conformer.**
  - **Phase B** (`--phase B`): synthetic sentence segmentation + CTC. Trains
    the Conformer + word head on stitched sentences.
- `build_word_dataset.py`: Builds the per-word processed dataset that Phase A
  trains on. Creates `segmentor/data/<word>/<take>/word.pt` + `processed.pt`
  + `mapping.json`.
- `verify_no_collapse.py`: Standalone audit — runs YAP on 30 real ISL-CSLRT
  clips and checks output is not NULL/zero-collapsed.
- `structure.md`: This file.

## Immediate Subdirectories

- `data_builder/`: Curation pipeline — sentence→word segmentation tooling.
  - `opencv_app.py`: OpenCV desktop segmentation UI (fullscreen video playback,
    SPACE = mark boundary, R = repeat, Q = quit).
  - `builder.py`: Backend logic — hashing, scanning, matching, composing, CSV
    persistence, checkpoint/resume.
  - `sentences.csv`: 1,003 sentences filtered to top-228 words, all with word
    videos.
  - `word_videos.csv`: 1,509 word→video entries (all `archive_extracted`).
  - `word_inventory.json`: 841 words with video paths from archive + isign.
  - `top228_words.json`: 228 most frequent words (matches
    `Backend/models/word_vocabulary/classes.json`).
  - `segmentation.csv`: Output CSV (empty, auto-created on first run).
  - `checkpoint.json`: Resume state (deleted on fresh start).
  - `noise_clips/`: 500 synthetic noise clips (250 all-blank + 250 random
    garbage) + `index.json`. Used by presence trainers.
  - `rebuild_top228.py`: Script to rebuild sentences.csv / word_videos.csv
    filtered to top-228 words.
- `models/`: Training outputs. `yap_stage1/` — Stage 1 weights
  (`yap_noise_robust.pth`, `yap_presence_best.pth`). `yap_stage2/` —
  Stage 2 weights (`yap_phaseA_best.pth`, `yap_seg_best.pth`,
  `yap_seg_ckpt.pth`).
- `data/`: Per-word processed dataset built by `build_word_dataset.py`.
  `<word>/<take>/word.pt` + `processed.pt`, plus `mapping.json`. Generated,
  not committed.
- `yap_live_capture/`: Live segmentation capture sessions (JSON logs).
- `__pycache__`: Python bytecode cache (generated, ignored).
