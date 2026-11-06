# Directory Structure: Backend/segmentor

YAP (You Adapt Progressively) gesture-segmentor workspace: the `yap_conformer.py`
model, the modular `yapv3/` training pipeline, per-word dataset builders, and the
training pipeline for a noise-robust gesture segmentor built on YAP + YAPpingConformer.

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

**Stage 1 must be trained FIRST** — it teaches YAP's presence head to distinguish
real ISL word signs from noise/garbage, with every other layer frozen. Stage 2
then trains boundary/end detection, and Stage 3 fine-tunes on real sentences.

| stage | script | needs | produces |
|---|---|---|---|
| 0 — word dataset | `build_word_dataset.py` | `models/yap_stage1/yap_noise_robust.pth` | `data/<word>/<take>/` + `mapping.json` |
| 1 — presence only | `presence_trainer.py` | word clips + on-the-fly junk | `yap_presence_best.pth` (presence head only) |
| 1 — boundary/end | `boundary_trainer.py` | word clips + on-the-fly junk | `yap_end_best.pth` (end head only) |
| 2A — word-entity matching | `train_segmentor.py --phase A` | stage 1 weights + `data/` | `yap_phaseA_best.pth` |
| 2B — synthetic segmentation | `train_segmentor.py --phase B` | stage 2A weights + word corpus | `yap_seg_best.pth` |
| 3 — real-sentence fine-tune | `yapv3/train_ctc_sentences.py` | stage 2 weights + `data_builder/sentences.csv` | `models/yap_ctc_sentences/` |
| 3 — real-sentence fine-tune | `yapv3/train_phase3.py` | stage 1/2 weights + word tensors | `models/yap_phase3/` |

**Phase A must come before Phase B.** Phase A trains entity→word mapping on
isolated word clips. Phase B trains segmentation + Conformer + CTC on stitched
synthetic sentences. Stage 3 fine-tunes the Conformer + CTC head on real
sentences from `data_builder/sentences.csv`.

## Files (training pipeline — modularized)

### Root-level scripts

- `yap_conformer.py`: **The whole model in ONE file** (~120 KB) — PoseCleaner,
  FeatureExpander, CausalConvStem, MemoryLayer, HiPPOLegS pooling, SegmentTracker,
  YAPCore, YAP, YAPStream, YAPpingConformer, all losses, plus smoke tests.
  Verify with `python yap_conformer.py` → ALL TESTS PASSED.
  **Two models in this file:** `YAP` (segmentor, presence + end only) and
  `YAPpingConformer` (YAP + Conformer + word head).
- `train_segmentor.py`: **Stage 2** — two phases: `--phase A` (word-clip entity
  matching, no Conformer) and `--phase B` (synthetic sentence segmentation +
  CTC). Includes per-epoch loss logging with NaN safeguard (P0-2).
- `segmentor_fine_tune_v4.py`: Stage 4 fine-tune entry point (legacy v4).
- `segmentor_fine_tune_v5.py`: Thin 35-line shim that re-exports the `segmentor_v5` package.
- `presence_trainer.py`: Standalone presence trainer (presence head only).
- `boundary_trainer.py`: Boundary / end-head-only trainer.
- `train_word_ctc.py`: CLI entry point for word-level CTC training (uses `word_ctc/` package).
- `train_word_presence.py`: Presence-head trainer driven by word/noise clips (.npy loader).
- `noise_aug.py`: MediaPipe noise augmentation (jitter, dropout, hand swap, depth noise, speed warp, idle padding).
- `build_word_dataset.py`: Builds `data/<word>/<take>/word.pt` + `processed.pt` + `mapping.json`.
- `build_sentence_dataset.py`: Per-sentence processed dataset for YAPpingConformer fine-tuning. Runs Yap+Conformer on each real sentence tensor, detects segments via SegmentTracker, shows boundaries for accept/reject, saves accepted sentences with entity vectors + segments.json. Supports --interactive or --auto-accept. Output: data_sentence/<slug>/<take>/.
- `verify_no_collapse.py`: Audit that YAP output is not NULL/zero-collapsed (standalone, not integrated into training — P0-4).
- `test.py`, `tester.py`, `tester_v2.py`, `test_segmentor.py`, `test_stage2_fixes.py`, `test_stage2_verify.py`: inference / test scripts.
- `yap_live_first_cut.py`: Live webcam tester that freezes after first detected segment.
- `display_segments.py`: Segment visualization.
- `audit_accuracy.py`, `audit_collapse_hardcore.py`: Post-training audit scripts.
- `class_probe.py`: Class-level activation probing.
- `structure.md`: This file.

### yapv3/ — modular v3 training pipeline (new, recommended)

- `yapv3/__init__.py`: Package marker.
- `yapv3/utils/__init__.py`: Public API re-exports.
- `yapv3/utils/data.py`: Word-clip discovery, sentence stitching (2-5 random words per sentence),
  augmentations (mirror, shift, tilt, Gaussian time-warp), zero-coord hold-last-valid,
  `SentenceClipDataset`, `collate_sentence_batch`. All clips preloaded into RAM.
- `yapv3/utils/checker.py`: Universal collapse checker — `run_universal_check()`, `check_model_forward()`, `check_history()`. Works across YAP presence, boundary, segmentation, and Conformer+CTC models. Detects collapse from forward-pass output variance/blank-ratio AND training history (NaN losses, loss divergence, accuracy near random, boundary F1=0).
- `yapv3/utils/ctc_sentence_data.py`: Real-sentence dataset — matches `sentences.csv` to `tensors_sentence_level/index.json`, remaps 262-class → 228-class indices by word name, generates frame-level boundary targets from word positions.
  - **Also loads `data_builder/segmentation_clean.csv`** — human-annotated boundary marks (frame_index) as real ground truth for boundary loss.
  - `load_segmentation_marks()`, `build_segmentation_boundary_items()`, `collate_segmentation_batch()`.
- `yapv3/utils/ctc_sentence_trainer.py`: Conformer + CTC training loop — `AntiCollapseCTCLoss` (CTC + emission barrier + blank cap), frozen backbone, greedy decode + WER metrics, boundary detection loss (end head BCE vs real frame-level targets from segmentation_clean.csv).
  - Accepts `segmentation_csv` parameter for human-annotated boundary validation.
  - Pretrainer-aligned scheduler (cosine warm restarts, plateau escape).
- `yapv3/utils/synthetic_sentence_data.py`: *(new)* Synthetic sentence data loader for Conformer+CTC pretraining.
  - Word 0 used in full; words 1..N are cut at (first_motion_frame + random 5-7 frames) so hands flow continuously without resetting to rest between signs.
  - The whole sentence is then Gaussian time-warped (random accel/decel) for relative-pace segmentation learning.
  - `SyntheticSentenceDataset`, `collate_synthetic_batch`, `build_synthetic_dataset`.
- `yapv3/utils/pretrainer_config.py`: *(new)* YAPpingConformer configured per `Backend/training/pretrainer.py` specs.
  - d_model=128, heads=4, layers=2, ff_mult=2, conv_kernel=3, dropout=0.1
  - lr=1.5e-3, weight_decay=1e-4, label_smoothing=0.08, batch_size=32, grad_clip=2.0
  - Cosine warm restart every 10 epochs, plateau escape patience=8, scale=0.02
  - `PretrainerYAPConfig`, `build_yapping_conformer_from_pretrained()`, `load_pretrained_weights_into_yap()`.
- `yapv3/utils/losses.py`: Presence (focal BCE), end (weighted BCE), layer-conf (deep supervision),
  soft-count (weak supervision on gesture count), `total_segmentation_loss()` with NaN safeguard.
- `yapv3/utils/model.py`: YAP model loading/wrapping — handles `{'yap': ...}`, `{'model': ...}`,
  `{'state_dict': ...}` and raw state_dict checkpoint schemas; strips `module.` / `yap.` prefixes.
- `yapv3/utils/trainer.py`: Phase-3 training loop — trains presence + end heads (everything else frozen),
  validates with production SegmentTracker boundary F1 / precision / recall / IoU / count MAE.
- `yapv3/train_phase3.py`: CLI entry point for Phase-3 gesture-segmentation pretraining on stitched word-sentences.
- `yapv3/train_phase4.py`: CLI entry point for Phase-4 real-sentence Conformer + CTC fine-tuning.
  - Supports `--use-pretrainer-config` to apply all pretrainer.py hyperparameters.
  - Boundary detection loss (end head BCE vs frame-level targets from sentences.csv).

### yapv3/ — real-sentence Conformer + CTC fine-tune (P0-3, now implemented)

- `yapv3/utils/ctc_sentence_data.py`: Real-sentence dataset from `data_builder/sentences.csv`.
- `yapv3/utils/ctc_sentence_trainer.py`: Conformer + CTC training with `AntiCollapseCTCLoss`,
  pretrainer-aligned scheduler (cosine warm restarts, plateau escape), boundary detection loss.
- `yapv3/utils/pretrainer_config.py`: YAPpingConformer configured per `Backend/training/pretrainer.py`.
- `yapv3/train_ctc_sentences.py`: CLI entry point for real-sentence fine-tuning.
- `yapv3/train_phase4.py`: Phase-4 entry point with `--use-pretrainer-config` flag.

### yap_v6/ — separate modular v6 implementation

`config.py`, `data.py`, `extractor.py`, `model.py`, `losses.py`,
`augmentation/`, `data/`, `evaluation/`, `training/`, `train_v6.py`.

### segmentor_v5/ — separate modular v5 implementation

`__init__.py`, `augment.py`, `boundaries.py`, `cache.py`, `config.py`,
`datasets.py`, `ema.py`, `extractor.py`, `losses.py`, `main.py`, `train.py`.

### word_ctc/ — word-level CTC package

`__init__.py`, `dataset.py`, `eval.py`, `loss.py`, `model.py`, `train.py`.
`ConformerCTC` model + `AntiCollapseCTCLoss` + `WordCTCDataset`.

### data_builder/ — curation pipeline (sentence→word segmentation tooling)

- `sentences.csv`: 1,003 sentences filtered to top-228 words, all with word videos.
  Columns: `sentence_id`, `sentence`, `sentence_hash`, `sentence_video`, `source`, `is_composed`.
  Video paths are absolute Windows paths pointing to `ISL_CSLRT_Corpus/Videos_Sentence_Level/`.
- `isign_sentences.csv`: iSign sentence corpus with `sentence_id`, `sentence`.
- `word_videos.csv`: 1,509 word→video entries (all `archive_extracted`).
- `word_inventory.json`: 841 words with video paths from archive + isign.
- `top228_words.json`: 228 most frequent words (matches `Backend/models/word_vocabulary/classes.json`).
- `segmentation.csv`: Output CSV (empty, auto-created on first run).
- `checkpoint.json`: Resume state (deleted on fresh start).
- `noise_clips/`: 500 synthetic noise clips (250 all-blank + 250 random garbage) + `index.json`.
- `rebuild_top228.py`: Script to rebuild `sentences.csv` / `word_videos.csv` filtered to top-228 words.

### models/ — training outputs

- `yap_stage1/`: Stage 1 weights (`yap_noise_robust.pth`, `yap_noise_ckpt.pth`, `history.json`).
- `yap_stage1_presence/`: Standalone presence trainer weights (`yap_noise_robust.pth`, `yap_noise_ckpt.pth`).
- `yap_stage2/`: Stage 2 weights (`yap_phaseA_best.pth`, `yap_phaseA_final.pth`, `phaseA_history.json`, `stage2.err`).
- `yap_segmentation_only_v1/`: Stage 2 v1 weights (`yap_phaseA_best.pth`, `history.json`, `source_split.json`).
- `yap_segmentation_f1_v2/`: Stage 2 v2 weights (`yap_phaseA_best.pth`, `history.json`, `source_split.json`).
- `yap_segmentation_f1_v3/`: Stage 2 v3 — `source_split.json` only (no weights, no history).
- `yap_presence_only/`: Presence-only weights (`yap_presence_best.pth`).
- `word_ctc/`: Word-level CTC weights (`best_model.pth`).
- `yap_phase3/`: Phase-3 gesture-segmentation weights (created by `yapv3/train_phase3.py`).
- `yap_ctc_sentences/`: Real-sentence Conformer+CTC weights (created by `yapv3/train_ctc_sentences.py`).

### data/ — per-word processed dataset

Built by `build_word_dataset.py`. `<word>/<take>/word.pt` + `processed.pt`, plus `mapping.json`.
Generated, not committed.

### Other subdirectories

- `features_cache/`: NPZ feature cache (generated, ignored).
- `__pycache__/`: Python bytecode cache (generated, ignored).
- `yap_live_capture/`: Live segmentation capture sessions (JSON logs, .mp4 videos).
- `segmentation_viz/`: Segmented visualization videos (.mp4).

### Removed (redundant, never applied to source)

The following were deleted as redundant patch scripts that were never applied to their
target source files: `fix_dup.py`, `fix_dup_v2.py`, `fix_orphan.py`, `fix_orphan2.py`,
`fix_isign_orphan.py`, `fix_indent.py`, `fix_key_handling.py`, `fix_keys_v2.py`,
`fix_keys_v3.py`, `debug_replace.py`, `check_v3.py`, `check_yap_v4.py`, `check_yap_v5.py`.
