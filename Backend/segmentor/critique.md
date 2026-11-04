# YAP Segmentor Pipeline Audit — Critique Only

**Scope:** All files under `Backend/segmentor/` that participate in training, data building, model definition, evaluation, and live inference for YAP ("You Adapt Progressively") gesture segmentor + YAPpingConformer word classifier.

**Rule:** This file is read-only critique. No existing source file is modified. Each entry lists a file and what needs to change. Collapse verification is included for existing trained models.

---

## 1. Executive Summary

YAP core = gesture PRESENCE detector (presence head: is a MediaPipe gesture happening right now?). YAPpingConformer wraps YAP with a Conformer + word head for CTC/sentence classification. Later YAP models (v2 segmentation line) are **presence detection tuned** for **gesture segmentation** — detecting MediaPipe gestures and segmenting word boundaries from the gesture stream. IOU is **not applicable** (not object detection).

The pipeline flow is: **presence detection → gesture segmentation → word segmentation from detected gestures**.

The pipeline has multiple parallel training entry points that diverge in config, loss terms, data layout, and checkpoint schemas. There is no single source of truth for stage ordering, no unified config, no cross-validation, and no automated collapse detection integrated into training loops.

**Critical finding: NO `.pth` weight files exist anywhere in `Backend/segmentor/models/`.** Only training logs, history JSONs, error logs, and source splits are present. No trained model weights are saved or committed. Every model directory is missing its actual weights.

Existing training logs show clear signs of collapse (NaN loss components, accuracy near zero, boundary F1 = 0.0 in Phase B models, loss divergence from negative to positive).

---

## 2. Model Architecture — `yap_conformer.py`

**Status:** Single monolithic file (~1000+ lines) containing PoseCleaner, FeatureExpander, CausalConvStem, MemoryLayer, HiPPOLegS, SegmentTracker, YAPCore, YAP, YAPStream, YAPpingConformer, all losses, targets, and smoke tests.

**What needs to change:**

| Item | Detail |
|------|--------|
| File bloat | Split into modules: `parts.py`, `cleaner.py`, `expander.py`, `stem.py`, `memory.py`, `pooling.py`, `tracker.py`, `yap_core.py`, `yap.py`, `conformer.py`, `losses.py`, `config.py`. One file is unmaintainable. |
| Config defaults unvalidated | All thresholds (`on_thr=0.6`, `off_thr=0.4`, `cut_thr=0.7`, `still_flush_s=3.5`, `overlap_fraction=0.5`) are documented as "GUESSES, not validated". Need a hyperparameter sweep on held-out real ISL-CSLRT sentences. |
| PoseCleaner worse on fast motion | Documented: with jitter 0.03, cleaner is WORSE than raw on fast motion (0.044 vs 0.030 RMS). 3 Hz flick gets 2.5x WORSE (0.0161 -> 0.0405). This is exactly where sign strokes live. Needs re-tuning on real MediaPipe data or replacement with a learned/zero-phase filter. |
| Edge-on palm frame | One frame of zero velocity after every edge-on pose (prev_ok requires frame_w > 0.5 on previous frame). ~15% of real hand frames are edge-on. Needs repair or gating. |
| No timestamps in stored data | ISL-CSLRT .pt tensors carry NO timestamps. `timestamps_from_fps()` fabricates them. The docstring admits ISL-CSLRT word-level set is "NOT a valid input for the segmenter" but trainers still load these tensors directly. |
| No CTC loss in YAPpingConformer forward | Conformer produces word_logits but there is no `sentence_ctc_loss()` callable integrated into any training loop that runs end-to-end. Phase B is documented but `train_segmentor.py` explicitly says "No Conformer, CTC or word classifier." |
| Smoke tests weak | `python yap_conformer.py` runs "ALL TESTS PASSED" but tests are shape/smoke only. No collapse test, no gradient-flow test, no boundary jitter test, no live-vs-offline equivalence test. |
| HiPPOLegS exact vs ODE | Documented as "computed EXACTLY, not by integrating the HiPPO ODE". Fine, but no unit test verifies order preservation or padding invariance (dt=0 no-op). |

---

## 3. Training Pipeline — Stage 1: PRESENCE Detection (YAP Core)

The presence head answers: "is a gesture happening right now?" Everything else is frozen.

### `presence_trainer.py`

**Status:** Trains YAP presence head only (everything else frozen). Mixes word clips (presence=1) with on-the-fly junk (presence=0). Output: `yap_presence_best.pth`.

**What needs to change:**

| Item | Detail |
|------|--------|
| Dual config drift | Imports `YAPpingConformer` and `SegmentorModelCfg` from `utils.segmentor_utils`, but `train_word_presence.py` and `train_segmentor.py` use different defaults (d_model=128 vs 256 for conformer, different conformer layers/heads). No single config source. |
| Validation loss mislabeled | `history["true_loss"].append(train_avg)` — validation is logged under the wrong key. |
| Junk ratio unused | `junk_ratio` parameter accepted but dataset always returns `len(items)*2` (exactly 50/50 word/junk). Parameter is dead code. |
| No presence threshold sweep | Trains to minimize focal loss but never evaluates precision/recall at multiple `on_thr`/`off_thr` values. |
| No class imbalance handling | 228 classes, max 20 per class, but frequent vs rare words not balanced. |
| Init path optional | `--init` defaults to None, so random-init runs are possible and silently produce unusable weights. |

### `train_word_presence.py`

**Status:** Parallel presence trainer. Loads `.npy` clips (not `.pt`), uses `MixedClipDataset`, outputs to `yap_stage1_presence/`.

**What needs to change:**

| Item | Detail |
|------|--------|
| Duplicate of `presence_trainer.py` | Two presence trainers with different data loaders (`.npy` vs `.pt`), different output dirs, different default batch sizes (8 vs 64), different learning rates. One should be deleted or unified. |
| `noise_ratio` dead code | Accepted but dataset builds fixed 50/50 mix. |
| `label_map.get(item["label"], -1)` | Unknown labels silently become -1 with no warning. |
| No validation metrics | Prints loss only; no presence F1, precision, recall per epoch. |
| Path hardcoding | Docstring shows Windows path `D:\MookVani\...`. Not portable. |

### `boundary_trainer.py`

**Status:** Trains YAP end head only (everything else frozen). End target = Gaussian bump at last frame of word clip; junk = end=0 everywhere. Answers: "does a gesture end right here?"

**What needs to change:**

| Item | Detail |
|------|--------|
| Gaussian end target is weak supervision | `end_targets(t, spans, sigma_s=0.05)` places a single Gaussian at the clip's last frame. Real sign endings have temporal extent; a point target at the boundary is ambiguous. |
| `junk_ratio` dead code | Same as presence_trainer: accepted but always 50/50 mix. |
| `true_loss` mislabeled | `history["true_loss"].append(train_avg)` — same bug as presence_trainer. |
| No end-head precision/recall | Loss only. No boundary IoU, no tolerance-sweep evaluation. |
| Init path optional | Same random-init risk. |

---

## 4. Training Pipeline — Stage 2: GESTURE SEGMENTATION (Boundary/End Detection)

Later YAP models trained on segmentation: where does a sign start and end? This is the end head + SegmentTracker, not the presence head.

### `train_segmentor.py`

**Status:** Stage 2, two phases. Phase A: word-clip entity matching (no Conformer, no CTC). Phase B: synthetic sentence segmentation + CTC (documented but not clearly implemented; the script's docstring says "No Conformer, CTC or word classifier").

**What needs to change:**

| Item | Detail |
|------|--------|
| Phase B missing | The script docstring says "No Conformer, CTC or word classifier." yet structure.md says Phase B trains Conformer + CTC on stitched sentences. No Phase B code path exists in this file. `--phase B` is accepted but produces the same segmentation-only training. |
| `max_len=900` truncation | Sentences truncated to 900 frames (30s at 30fps). Spans beyond max_len are dropped silently. No warning about data loss. |
| `discover()` requires 163-dim layout | `obj.numel() % 163 == 0` check. Any non-corpus layout is rejected. |
| Validation metrics NaN in phaseA_history.json | All `train_pres`, `train_end`, `train_layer`, `val_pres`, `val_end`, `val_layer`, `val_count`, `val_gap`, `val_dip`, `val_boundary_f1_*`, `val_segment_*`, `val_false_cuts` are `NaN` for epochs 1-61, only populated at epoch 62. This means the loss decomposition and boundary metrics were broken for 61 of 62 epochs. |
| `val_segment_count_mae = 3.0`, `val_segment_iou = 0.0`, `val_boundary_f1_200ms = 0.0` | Final epoch shows zero boundary IoU and zero F1. Model does not detect boundaries on validation. |
| `source_split.json` uses absolute Windows paths | `D:\MookVani\Backend\segmentor\data\...` — not portable, breaks on any non-Windows machine. |
| Checkpoint saves only `yap.state_dict()` | Conformer/word head weights (if Phase B existed) would not be saved. |
| `--init` default points to `yap_stage1/yap_noise_robust.pth` | That file may not exist; script raises `FileNotFoundError` silently if missing. |

### `build_word_dataset.py`

**Status:** Phase 1 creates `segmentor/data/<word>/<take>/word.pt`. Phase 2 runs YAP Stage 1 front-end on each clip, saves `processed.pt` entity vectors. Two model instances on GPU.

**What needs to change:**

| Item | Detail |
|------|--------|
| `processed.pt` computed with random weights if `--init` missing | `[WARN] no --init given, using random weights` — Phase 2 proceeds with random YAP, producing garbage entity vectors used as Phase A targets. |
| Two model instances alternate per clip | `model_idx = (model_idx + 1) % 2` — both instances have identical weights (loaded from same checkpoint), so this provides no benefit. |
| `conf = torch.ones(...)`, `valid = torch.ones(...)` | All confidence/valid set to 1 regardless of actual MediaPipe quality flags stored in the 163-dim tensor's last 4 channels. Wastes the quality information. |
| `fill_missing_coords` max_gap_frames=15 hardcoded | No data-driven gap analysis. |
| `mapping.json` rewritten after phase 2 | Overwrites phase-1 mapping without recording which clips failed (`no_entity` count tracked but not saved to mapping). |
| `--phase 3` runs both but no idempotency check | Re-running phase 3 re-processes already-done clips (resume check exists but `word_path.exists()` check is racy with ThreadPoolExecutor). |

---

## 5. Data Pipeline — `data_builder/`

### `opencv_app.py`

**Status:** OpenCV desktop segmentation UI for sentence→word boundary marking. Target of multiple `fix_*.py` scripts.

**What needs to change:**

| Item | Detail |
|------|--------|
| `fix_keys_v3.py` — key discard bug | Original loop read key, then throttled/continued before processing it, so keys pressed during sleep were lost. Fix reorders: read key first, process immediately, then throttle. **Already fixed in fix_keys_v3.py but fix is a separate script, not applied to opencv_app.py itself.** |
| `fix_dup.py` — duplicate elif block | Leftover duplicate `elif` blocks (lines 607-624) including a malformed `elif key == ord('t') or key == ord('T):` (missing closing quote). Fix removes them. **Not applied to source.** |
| `fix_orphan.py` — orphaned `elif key == ord('t')` block | Dead code block after `next_frame_time += frame_interval`. **Not applied to source.** |
| `fix_isign_orphan.py` — orphaned `self.db.append()` block | Mis-indented database write block at lines 750-794. **Not applied to source.** |
| **None of the fix scripts have been applied** | `opencv_app.py` still contains the bugs the fix scripts were written to resolve. |
| Windows-only paths | `D:\MookVani\backend\segmentor\data_builder\opencv_app.py` hardcoded in every fix script. |
| No test coverage | No unit tests for the segmentation UI, key handling, or CSV persistence. |

### `data_builder/*.py` (check_words.py, check_csv.py, check_archive.py, relabel_he_to_you.py, move_free_today.py, fix_keys_v2.py)

**Status:** Ad-hoc data-curation scripts with no tests, no docstrings, no consistent CLI.

**What needs to change:**

| Item | Detail |
|------|--------|
| No unified entry point | Each script is one-off. No `main()`, no `argparse`, no logging. |
| No idempotency | Re-running may duplicate or corrupt CSV entries. |
| No type hints | All scripts lack type annotations. |
| `check_words.py`, `check_csv.py`, `check_archive.py` | Names suggest validation but no assertions, no exit codes, no report output. |

---

## 6. Evaluation / Collapse Verification

### `verify_no_collapse.py`

**Status:** Standalone audit. Runs YAP on 30 real ISL-CSLRT clips, checks output is not NULL/zero-collapsed. Also checks raw input for zero-columns.

**What needs to change:**

| Item | Detail |
|------|--------|
| Not integrated into training | Runs standalone only. No training loop calls it. No CI gate. |
| `load_flexible` import from `utils.checkpoint_utils` | May not exist; import path fragile. |
| Sample size too small | 30 clips across 10 words. No statistical power. |
| No per-class breakdown | Collapse could be class-specific (e.g. only hand shapes) and hidden in aggregate. |
| No comparison baseline | No "before vs after training" comparison. |
| No gradient-collapse check | Only checks forward output, not whether gradients vanish during training. |

### `tester.py` / `tester_v2.py` / `yap_live_first_cut.py`

**Status:** Live webcam testers using MediaPipe Tasks. `tester.py` = continuous segmentation. `tester_v2.py` = segmentation + word classification. `yap_live_first_cut.py` = freezes after first detected segment.

**What needs to change:**

| Item | Detail |
|------|--------|
| `tester.py` and `yap_live_first_cut.py` are near-duplicates | Same `extract_parts`, `overlay`, `chart`, `draw_landmarks`, `make_landmarkers`, `load_yap`. Should be one shared module. |
| `tester_v2.py` word prediction is broken | `predict_word_from_entities()` indexes `classes[ci - 1]` where `ci` comes from `pred_classes = pred_indices + 1`. If `pred_indices=0` (blank), `ci=1`, `ci-1=0` → predicts `classes[0]` for blank. Off-by-one: blank class should be skipped, not mapped to class 0. |
| `tester_v2.py` entity buffer matching is fuzzy | `seg_entities = [it for it in entity_buf if e.start_t - 0.1 <= it["t"] <= e.end_t + 0.1]` — 0.1s window on each side may include entities from adjacent segments or miss entities at segment boundaries. |
| `tester_v2.py` Conformer weights optional | Conformer initialized randomly if no Phase B checkpoint. Word predictions are random without warning. |
| No headless mode | All testers require OpenCV GUI (`cv2.imshow`). Cannot run on server/headless. |
| No automated evaluation | No comparison of live detections against ground-truth segmentation. |

### `test_stage2_fixes.py`

**Status:** Unit tests for synthetic data, splits, masks, device safety. Uses mocks for missing modules.

**What needs to change:**

| Item | Detail |
|------|--------|
| Tests use `torch.full((25+i,163),float(cid+1))` | Synthetic constant data — doesn't exercise real pose variation, missing data, or temporal structure. |
| `test_npy_clip` loads .np with `torch.load` | `.npy` files are not PyTorch tensors; `torch.load` on `.npy` will fail in real usage. |
| No integration tests | No test that runs a full training epoch end-to-end. |
| No collapse test | No test asserting loss stays finite or accuracy doesn't collapse to zero. |

---

## 7. Model Collapse Verification — Existing Trained Models

### Pipeline architecture (corrected)

YAP core = **PRESENCE detection** (presence head: "is a MediaPipe gesture happening right now?"). The presence-tuned YAP model (v2 segmentation line) detects MediaPipe gestures, and **gesture segmentation** follows — splitting the continuous detected gesture stream into individual word segments. IOU is **not applicable** here because the model is not doing bounding-box/object-detection-style segmentation; it is detecting MediaPipe gesture presence and segmenting word boundaries from that stream.

### CRITICAL: No `.pth` weight files on this device

**Note:** `Backend/segmentor/models/` contains **zero `.pth` files on this device**. Every model directory has logs/history/splits but no trained weights locally. Weights may exist on another device and are not part of this audit. No model can be loaded or deployed from this device without the weights.

### `models/yap_stage1/` (Stage 1 — PRESENCE detection)

**Files:** `yap_noise_robust.pth`, `yap_presence_best.pth` (not on this device), `stage1.err`, `history.json`, `stage1_training_report.txt`

**Collapse audit findings (presence head):**

| Metric | Value | Verdict |
|--------|-------|---------|
| `train_acc` | `0.0039` → `0.0062` → `0.0048` (0.3–0.6%) | **COLLAPSED** — accuracy near random. Model is not learning MediaPipe gesture presence. |
| `train_f1` | `4e-5` to `5.4e-4` | **COLLAPSED** — F1 near zero. |
| `train_loss` | `-4.76` → `-8.43` → `+4.93` | Loss diverged from negative to positive; training unstable. |
| `true_loss` | `7.5` → `8.8` → `5.3` | Validation loss high and unstable. |
| `blank` ratio | `0.0` → `0.70` | Increasing blank prediction — model predicts blank/noise for most frames. |
| `div` | `-24.6` → `-2.1` | Divergence metric converging toward zero (collapse indicator). |
| `stage1.err` | Present | Error log shows loss from `-8.43` (epoch 2) to `+4.93` (epoch 30) — divergence confirmed. |

**stage1.err key observation:** Epoch 2 loss = -8.4346 (best), but by epoch 30 loss = +4.9325. Training diverged after epoch 2. Presence model collapsed.

### `models/yap_stage1_presence/` (Standalone presence trainer)

**Files:** `stage1_training_report.txt` (weights `.pth` not on this device)

**Collapse audit findings (presence head):**

| Metric | Value | Verdict |
|--------|-------|---------|
| `train_acc` | `0.0066` → `0.0270` → `0.0270` | **COLLAPSED** — 2.7% accuracy. Not learning word presence. |
| `train_loss` | `-14.43` → `-20.24` → `-20.24` | Loss went negative and stagnated. Focal loss with class imbalance can produce negative loss without learning. |
| `n=6` | Only 6 epochs | Too few epochs to converge; early stopping or bug. |

### `models/yap_segmentation_only_v1/` (Stage 2 — presence-tuned, gesture segmentation)

**Files:** `history.json` (2 epochs), `source_split.json` (weights `.pth` not on this device)

**Collapse audit findings:**

| Metric | Epoch 1 | Epoch 2 | Verdict |
|--------|---------|---------|---------|
| `val_presence` | 0.013 | 0.009 | Presence loss low but could be predicting all zeros. |
| `val_end` | 0.289 | 0.248 | End loss dominant — this is the gesture-boundary signal. |
| `val_layer` | 0.274 | 0.127 | Layer loss decreasing. |
| `train` loss | 0.691 | 0.369 | Decreasing. |
| `val` loss | 0.439 | 0.321 | Decreasing. |

**Verdict:** Only 2 epochs trained. Cannot assess convergence. No boundary F1 or segmentation metrics reported. IOU is not applicable — the model detects MediaPipe gesture presence and segments word boundaries from that stream, not bounding boxes. Insufficient training to determine if gesture segmentation works.

### `models/yap_stage2/` (Stage 2 Phase A — presence-tuned, gesture segmentation)

**Files:** `phaseA_history.json` (62 epochs), `stage2.err` (weights `.pth` not on this device)

**Collapse audit findings:**

| Metric | Value | Verdict |
|--------|---------|---------|
| `train_loss` | 1.99 → 1.04 (epoch 62) | Decreased but plateaued around 1.1–1.3 for most of training. |
| `val_loss` | 1.55 → 1.49 (epoch 62, SPIKE) | Validation loss spiked at final epoch — overfitting or data issue. |
| `train_pres`/`train_end`/`train_layer` | **NaN for epochs 1–61** | Loss decomposition broken for entire training run. |
| `val_boundary_f1_200ms` | **NaN** → `0.0` (epoch 62) | Boundary F1 zero — model may not be detecting gesture boundaries. |
| `val_segment_iou` | `0.0` | **NOT APPLICABLE** — IOU is not the right metric for gesture+word segmentation from MediaPipe presence. |
| `val_segment_count_mae` | `3.0` | High error — predicts wrong number of segments/words. |
| `val_false_cuts` | `0.0` | Combined with boundary F1=0.0, suggests model predicts no cuts at all (all silence). |
| `lr` | 1e-4 → 0.0 | Cosine annealing reached zero — training ended at minimum LR. |

**Overall Stage 2 verdict:** Phase A presence-tuned model has **zero boundary F1, NaN metrics for 61/62 epochs**. IOU is not applicable. It does not appear to learn gesture boundary detection for word segmentation.

### `models/yap_segmentation_f1_v2/` (Stage 2 — PRESENCE detection tuned, gesture segmentation v2)

**Files:** `history.json` (4 epochs), `source_split.json` (weights `.pth` not on this device)

**Collapse audit findings:**

| Metric | Epoch 1 | Epoch 4 | Verdict |
|--------|---------|---------|---------|
| `boundary_f1_200ms` | 0.635 | 0.682 | Moderate gesture boundary detection. |
| `boundary_precision_200ms` | 0.569 | 0.607 | Precision on gesture boundary cuts. |
| `boundary_recall_200ms` | 0.718 | 0.777 | Recall on gesture boundary cuts. |
| `boundary_cut_reasons` | cut:832, max_len:207, presence_drop:451 | cut:897, max_len:163, presence_drop:452 | ~20% of cuts are `max_len` (forced safety limit, not learned gesture boundaries). |
| `train` loss | 0.277 | 0.231 | Slowly decreasing. |
| `val` loss | 0.252 | 0.228 | Slowly decreasing, close to train. |

**Verdict:** This is a **presence detection tuned model** used for gesture segmentation (detect MediaPipe gestures → segment words from them). Boundary F1 ~0.68 after 4 epochs — not collapsed but early. IOU is **not applicable** — the model outputs gesture presence + word boundary cuts, not detection boxes. Only 4 epochs — training may have been stopped prematurely.

### `models/yap_segmentation_f1_v3/` (Stage 2 — PRESENCE detection tuned, gesture segmentation v3)

**Files:** `source_split.json` ONLY (no `history.json`, no `.pth` weights on this device)

**Collapse audit findings:**

| Item | Verdict |
|------|---------|
| No `history.json` | Cannot assess training dynamics. |
| No `.pth` weights | Model weights do not exist. |
| Only `source_split.json` | Data split exists but no training output. Incomplete experiment. |

---

## 8. Cross-Cutting Issues

| Category | Issue | Files Affected |
|----------|-------|---------------|
| Config duplication | `SegmentorModelCfg` in `utils/segmentor_utils.py` vs inline argparse defaults in every trainer | `train_segmentor.py`, `presence_trainer.py`, `boundary_trainer.py`, `train_word_presence.py`, `build_word_dataset.py` |
| Checkpoint schema inconsistency | Some save `{'yap': ...}`, some `{'model': ...}`, some raw `state_dict()` | `train_segmentor.py` (saves `{'yap':...}`), `segmentor_utils.py` (saves `{'model':...}`), `yap_live_first_cut.py` (expects `{'yap':...}`) |
| Windows paths hardcoded | `D:\MookVani\...` in fix scripts, `source_split.json`, docstrings | `fix_*.py`, `source_split.json`, `train_word_presence.py` docstring |
| No `.npy` → `.pt` unification | `train_word_presence.py` loads `.npy`; others load `.pt` | `train_word_presence.py` vs `presence_trainer.py`/`train_segmentor.py` |
| No validation of `--init` existence | Trainers proceed with random weights if init missing | `train_segmentor.py`, `presence_trainer.py`, `boundary_trainer.py`, `train_word_presence.py` |
| No CI / automated tests | `test_stage2_fixes.py` is the only test file; no collapse test, no integration test | All |
| `yap_conformer.py` is a god file | ~1000 lines, 20+ classes, no module split | `yap_conformer.py` |
| No single entry point for training | 5+ separate training scripts with different CLI conventions | All `train_*.py` |
| Loss naming inconsistency | `train_loss`, `true_loss`, `train_pres`, `val_pres`, `train_end`, `val_end` — different keys across scripts | `presence_trainer.py`, `boundary_trainer.py`, `phaseA_history.json` |
| No learning rate finder, no early stopping, no gradient clipping report | Fixed LR schedules, clipping applied but not logged | All `train_*.py` |
| **No trained weights on this device** | Zero `.pth` files in `models/` locally (may exist on another device) | All model directories |
| **IOU not applicable** | Model outputs gesture presence + word boundary cuts, not detection boxes — IOU should not be used as a metric | `phaseA_history.json`, `yap_segmentation_f1_v2/` |

---

## 9. Recommended Changes (Priority Order)

### P0 — Fix collapse (immediate)
1. **No `.pth` files on this device** — Weights may exist elsewhere; confirm location before deploying.
2. **`phaseA_history.json`**: NaN loss components for 61/62 epochs — fix loss computation logging in `train_segmentor.py` so `train_pres`/`train_end`/`train_layer` are recorded every epoch.
3. **`train_segmentor.py` Phase B**: Implement the missing Phase B (Conformer + CTC on synthetic sentences) or remove `--phase B` option to avoid false expectation.
4. **`verify_no_collapse.py`**: Integrate into training loop as a per-epoch callback; add per-class breakdown and CI gate.
5. **`models/yap_stage1/`**: Retrain — presence accuracy 0.4%, F1 ~1e-4, negative loss, loss diverged from -8.43 to +4.93. Presence model is not learning MediaPipe gesture detection.
6. **`models/yap_stage2/`**: Retrain — boundary F1 = 0.0, NaN metrics for 61/62 epochs, false_cuts = 0.0 (predicting no cuts). Presence-tuned gesture segmentation model does not detect gesture boundaries for word segmentation.

### P1 — Unify pipeline
7. Merge `presence_trainer.py` and `train_word_presence.py` into one presence trainer.
8. Merge `tester.py`, `tester_v2.py`, `yap_live_first_cut.py` into one tester module with headless mode.
9. Create a single training config (YAML or dataclass) shared by all trainers.
10. Apply `fix_keys_v3.py`, `fix_dup.py`, `fix_orphan.py`, `fix_isign_orphan.py` fixes to `data_builder/opencv_app.py`.

### P2 — Data & validation
11. Convert `source_split.json` absolute Windows paths to relative paths.
12. Add boundary-F1, precision, recall metrics to all trainers (not just `train_segmentor.py`). IOU is not applicable — YAP detects gesture presence + word boundary cuts, not detection boxes.
13. Add class imbalance handling (weighted sampling) for 228-word vocabulary.
14. Add timestamps to stored ISL-CSLRT data or stop using word-level JPGs as segmentor input.

### P3 — Architecture
15. Split `yap_conformer.py` into modules.
16. Add unit tests for HiPPOLegS padding invariance, SegmentTracker hysteresis, PoseCleaner fast-motion behavior.
17. Add smoke test that runs `python yap_conformer.py` and asserts no NaN in outputs.

---

**Bottom line:** The YAP segmentor pipeline is architecturally sound (presence/end separation, Conformer sentence model, HiPPOLegS pooling, hysteresis tracker) but the training pipeline is fractured, unvalidated, and **no trained model weights exist on this device**. Existing training logs show clear collapse signatures: presence accuracy ~0.4%, boundary F1 = 0.0, NaN loss metrics for 61/62 epochs, loss divergence from negative to positive. IOU is not applicable — YAP detects MediaPipe gesture presence and segments word boundaries, not detection boxes. No source file should be modified based on this critique — it is an audit only.
