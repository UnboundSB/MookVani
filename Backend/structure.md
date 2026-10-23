# Directory Structure: Backend

Core backend engine containing ML models, feature extractors, training loops, and inference services.

## Files
- `live_inference.py`: Real-time webcam sentence recognition using the Phase 1 CTC sentence model (Phase1SentenceModel with frozen Phase 6 backbone). Greedily decodes continuous ISL sign sequences into sentences — or, by default, with the **prefix beam search rescored by the word-level language model** (`--beam_width 25 --beam_alpha 0.5`, the LM contributes the word ORDER only). Supports `--simulate` mode for streaming saved .pt tensor files through the CTC decoder. A checkpoint whose head does not match the published vocabulary is loaded through `utils/checkpoint_utils.py` (the head is dropped and reported instead of crashing).
- `merge_isign_to_dataset.py`: Merges and aligns iSign dataset annotations into main dataset directories.
- `requirements.txt`: Python package dependencies required for backend execution and training.
- `run_pipeline.py`: **Pipeline orchestrator** with three presets: `--preset accuracy` (the maximum-accuracy recipe: the canonical split + the language model, the pretrainer, the five word phases, the CTC stage with the per-layer learning rates / the staged release / the strided front-end, the synthetic sentence stage and the real-domain fine-tune with the three-source mix and the beam+LM decoding), `--preset fast` (the same without the beam search and the LM) and `--preset minimal` (just refreshes the manifest and the language model). `--only <step>` runs a single stage, `--list` shows every step with its flags.
- `run_train_synthetic.py`: Script to launch training jobs on synthetic sentence datasets.
- `structure.md`: Directory structure documentation for Backend.

## Immediate Subdirectories
- `eda/`: Exploratory data analysis scripts and dataset inspection routines.
- `features/`: Feature extraction logic and MediaPipe holistic landmark processing.
- `models/`: Neural network architectures (Conformer, ISL models) and saved checkpoint weights.
- `scripts/`: Data augmentation, extraction pipelines, directory management, and evaluation utilities.
- `segmentor/`: YAP (You Adapt Progressively) gesture segmentor workspace. Contains `yap_conformer.py` (the full YAP + YAPpingConformer implementation), noise augmentation, and the two-stage training pipeline (noise filtration + synthetic sentence segmentation).
- `training/`: Model training routines, fine-tuning scripts, and validation evaluation scripts.
- `utils/`: The word-level and sentence-level dataloaders, the canonical pose augmentation set, the checkpoint helpers, gloss mapping and the evaluation metrics.

