# MookVani

MookVani is an end-to-end Continuous Indian Sign Language (ISL) Recognition and Translation engine. Built on a hybrid architecture combining PyTorch Conformer blocks with Bidirectional LSTMs and Connectionist Temporal Classification (CTC), it translates dynamic skeletal gestures into grammatically correct sentence sequences in real-time.

## Features

- **Continuous Sentence Recognition:** Evaluates a live stream of keypoints and decodes continuous sign sequences without needing segmented words.
- **ISL_Conformer Architecture:** Multi-stream self-attention encoder utilizing a 163-dimensional canonical skeleton (Hands, Pose, Face) optimized for MediaPipe landmarks.
- **Synthetic Data Engine (`SyntheticSentenceDataset`):** 
  - Dynamically synthesizes grammatically valid training sequences on the fly using the ISL CSLTR corpus grammar trees.
  - Implements dynamic scaling, temporal warping, and 3D spatial noise for massive robustness, overcoming data scarcity by converting 20 base samples per class into infinite training variations.
  - Generates robust `NONE` background noise handling to eliminate idle hallucination.
- **Strict Validation Protocol:** Guarantees zero data leakage by evaluating strictly on unseen holdout signers.
- **Real-time Live Inference:** Plugs straight into a webcam stream for instantaneous continuous translation output, powered by MediaPipe and CPU/GPU PyTorch execution.

## Repository Structure

- `Backend/models/`: Neural Network architectures (`isl_conformer.py`, `isl_sentence_model.py`) and pre-trained weights (`.pth`).
- `Backend/features/`: MediaPipe canonical keypoint extraction and normalizations (`extract_features.py`).
- `Backend/data/`: Data loading pipelines and the dynamic synthetic generation engine (`synthetic_dataset.py`, `datasets.py`).
- `Backend/training/`: Training modules for word-level initialization and end-to-end synthetic sentence training (`train_synthetic.py`).
- `Backend/scripts/`: Various utility scripts for dataset processing, extraction, and visualizations.
- `Backend/live_inference.py`: Live webcam inference script using MediaPipe Tasks API.

## Setup

1. Create a Python environment (`python -m venv venv`)
2. Activate environment: `.\venv\Scripts\activate` (Windows)
3. Install requirements (`pip install -r Backend/requirements.txt`)
4. Ensure MediaPipe `.task` models are in the root directory.
5. Run the live inference stream:
```bash
cd Backend
python live_inference.py
```

## Performance
The current model operates over a 115-word vocabulary, demonstrating highly robust real-world Word Error Rate (WER) capabilities and handling diverse background conditions due to its synthetic noise training routines.
