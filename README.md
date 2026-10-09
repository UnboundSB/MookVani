# MookVani

MookVani is an end-to-end Continuous Indian Sign Language (ISL) Recognition and Translation engine. Built on a hybrid architecture combining PyTorch Conformer blocks with Bidirectional LSTMs and Connectionist Temporal Classification (CTC), it translates dynamic skeletal gestures into grammatically correct sentence sequences in real-time.

## Features

- **Continuous Sentence Recognition:** Evaluates a live stream of keypoints and decodes continuous sign sequences without needing segmented words.
- **WordCategorizerModel Architecture:** Multi-stream self-attention encoder utilizing a 163-dimensional canonical skeleton (Hands, Pose, Face) optimized for MediaPipe landmarks.
- **ISLSentenceReconformer Engine:** 
  - An advanced wrapper mapping frame-level outputs to a Bidirectional LSTM block to temporally smooth predictions before classification.
  - Dynamically synthesizes heavily augmented data in real-time within `ISLAugmentedWordDataset` (applying 3D spatial jitter, coordinate shifts, tilt scaling, and continuous temporal speed-up/slow-down interpolation) for infinite training variations.
  - Initializes from completely non-randomized, perfect synthetic pre-trained weights to ensure hyper-robust transfer learning.
- **Strict Validation Protocol:** Guarantees zero data leakage by evaluating strictly on unseen holdout signers and providing ultra-detailed F1, Precision, and Recall metrics.
- **Real-time Live Inference:** Plugs straight into a webcam stream for instantaneous continuous translation output, powered by MediaPipe and CPU/GPU PyTorch execution.

## Repository Structure

- `Backend/models/`: Neural Network architectures (`isl_conformer.py`, `isl_sentence_model.py`) and pre-trained weights (`.pth`).
- `Backend/features/`: MediaPipe canonical keypoint extraction and normalizations (`extract_features.py`).
- `Backend/data/`: Tensors and dataset storage location (ignored in Git).
- `Backend/training/`: Training modules for the `ISLSentenceReconformer` (`train_reconformer_word.py`) and plot generation routines.
- `Backend/scripts/`: Heavy lifting utilities for parsing raw `.mp4` files into parallelized 163-dimensional `(T, 163)` coordinate tensors (`extract_word.py`).
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
