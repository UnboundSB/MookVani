# Directory Structure: MookVani Root

Root directory of the MookVani Indian Sign Language (ISL) recognition platform.

## Files

- `.gitignore`: Git ignore rules for datasets, model checkpoints, virtual environments, caches, and logs.
- `README.md`: High-level documentation describing architecture, setup, dataset pipelines, and model details.
- `face_landmarker.task`: MediaPipe pre-trained model bundle for facial landmark extraction.
- `hand_landmarker.task`: MediaPipe pre-trained model bundle for hand landmark extraction.
- `holistic_landmarker.task`: MediaPipe holistic landmarker bundle extracting face, hand, and pose landmarks.
- `pose_landmarker.task`: MediaPipe pose landmarker bundle for body pose estimation.
- `structure.md`: Directory structure documentation for MookVani root.
- `yap_conformer.py`: **NOTE**: This file does NOT exist at the root level. The YAP model lives at `Backend/segmentor/yap_conformer.py`. (Protected - Do Not Modify.)

## Immediate Subdirectories

- `Backend/`: ML models, feature extraction pipelines, data loaders, training loops, and inference services.
- `.agents/`: Project-level agent rules, guidelines, and persistent instructions.
