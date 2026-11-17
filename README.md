# Indian Sign Language Recognition (ISL Reconformer)

Real-time Indian Sign Language (ISL) gesture recognition system powered by MediaPipe hand landmark extraction and a trained Feedforward Neural Network (`islreconformer.h5`).

---

## Repository Architecture & Code Responsibilities

| File | Purpose & Responsibility |
|---|---|
| **[`islreconformer.h5`](file:///d:/islcsltr/Indian-Sign-Language-Detection/islreconformer.h5)** | The trained ISL deep learning neural network model (35 classes: digits 1–9, letters A–Z). Expects 84 normalized two-hand coordinate inputs; a one-hand fallback model would expect 42. |
| **[`isl_reconformer.py`](file:///d:/islcsltr/Indian-Sign-Language-Detection/isl_reconformer.py)** | **Core Engine & React/API Connector**. Provides `ISLReconformer` class and `predict_isl_frame()` function. Extracts 21 MediaPipe hand landmarks, performs relative + max-abs normalization, and outputs structured prediction JSON every 1 second. |
| **[`live_inference.py`](file:///d:/islcsltr/Indian-Sign-Language-Detection/live_inference.py)** | **Live 15 FPS Video Pipeline**. Captures webcam feed using DirectShow at 15 FPS, runs inference on each frame, pools votes over a 1-second rolling window, renders HUD (label, confidence bar, FPS counter, hand skeleton), and serves an instant browser stream at `http://localhost:5000`. |
| **[`dataset_keypoint_generation.py`](file:///d:/islcsltr/Indian-Sign-Language-Detection/dataset_keypoint_generation.py)** | **Data Preprocessing Script**. Processes raw sign language training images, calculates normalized 21 keypoints using MediaPipe, and exports feature rows into `keypoint.csv`. |
| **[`keypoint.csv`](file:///d:/islcsltr/Indian-Sign-Language-Detection/keypoint.csv)** | Legacy single-hand feature sample containing 7,669 rows (42 coordinates per row + gesture label). The two-hand model expects 84 inputs. |

| **[`isl_detection.py`](file:///d:/islcsltr/Indian-Sign-Language-Detection/isl_detection.py)** | Compatibility entry point. Runs the active `live_inference.py` pipeline. |
| **[`hand_landmarker.task`](file:///d:/islcsltr/Indian-Sign-Language-Detection/hand_landmarker.task)** | MediaPipe Tasks Vision model asset bundle used for high-precision hand landmark tracking. |

---

## Coordinate Extraction & Normalization Flow

The `islreconformer` model supports both one-hand and two-hand input shapes. For the current two-hand model:
1. **Landmark Detection**: Up to 21 3D hand joints per hand are detected from RGB frames using MediaPipe.
2. **Wrist Centering**: Each hand’s coordinates are translated relative to its own Landmark 0 (wrist origin):
   $$x_i' = x_i - x_0, \quad y_i' = y_i - y_0$$
3. **Flattening**: Each hand is flattened into a 42-dimensional vector: $[x_0', y_0', x_1', y_1', \dots, x_{20}', y_{20}']$.
4. **Two-Hand Combination**: Hand 1 and hand 2 vectors are concatenated into an 84-dimensional input. If only one hand is visible, the second half is zero-padded.
5. **Max-Abs Normalization**: Coordinates are scaled by the maximum absolute coordinate value:
   $$v_{norm} = \frac{v}{\max(|v|)}$$
6. **Inference**: Target 15 FPS sampling with majority-voting window updating predictions every 1.0 second.

---

## Quick Usage

### Live Stream (15 FPS + 1s Updates):
```powershell
.venv\Scripts\python.exe live_inference.py
```
Open **`http://localhost:5000`** in any browser to view the live HUD stream.

### React / API Integration:
```python
from isl_reconformer import predict_isl_frame

# Feed BGR frame from webcam / canvas
result = predict_isl_frame(frame)
# Returns: {"prediction": "A", "confidence": 0.94, "hands_count": 1, ...}
```
