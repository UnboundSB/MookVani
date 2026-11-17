"""
ISL Reconformer Engine
======================

Reusable, deterministic inference engine for the trained ISL model.

Pipeline:
  camera/image frame
    -> MediaPipe Tasks HandLandmarker
    -> 21 pixel landmarks (x, y) per detected hand
    -> wrist-relative + max-abs normalization (42 features per hand)
    -> single-hand model input 42, or combined two-hand input 84
    -> Keras model
    -> stable 1-second voting over predictions

The preprocessing here intentionally mirrors the training-time preprocessing:
the wrist point becomes (0, 0), and every coordinate is divided by the largest
absolute coordinate in the same frame.
"""

from __future__ import annotations

import collections
import copy
import itertools
import os
import string
import time
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Silence TensorFlow chatter before importing the Keras loader.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from tensorflow import keras

import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core import base_options as mp_base_options


# Exact class order used during training.
ALPHABET: List[str] = [str(d) for d in range(1, 10)] + list(string.ascii_uppercase)

# Clock increment for VIDEO running mode. 67 ms is roughly 15 FPS and keeps
# MediaPipe timestamps strictly increasing without depending on wall-clock time.
_VIDEO_STEP_MS = 67


class ISLReconformer:
    """Self-contained hand-landmark + Keras inference engine."""

    FEATURE_COUNT = 42
    SINGLE_HAND_FEATURES = 42
    TWO_HAND_FEATURES = 84
    LANDMARK_COUNT = 21

    def __init__(
        self,
        model_path: Optional[str] = None,
        task_path: Optional[str] = None,
        vote_window_seconds: float = 1.0,
        min_prediction_confidence: float = 0.0,
        stream_fps: int = 15,
    ) -> None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.abspath(model_path or os.path.join(base_dir, "islreconformer.h5"))
        self.task_path = os.path.abspath(task_path or os.path.join(base_dir, "hand_landmarker.task"))

        self._require_file(self.model_path, "Keras model")
        self._require_file(self.task_path, "MediaPipe hand landmarker model")

        self.min_prediction_confidence = float(min_prediction_confidence)
        self.vote_window_s = max(0.05, float(vote_window_seconds))
        self.stream_fps = max(1, int(stream_fps))

        self.model = keras.models.load_model(self.model_path)

        self.input_feature_count = self.SINGLE_HAND_FEATURES
        self.input_shape = None
        self.output_shape = None

        self._validate_model_contract()

        self.detector = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_base_options.BaseOptions(model_asset_path=self.task_path),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

        # Stable-output voting state.
        self.window_votes: collections.deque = collections.deque()
        self.window_confs: collections.deque = collections.deque()
        self.last_update_t = time.time()

        self.current_label = "-"
        self.current_confidence = 0.0

        self._timestamp_ms = 0
        self._frame_index = 0

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _require_file(path: str, label: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{label} not found: {path}\n"
                f"Place the required file next to this script and retry."
            )
        if os.path.getsize(path) <= 0:
            raise ValueError(f"{label} is empty: {path}")

    def _validate_model_contract(self) -> None:
        """Fail immediately if the model does not match the expected pipeline."""
        try:
            input_shape = self.model.input_shape
            output_shape = self.model.output_shape
        except Exception as exc:
            raise ValueError(f"Unable to inspect model input/output shapes: {exc}") from exc

        input_dims = list(input_shape[1:]) if isinstance(input_shape, (list, tuple)) else []
        output_dims = list(output_shape[1:]) if isinstance(output_shape, (list, tuple)) else []

        self.input_shape = input_shape
        self.output_shape = output_shape

        if input_dims:
            input_count = int(input_dims[-1])

            if input_count == self.SINGLE_HAND_FEATURES:
                self.input_feature_count = self.SINGLE_HAND_FEATURES
            elif input_count == self.TWO_HAND_FEATURES:
                self.input_feature_count = self.TWO_HAND_FEATURES
            else:
                raise ValueError(
                    f"Model expects {input_count} input features, "
                    f"but this pipeline supports {self.SINGLE_HAND_FEATURES} "
                    f"single-hand or {self.TWO_HAND_FEATURES} combined two-hand features."
                )
        else:
            self.input_feature_count = self.SINGLE_HAND_FEATURES

        if output_dims:
            classes = int(output_dims[-1])
            if classes != len(ALPHABET):
                raise ValueError(
                    f"Model has {classes} outputs, but ALPHABET has {len(ALPHABET)} labels. "
                    "Update ALPHABET to match the trained classes."
                )

    # ------------------------------------------------------------------
    # Training-equivalent preprocessing
    # ------------------------------------------------------------------
    @staticmethod
    def calc_landmark_list(
        image_shape: Tuple[int, int, int],
        norm_landmarks,
    ) -> List[List[int]]:
        """Convert normalized landmarks to integer pixel coordinates."""
        h, w = int(image_shape[0]), int(image_shape[1])
        points: List[List[int]] = []

        for lm in norm_landmarks:
            x = min(int(float(lm.x) * w), w - 1)
            y = min(int(float(lm.y) * h), h - 1)
            points.append([max(0, x), max(0, y)])

        if len(points) != ISLReconformer.LANDMARK_COUNT:
            raise ValueError(
                f"Expected {ISLReconformer.LANDMARK_COUNT} hand landmarks, got {len(points)}."
            )

        return points

    @staticmethod
    def pre_process_landmark(landmark_list: List[List[int]]) -> List[float]:
        """Wrist-relative + max-abs normalization to a 42-float feature vector."""
        if len(landmark_list) != ISLReconformer.LANDMARK_COUNT:
            raise ValueError(
                f"Expected {ISLReconformer.LANDMARK_COUNT} landmarks, got {len(landmark_list)}."
            )

        tmp = copy.deepcopy(landmark_list)

        # Wrist is index 0 in MediaPipe hand landmarks.
        base_x, base_y = tmp[0]
        for i in range(len(tmp)):
            tmp[i][0] -= base_x
            tmp[i][1] -= base_y

        flat = list(itertools.chain.from_iterable(tmp))
        max_value = max(map(abs, flat))

        if max_value == 0:
            return [0.0 for _ in flat]

        return [float(v) / float(max_value) for v in flat]

    def features_from_landmarks(self, landmark_list: List[List[int]]) -> np.ndarray:
        """Return one hand's 42-feature vector."""
        features = self.pre_process_landmark(landmark_list)
        x = np.asarray(features, dtype=np.float32).reshape(1, -1)

        if x.shape[1] != self.SINGLE_HAND_FEATURES:
            raise ValueError(
                f"Feature vector has {x.shape[1]} values, "
                f"expected {self.SINGLE_HAND_FEATURES}."
            )

        return x

    def combined_two_hand_features(
        self,
        hand_feature_vectors: List[List[float]],
    ) -> List[float]:
        """Combine one or two 42-float hand vectors into an 84-float vector."""
        if not hand_feature_vectors:
            return [0.0] * self.TWO_HAND_FEATURES

        padded = list(hand_feature_vectors[:2])

        while len(padded) < 2:
            padded.append([0.0] * self.SINGLE_HAND_FEATURES)

        return list(padded[0]) + list(padded[1])

    def features_for_frame_hands(
        self,
        landmark_lists: List[List[List[int]]],
    ) -> List[np.ndarray]:
        """
        Build model input vectors for all detected hands.

        - 42-input models: one vector per hand.
        - 84-input models: one vector containing both hands; a single hand is
          zero-padded into the second half.
        """
        hand_features = [
            self.pre_process_landmark(landmark_list)
            for landmark_list in landmark_lists
        ]

        if self.input_feature_count == self.SINGLE_HAND_FEATURES:
            return [
                np.asarray(features, dtype=np.float32).reshape(1, -1)
                for features in hand_features
            ]

        if not hand_features:
            return []

        combined = self.combined_two_hand_features(hand_features)
        return [np.asarray(combined, dtype=np.float32).reshape(1, -1)]

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _predict_feature_vector(
        self,
        x: np.ndarray,
    ) -> Tuple[str, float, int, object]:
        """Run the Keras model on a prepared feature vector."""
        if x.shape[1] != self.input_feature_count:
            raise ValueError(
                f"Feature vector has {x.shape[1]} values, "
                f"expected {self.input_feature_count}."
            )

        probs = self.model.predict(x, verbose=0)[0]

        if len(probs) != len(ALPHABET):
            raise ValueError(
                f"Model produced {len(probs)} probabilities, expected {len(ALPHABET)}."
            )

        class_idx = int(np.argmax(probs))
        return ALPHABET[class_idx], float(probs[class_idx]), class_idx, probs

    def predict_landmarks(
        self, landmark_list: List[List[int]]
    ) -> Tuple[str, float, int, object]:
        """Predict a class from one hand's 21 pixel landmarks."""
        x = self.features_from_landmarks(landmark_list)

        if self.input_feature_count != self.SINGLE_HAND_FEATURES:
            raise ValueError(
                "This model expects combined two-hand features. "
                "Use process_frame() for full camera input."
            )

        return self._predict_feature_vector(x)

    def _next_timestamp_ms(self) -> int:
        self._timestamp_ms += _VIDEO_STEP_MS
        return self._timestamp_ms

    def process_frame(self, frame_bgr: np.ndarray) -> Dict[str, Union[str, float, int, List]]:
        """Process a single BGR camera/image frame."""
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Received an empty frame.")

        self._frame_index += 1

        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.detector.detect_for_video(mp_image, self._next_timestamp_ms())
        hand_landmarks = getattr(result, "hand_landmarks", None) or []

        hands_payload: List[Dict[str, Union[str, float, List]]] = []
        landmark_payloads: List[List[List[float]]] = []

        for hand_lms in hand_landmarks:
            landmark_payloads.append(
                [
                    [float(lm.x), float(lm.y), float(lm.z)]
                    for lm in hand_lms
                ]
            )

        landmark_lists = [
            self.calc_landmark_list(frame_bgr.shape, hand_lms)
            for hand_lms in hand_landmarks
        ]

        feature_vectors = self.features_for_frame_hands(landmark_lists)
        frame_predictions: List[Tuple[str, float, int, object]] = []

        for x in feature_vectors:
            frame_predictions.append(self._predict_feature_vector(x))

        # Map each prediction back to the hands it represents.
        prediction_groups: List[List[int]] = []

        if self.input_feature_count == self.TWO_HAND_FEATURES:
            hand_indices = list(range(len(hand_landmarks)))
            prediction_groups.append(hand_indices[:2])
        else:
            prediction_groups = [[index] for index in range(len(hand_landmarks))]

        best_class: Optional[int] = None
        best_conf = 0.0

        for group_indices, (label, conf, class_idx, probs) in zip(
            prediction_groups, frame_predictions
        ):
            ranked = np.argsort(probs)[::-1][:3]
            top_predictions = [
                {
                    "label": ALPHABET[int(i)],
                    "class_index": int(i),
                    "probability": float(probs[int(i)]),
                }
                for i in ranked
            ]

            for hand_index in group_indices:
                hands_payload.append(
                    {
                        "hand_index": hand_index,
                        "label": label,
                        "confidence": round(conf, 4),
                        "class_index": class_idx,
                        "top_predictions": top_predictions,
                        "landmarks": landmark_payloads[hand_index],
                    }
                )

            if conf >= self.min_prediction_confidence and conf > best_conf:
                best_conf = conf
                best_class = class_idx

        if best_class is not None:
            self.window_votes.append(best_class)
            self.window_confs.append(best_conf)

        now = time.time()
        updated = False
        if now - self.last_update_t >= self.vote_window_s:
            if self.window_votes:
                top_class = collections.Counter(self.window_votes).most_common(1)[0][0]
                winner_confs = [
                    conf
                    for cls, conf in zip(self.window_votes, self.window_confs)
                    if cls == top_class
                ]
                self.current_label = ALPHABET[top_class]
                self.current_confidence = float(np.mean(winner_confs)) if winner_confs else 0.0
            else:
                self.current_label = "-"
                self.current_confidence = 0.0

            self.window_votes.clear()
            self.window_confs.clear()
            self.last_update_t = now
            updated = True

        return {
            "prediction": self.current_label,
            "confidence": round(float(self.current_confidence), 4),
            "hands_count": len(hand_landmarks),
            "hands": hands_payload,
            "prediction_updated": updated,
            "frame_index": self._frame_index,
            "input_mode": (
                "two_hand_combined"
                if self.input_feature_count == self.TWO_HAND_FEATURES
                else "single_hand"
            ),
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "input_feature_count": self.input_feature_count,
        }

    def close(self) -> None:
        """Release native MediaPipe resources."""
        detector = getattr(self, "detector", None)
        if detector is not None:
            try:
                detector.close()
            except Exception:
                pass


# Optional singleton helper for APIs/web workers.
_engine: Optional[ISLReconformer] = None


def predict_isl_frame(frame_bgr: np.ndarray) -> Dict:
    """Singleton frame helper for FastAPI/React/backend usage."""
    global _engine
    if _engine is None:
        _engine = ISLReconformer()
    return _engine.process_frame(frame_bgr)