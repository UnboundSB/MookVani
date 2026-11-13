"""
Minimal Word-Level ISL Inference — Uses existing Phase 4/5/6 word classifiers.

These are simple classifiers (WordCategorizerModel), not CTC models.
They classify ONE isolated sign at a time — no sentence decoding needed.

Usage:
    python infer_word.py --ckpt models/phase4_word_model/best_phase4_model.pth
    python infer_word.py --ckpt models/phase5_word_model/best_phase5_model.pth
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BACKEND_DIR)
if BACKEND_DIR not in sys.path: sys.path.insert(0, BACKEND_DIR)

from models.isl_conformer import WordCategorizerModel
from utils.checkpoint_utils import extract_state_dict, load_flexible
from features.extract_features import extract_frame_features_video_mode
from training.train_conformer_ctc import harmonize_frame

import cv2
import mediapipe as mp


class WordInference:
    def __init__(self, ckpt_path, classes_path, device):
        with open(classes_path) as f:
            classes = json.load(f)
        if isinstance(classes, dict):
            classes = classes["classes"]

        self.classes = classes
        self.device = device
        self.model = WordCategorizerModel(len(classes)).to(device)

        raw = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = extract_state_dict(raw)
        missing, unexpected = load_flexible(self.model, sd, label="WORD")
        bad = [k for k in missing if not k.endswith("pe")]
        if bad:
            print(f"[WARN] {len(bad)} mismatched keys (vocab change ok): {bad[:3]}")

        self.model.eval()
        print(f"[LOAD] {os.path.basename(ckpt_path)} | {len(classes)} classes | {device}")

    @torch.no_grad()
    def predict(self, feat_seq):
        """feat_seq: (T, 163) numpy array -> returns (pred_idx, confidence, top5)"""
        x = torch.from_numpy(feat_seq).float().unsqueeze(0).to(self.device)
        lengths = torch.tensor([feat_seq.shape[0]], dtype=torch.long, device=self.device)
        log_probs, _ = self.model(x, lengths)  # (1, T_out, C+1)
        probs = log_probs.exp()[0]  # (T_out, C+1)

        # Average over time (exclude blank)
        word_probs = probs[:, :-1].mean(0)  # (C,)
        pred_idx = word_probs.argmax().item()
        confidence = word_probs[pred_idx].item()

        top5 = torch.topk(word_probs, 5)
        top5_list = [(self.classes[i], p.item()) for i, p in zip(top5.indices, top5.values)]

        return pred_idx, confidence, top5_list


def extract_sequence(source=0, max_frames=300, fps=30):
    """Extract 163-dim sequence from webcam/video OR load from .pt file."""
    # If it's a .pt file, load it directly
    if isinstance(source, str) and source.endswith('.pt'):
        seq = torch.load(source, weights_only=True).float().numpy()
        if seq.ndim == 3:
            seq = seq[0]
        return seq[:max_frames]

    # Otherwise use camera/video
    BaseOptions = mp.tasks.BaseOptions
    Mode = mp.tasks.vision.RunningMode.VIDEO
    mk = lambda cls, name, **kw: cls(
        base_options=BaseOptions(model_asset_path=os.path.join(ROOT_DIR, name)),
        running_mode=Mode, **kw)

    hand_opt = mk(mp.tasks.vision.HandLandmarkerOptions, "hand_landmarker.task", num_hands=2)
    pose_opt = mk(mp.tasks.vision.PoseLandmarkerOptions, "pose_landmarker.task")
    face_opt = mk(mp.tasks.vision.FaceLandmarkerOptions, "face_landmarker.task")

    cap = cv2.VideoCapture(source if isinstance(source, str) and not source.isdigit() else int(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {source}")

    file_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    last_lh = last_rh = last_arms = last_face = (
        np.zeros((21, 3)), np.zeros((21, 3)), np.zeros((4, 3)), np.zeros((7, 3))
    )
    last_ts_ms = -1
    frames = []

    with mp.tasks.vision.HandLandmarker.create_from_options(hand_opt) as hand_lm, \
         mp.tasks.vision.PoseLandmarker.create_from_options(pose_opt) as pose_lm, \
         mp.tasks.vision.FaceLandmarker.create_from_options(face_opt) as face_lm:

        print("Sign a word, then press 'q' to stop recording...")
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret: break

            t = len(frames) / file_fps
            ts_ms = max(int(t * 1000), last_ts_ms + 1)
            last_ts_ms = ts_ms

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            hands = hand_lm.detect_for_video(mp_image, ts_ms)
            pose = pose_lm.detect_for_video(mp_image, ts_ms)
            face = face_lm.detect_for_video(mp_image, ts_ms)

            lh, rh = np.zeros((21, 3)), np.zeros((21, 3))
            lh_flag = rh_flag = False
            if hands and hands.hand_landmarks:
                for i, handed in enumerate(hands.handedness):
                    pts = np.array([[p.x, p.y, p.z] for p in hands.hand_landmarks[i]])
                    if handed[0].category_name == "Left":
                        lh, lh_flag = pts, True
                    else:
                        rh, rh_flag = pts, True

            feat, last_lh, last_rh, last_arms, last_face = extract_frame_features_video_mode(
                lh, rh, lh_flag, rh_flag,
                pose.pose_landmarks[0] if pose and pose.pose_landmarks else None,
                face.face_landmarks[0] if face and face.face_landmarks else None,
                last_lh, last_rh, last_arms, last_face)

            feat = harmonize_frame(feat)
            frames.append(feat)

            # Display
            cv2.putText(frame, f"Frames: {len(frames)}/{max_frames}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Sign a word (press 'q' to classify)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    return np.array(frames)


def main():
    parser = argparse.ArgumentParser(description="Word-level ISL inference")
    parser.add_argument("--ckpt", default=os.path.join(BACKEND_DIR, "models", "phase5_word_model", "best_phase5_model.pth"))
    parser.add_argument("--classes", default=os.path.join(BACKEND_DIR, "models", "phase5_word_model", "classes.json"))
    parser.add_argument("--source", default="0", help="webcam index or video file")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--min_conf", type=float, default=0.3, help="Minimum confidence to accept prediction")
    args = parser.parse_args()

    device = torch.device(args.device)
    infer = WordInference(args.ckpt, args.classes, device)

    print("\n--- Ready ---")
    print(f"Press 'q' while recording to classify the sign")
    print(f"Min confidence: {args.min_conf}")
    print(f"Vocabulary: {len(infer.classes)} words\n")

    while True:
        seq = extract_sequence(args.source)
        if len(seq) < 5:
            print("Too short, try again")
            continue

        pred_idx, conf, top5 = infer.predict(seq)
        word = infer.classes[pred_idx]

        print(f"\n{'='*50}")
        print(f"PREDICTION: {word}  (confidence: {conf:.2%})")
        print(f"Top-5:")
        for w, p in top5:
            print(f"  {w}: {p:.2%}")
        print(f"{'='*50}\n")

        if conf < args.min_conf:
            print(f"[LOW CONFIDENCE] Below {args.min_conf:.0%} threshold")

        if input("Another? (y/n): ").lower() != 'y':
            break


if __name__ == "__main__":
    main()