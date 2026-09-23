import os
import sys
import cv2
import torch
import json
import numpy as np
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Add current dir to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.isl_sentence_model import ISL_Sentence_Model
from features.extract_features import extract_frame_features_video_mode

# --- Setup MediaPipe ---
BaseOptions = mp.tasks.BaseOptions
VisionMode = mp.tasks.vision.RunningMode.VIDEO

hand_options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="../hand_landmarker.task"),
    running_mode=VisionMode, num_hands=2)
pose_options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="../pose_landmarker.task"),
    running_mode=VisionMode)
face_options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="../face_landmarker.task"),
    running_mode=VisionMode)

def decode_ctc(sequence, idx_to_class, blank_id=0):
    decoded = []
    last_tok = None
    for tok in sequence:
        if tok != blank_id and tok != last_tok:
            decoded.append(idx_to_class.get(tok, f"<UNK:{tok}>"))
        last_tok = tok
    return " ".join(decoded)

def run_live_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_path = "models/word_class_to_idx.json"
    model_path = "models/best_sentence_model.pth"

    if not os.path.exists(vocab_path) or not os.path.exists(model_path):
        print("Error: Ensure best_sentence_model.pth and word_class_to_idx.json exist in models/")
        return

    with open(vocab_path, "r") as f:
        class_to_idx = json.load(f)
    
    idx_to_class = {i + 1: c for c, i in class_to_idx.items()}
    num_classes = len(class_to_idx)
    
    print("Loading model...")
    model = ISL_Sentence_Model(num_classes=num_classes, d_model=256, lstm_layers=1, dropout=0.0).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam started! Press 'q' to quit. Start signing!")

    last_lh, last_rh = np.zeros((21, 3)), np.zeros((21, 3))
    last_arms, last_face = np.zeros((4, 3)), np.zeros((7, 3))
    
    frame_buffer = []
    no_hand_counter = 0
    current_prediction = ""
    start_time_ms = int(time.time() * 1000)

    with vision.HandLandmarker.create_from_options(hand_options) as hand_lm, \
         vision.PoseLandmarker.create_from_options(pose_options) as pose_lm, \
         vision.FaceLandmarker.create_from_options(face_options) as face_lm:

        while True:
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1) # Mirror for user convenience
            f_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=f_rgb)
            
            timestamp_ms = int(time.time() * 1000) - start_time_ms

            hands = hand_lm.detect_for_video(mp_image, timestamp_ms)
            pose = pose_lm.detect_for_video(mp_image, timestamp_ms)
            face = face_lm.detect_for_video(mp_image, timestamp_ms)

            hands_lh, hands_rh = np.zeros((21, 3)), np.zeros((21, 3))
            lh_flag, rh_flag = False, False

            if hands and hands.hand_landmarks:
                no_hand_counter = 0
                for idx, handedness in enumerate(hands.handedness):
                    label = handedness[0].category_name
                    pts = np.array([[lm.x, lm.y, lm.z] for lm in hands.hand_landmarks[idx]])
                    if label == 'Left':
                        hands_lh = pts; lh_flag = True
                    else:
                        hands_rh = pts; rh_flag = True
            else:
                no_hand_counter += 1

            pose_lms = pose.pose_landmarks if pose and pose.pose_landmarks else None
            face_lms = face.face_landmarks if face and face.face_landmarks else None

            feat_vector, last_lh, last_rh, last_arms, last_face = extract_frame_features_video_mode(
                hands_lh, hands_rh, lh_flag, rh_flag,
                pose_lms, face_lms,
                last_lh, last_rh, last_arms, last_face
            )

            # If hands are visible, or we are within a short grace period, append to buffer
            if no_hand_counter < 15:
                frame_buffer.append(feat_vector)
            else:
                # Idle for a while, clear buffer and prediction
                if len(frame_buffer) > 0:
                    frame_buffer.clear()
                    current_prediction = ""
                    # Reset last states
                    last_lh, last_rh = np.zeros((21, 3)), np.zeros((21, 3))
                    last_arms, last_face = np.zeros((4, 3)), np.zeros((7, 3))

            # Run inference periodically if we have enough frames
            if len(frame_buffer) >= 10 and len(frame_buffer) % 5 == 0:
                tensor_data = torch.tensor(np.array(frame_buffer), dtype=torch.float32).unsqueeze(0).to(device)
                lengths = torch.tensor([tensor_data.size(1)], dtype=torch.long).to(device)
                
                with torch.no_grad():
                    log_probs, _ = model(tensor_data, lengths)
                
                probs = torch.exp(log_probs).squeeze(0) # (T, C)
                max_probs, predicted_seq = probs.max(dim=-1)
                predicted_seq = predicted_seq.cpu().numpy()
                max_probs = max_probs.cpu().numpy()
                
                # Confidence Thresholding (ignore predictions < 60% confident)
                for i in range(len(predicted_seq)):
                    if max_probs[i] < 0.6:
                        predicted_seq[i] = 0 # Force to CTC blank
                        
                current_prediction = decode_ctc(predicted_seq, idx_to_class)

            # Enlarge frame for better visibility
            frame = cv2.resize(frame, (1280, 720))

            # Display logic
            status_color = (0, 255, 0) if no_hand_counter < 15 else (0, 0, 255)
            cv2.circle(frame, (30, 30), 10, status_color, -1)
            
            # Show prediction (Black outline)
            cv2.putText(frame, current_prediction, (15, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 6, cv2.LINE_AA)
            # Show prediction (Deep Blue fill in BGR: 255, 130, 0)
            cv2.putText(frame, current_prediction, (15, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 130, 0), 3, cv2.LINE_AA)
                        
            cv2.putText(frame, f"Frames: {len(frame_buffer)}", (15, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)

            cv2.imshow("MookVani Live ISL Translation", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_inference()
