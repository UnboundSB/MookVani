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

from models.isl_sentence_model import ISLSentenceReconformer
from features.extract_features import extract_frame_features_video_mode

# --- Setup MediaPipe ---
BaseOptions = mp.tasks.BaseOptions
VisionMode = mp.tasks.vision.RunningMode.VIDEO
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAND_TASK = os.path.join(ROOT_DIR, "hand_landmarker.task")
POSE_TASK = os.path.join(ROOT_DIR, "pose_landmarker.task")
FACE_TASK = os.path.join(ROOT_DIR, "face_landmarker.task")

hand_options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_TASK),
    running_mode=VisionMode, num_hands=2)
pose_options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=POSE_TASK),
    running_mode=VisionMode)
face_options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=FACE_TASK),
    running_mode=VisionMode)

def load_mookvani_model(device):
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "word_level", "reconformer_word_135_best.pth")
    CLASSES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "word_level", "classes_135.json")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_PATH):
        raise FileNotFoundError(f"Model or classes file missing! Looked for {MODEL_PATH} and {CLASSES_PATH}")

    with open(CLASSES_PATH, "r") as f:
        meta = json.load(f)
    classes = meta["classes"]
    num_classes = len(classes) - 1 # 135 gesture classes (+ 1 NONE = 136 outputs)
    idx_to_class = {i: c for i, c in enumerate(classes)}

    print(f"Loading Reconformer (135 classes + NONE) on {device}...")
    model = ISLSentenceReconformer(num_classes=num_classes, d_model=128, lstm_layers=1, dropout=0.0, use_rnn=True).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print("Model loaded successfully!")
    return model, classes, idx_to_class

def run_live_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, classes, idx_to_class = load_mookvani_model(device)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam started! Press 'q' to quit. Start signing!")

    last_lh, last_rh = np.zeros((21, 3)), np.zeros((21, 3))
    last_arms, last_face = np.zeros((4, 3)), np.zeros((7, 3))
    
    frame_buffer = []
    no_hand_counter = 0
    still_counter = 0
    cooldown_counter = 0
    display_hold_counter = 0
    current_prediction = ""
    top3_info = []
    prev_lh_wrist = np.zeros(2)
    prev_rh_wrist = np.zeros(2)
    start_time_ms = int(time.time() * 1000)

    with vision.HandLandmarker.create_from_options(hand_options) as hand_lm, \
         vision.PoseLandmarker.create_from_options(pose_options) as pose_lm, \
         vision.FaceLandmarker.create_from_options(face_options) as face_lm:

        while True:
            ret, frame = cap.read()
            if not ret: break

            # DO NOT FLIP THE FRAME! Flipping inverts the X coordinates and breaks the 
            # trained neural network's spatial mapping (it was trained on unmirrored video).
            f_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=f_rgb)
            
            timestamp_ms = int(time.time() * 1000) - start_time_ms

            hands = hand_lm.detect_for_video(mp_image, timestamp_ms)
            pose = pose_lm.detect_for_video(mp_image, timestamp_ms)
            face = face_lm.detect_for_video(mp_image, timestamp_ms)

            hands_lh, hands_rh = np.zeros((21, 3)), np.zeros((21, 3))
            lh_flag, rh_flag = False, False

            if hands and hands.hand_landmarks:
                for idx, handedness in enumerate(hands.handedness):
                    label = handedness[0].category_name
                    pts = np.array([[lm.x, lm.y, lm.z] for lm in hands.hand_landmarks[idx]])
                    if label == 'Left':
                        hands_lh = pts; lh_flag = True
                    else:
                        hands_rh = pts; rh_flag = True

            pose_lms = pose.pose_landmarks if pose and pose.pose_landmarks else None
            face_lms = face.face_landmarks if face and face.face_landmarks else None

            feat_vector, last_lh, last_rh, last_arms, last_face = extract_frame_features_video_mode(
                hands_lh, hands_rh, lh_flag, rh_flag,
                pose_lms, face_lms,
                last_lh, last_rh, last_arms, last_face
            )

            # Measure wrist velocity to detect active signing vs stationary idle hands
            lh_speed = np.linalg.norm(hands_lh[0, :2] - prev_lh_wrist) if (lh_flag and np.any(prev_lh_wrist != 0)) else 0.0
            rh_speed = np.linalg.norm(hands_rh[0, :2] - prev_rh_wrist) if (rh_flag and np.any(prev_rh_wrist != 0)) else 0.0
            hand_speed = max(lh_speed, rh_speed)
            if lh_flag: prev_lh_wrist = hands_lh[0, :2].copy()
            if rh_flag: prev_rh_wrist = hands_rh[0, :2].copy()

            # Countdown timers for display hold and post-detection cooldown
            if cooldown_counter > 0:
                cooldown_counter -= 1
            if display_hold_counter > 0:
                display_hold_counter -= 1
            elif display_hold_counter == 0 and current_prediction and not current_prediction.startswith("["):
                current_prediction = ""

            # Buffer accumulation with motion gating
            if hands and (lh_flag or rh_flag):
                no_hand_counter = 0
                if hand_speed < 0.008:
                    still_counter += 1
                else:
                    still_counter = 0

                # Append frame only during active signing (outside of post-emission cooldown)
                if cooldown_counter == 0:
                    frame_buffer.append(feat_vector)
                    if len(frame_buffer) > 75:
                        frame_buffer = frame_buffer[-75:]

                # If hands remain stationary/still for > 18 frames (~0.6s), flush buffer
                # (Prevents resting static hand positions from accumulating as fake signs like 'tall'/'wide')
                if still_counter > 18 and len(frame_buffer) > 0:
                    frame_buffer.clear()
                    top3_info = []
            else:
                no_hand_counter += 1
                still_counter = 0
                # Hands lowered or left frame: flush buffer and reset tracking state
                if no_hand_counter >= 12:
                    if len(frame_buffer) > 0:
                        frame_buffer.clear()
                        top3_info = []
                    prev_lh_wrist = np.zeros(2)
                    prev_rh_wrist = np.zeros(2)
                    last_lh, last_rh = np.zeros((21, 3)), np.zeros((21, 3))
                    last_arms, last_face = np.zeros((4, 3)), np.zeros((7, 3))

            # Trigger inference at sign completion:
            # (a) Hand just slowed down / stroke ended (still_counter >= 2 with at least 25 frames)
            # OR (b) Buffer reached 55 frames of continuous active motion
            stroke_ended = (still_counter >= 2 and len(frame_buffer) >= 25)
            buffer_ripe = (len(frame_buffer) >= 55 and len(frame_buffer) % 4 == 0)

            if cooldown_counter == 0 and (stroke_ended or buffer_ripe):
                tensor_data = torch.tensor(np.array(frame_buffer), dtype=torch.float32).unsqueeze(0).to(device)
                lengths = torch.tensor([tensor_data.size(1)], dtype=torch.long).to(device)
                
                with torch.no_grad():
                    log_probs, _ = model(tensor_data, lengths)
                    
                # Correct probability pooling: average probabilities directly (sums to 1.0)
                probs = torch.exp(log_probs).mean(dim=1).squeeze(0) # (C,)
                
                top_probs, top_indices = torch.topk(probs, 3)
                top3_info = [(idx_to_class.get(idx.item(), "NONE"), p.item()) for idx, p in zip(top_indices, top_probs)]
                
                max_prob = top_probs[0].item()
                predicted_idx = top_indices[0].item()
                word = idx_to_class.get(predicted_idx, "NONE")
                
                print(f"DEBUG: Predicted [{word}] | Prob: {max_prob:.2f} | Buffer: {len(frame_buffer)} | Speed: {hand_speed:.3f}")
                
                # Active gesture detection threshold
                if max_prob >= 0.38 and word != "NONE":
                    current_prediction = f"{word.upper()} ({max_prob*100:.0f}%)"
                    display_hold_counter = 75  # Hold visible for ~2.5s
                    cooldown_counter = 18       # 0.6s cooldown before accepting next sign
                    # FLUSH BUFFER: Completely clear buffer so this gesture cannot re-trigger
                    frame_buffer.clear()
                    print(f">>> [DETECTED & BUFFER FLUSHED] -> {current_prediction}")
                elif stroke_ended and len(frame_buffer) > 0:
                    # Stroke finished but confidence didn't reach threshold: flush to avoid stale re-prediction
                    frame_buffer.clear()

            # Draw landmarks manually with OpenCV to avoid mediapipe solutions versioning issues
            if hands and hands.hand_landmarks:
                for idx, hand_landmarks in enumerate(hands.hand_landmarks):
                    color = (255, 0, 0) if hands.handedness[idx][0].category_name == 'Left' else (0, 0, 255)
                    for lm in hand_landmarks:
                        cx, cy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                        cv2.circle(frame, (cx, cy), 3, color, -1)
            
            if pose_lms:
                for lm in pose_lms[0]:
                    cx, cy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    cv2.circle(frame, (cx, cy), 2, (0, 255, 255), -1)

            # Enlarge frame for better visibility
            frame = cv2.resize(frame, (1280, 720))

            # Semi-transparent HUD overlay on top-left
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (450, 170), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            # Tracking status indicator
            status_color = (0, 255, 0) if no_hand_counter < 15 else (0, 0, 255)
            status_text = "Hands Active" if no_hand_counter < 15 else "Hands Idle"
            cv2.circle(frame, (30, 35), 8, status_color, -1)
            cv2.putText(frame, f"MOOKVANI ISL: {status_text} (Buf: {len(frame_buffer)})", (48, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
            
            # Show prediction
            pred_color = (0, 230, 255) if current_prediction and "NONE" not in current_prediction else (180, 180, 180)
            cv2.putText(frame, current_prediction if current_prediction else "[Awaiting gesture...]", (25, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, pred_color, 2, cv2.LINE_AA)
                        
            # Show top 3 candidate predictions
            if top3_info:
                cand_str = " | ".join([f"{w}: {p*100:.0f}%" for w, p in top3_info])
                cv2.putText(frame, f"Top Candidates: {cand_str}", (25, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Start signing in front of the camera...", (25, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)

            cv2.putText(frame, "Press 'q' to quit", (25, 155), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 130, 130), 1, cv2.LINE_AA)

            cv2.imshow("MookVani Live ISL Translation", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def run_simulated_inference():
    """
    Demonstrates live streaming inference using real gesture pose landmark files,
    streaming frames through the buffer with real-time sliding window inference.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, classes, idx_to_class = load_mookvani_model(device)
    
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tensors_include50_word_level")
    
    test_words = ["brother", "priest", "hospital", "time", "boat", "mother"]
    available_words = [w for w in test_words if os.path.isdir(os.path.join(DATA_DIR, w))]
    if not available_words:
        available_words = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)) and d != "NONE"][:5]
        
    print("\n" + "=" * 75)
    print("MOOKVANI LIVE INFERENCE SIMULATION (REAL-TIME SLIDING WINDOW)")
    print(f"Testing Words: {', '.join(available_words)}")
    print("=" * 75)
    
    for word_target in available_words:
        word_dir = os.path.join(DATA_DIR, word_target)
        files = [os.path.join(word_dir, f) for f in os.listdir(word_dir) if f.endswith(".pt")]
        if not files:
            continue
            
        test_file = files[0]
        video_tensor = torch.load(test_file, weights_only=True) # [T, 163]
        total_frames = video_tensor.shape[0]
        
        print(f"\n>>> SIGNING GESTURE: [{word_target.upper()}] ({total_frames} frames from {os.path.basename(test_file)})")
        print(f"{'FRAME':<8} | {'BUFFER':<8} | {'CONFIDENCE':<12} | {'PREDICTION':<20} | {'STATUS':<15}")
        print("-" * 75)
        
        frame_buffer = []
        last_prediction = ""
        detected = False
        
        for f_idx in range(total_frames):
            frame_feat = video_tensor[f_idx].numpy()
            frame_buffer.append(frame_feat)
            if len(frame_buffer) > 100:
                frame_buffer = frame_buffer[-100:]
                
            # Run sliding window inference every 5 frames if buffer has at least 20 frames
            if len(frame_buffer) >= 20 and len(frame_buffer) % 5 == 0:
                active_window = frame_buffer[-70:]
                tensor_data = torch.tensor(np.array(active_window), dtype=torch.float32).unsqueeze(0).to(device)
                lengths = torch.tensor([tensor_data.size(1)], dtype=torch.long).to(device)
                
                with torch.no_grad():
                    log_probs, _ = model(tensor_data, lengths)
                    
                # Correct probability pooling: average probabilities directly (sums to 1.0)
                probs = torch.exp(log_probs).mean(dim=1).squeeze(0)
                max_prob, pred_idx = probs.max(dim=-1)
                pred_idx = pred_idx.item()
                max_prob = max_prob.item()
                pred_word = idx_to_class.get(pred_idx, "NONE")
                
                status = ""
                if pred_word == word_target:
                    status = "[MATCH!]"
                    detected = True
                elif pred_word != "NONE":
                    status = "Active"
                else:
                    status = "Rest/Transition"
                    
                print(f"{f_idx+1:<8} | {len(active_window):<8} | {max_prob*100:>5.1f}%       | {pred_word:<20} | {status}")
                
        print(f"Result for [{word_target}]: {'SUCCESSFULLY DETECTED' if detected else 'Completed'}")
        
    print("\n" + "=" * 75)
    print("LIVE SIMULATION COMPLETE!")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["--test", "--simulate", "-t"]:
        run_simulated_inference()
    else:
        run_live_inference()

