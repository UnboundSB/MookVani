"""
MookVani Live ISL Inference — Phase 1 CTC Sentence Model
==========================================================
Uses the Phase 1 CTC sentence model (frozen Phase 6 backbone + CTC head)
for continuous ISL sentence recognition from webcam input.

Modes:
  python live_inference.py             -> Live webcam sentence recognition
  python live_inference.py --simulate  -> Simulated inference on .pt tensors
"""

import os
import sys
import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import mediapipe as mp
import time

# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features.extract_features import extract_frame_features_video_mode

# ---------------------------------------------------------------------------
# MEDIAPIPE SETUP
# ---------------------------------------------------------------------------
BaseOptions = mp.tasks.BaseOptions
VisionMode  = mp.tasks.vision.RunningMode.VIDEO
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAND_TASK   = os.path.join(ROOT_DIR, "hand_landmarker.task")
POSE_TASK   = os.path.join(ROOT_DIR, "pose_landmarker.task")
FACE_TASK   = os.path.join(ROOT_DIR, "face_landmarker.task")

hand_options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_TASK),
    running_mode=VisionMode, num_hands=2)
pose_options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=POSE_TASK),
    running_mode=VisionMode)
face_options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=FACE_TASK),
    running_mode=VisionMode)

# ---------------------------------------------------------------------------
# MODEL ARCHITECTURE (mirrors train_synthetic_sentences.py exactly)
# ---------------------------------------------------------------------------
class PosEnc(nn.Module):
    def __init__(self, d, maxlen=4096):
        super().__init__()
        self.d = d
        pe = self._build_pe(maxlen, d)
        self.register_buffer("pe", pe)

    def _build_pe(self, maxlen, d):
        pe  = torch.zeros(maxlen, d)
        pos = torch.arange(0, maxlen).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    def forward(self, x):
        T = x.size(1)
        if T > self.pe.size(1):
            self.pe = self._build_pe(max(T + 256, self.pe.size(1) * 2), self.d).to(x.device)
        return x + self.pe[:, :T, :]

class TCNN(nn.Module):
    def __init__(self, ic, oc, k=3, drop=0.10):
        super().__init__()
        p = k // 2
        self.c1  = nn.Conv1d(ic, oc, k, padding=p)
        self.b1  = nn.BatchNorm1d(oc)
        self.g1  = nn.GELU()
        self.d1  = nn.Dropout(drop)
        self.c2  = nn.Conv1d(oc, oc, k, padding=p)
        self.b2  = nn.BatchNorm1d(oc)
        self.g2  = nn.GELU()
        self.d2  = nn.Dropout(drop)
        self.res = nn.Conv1d(ic, oc, 1) if ic != oc else nn.Identity()

    def forward(self, x):
        r = self.res(x)
        o = self.d1(self.g1(self.b1(self.c1(x))))
        o = self.d2(self.g2(self.b2(self.c2(o))))
        return o + r

class Phase1SentenceModel(nn.Module):
    def __init__(self, num_classes, d=128, gru_layers=2, n_head=4, n_trans=2, drop=0.15):
        super().__init__()
        self.num_classes  = num_classes
        self.ctc_blank_id = num_classes
        self.proj  = nn.Sequential(nn.Linear(163, d), nn.LayerNorm(d), nn.Dropout(drop))
        self.gru   = nn.GRU(d, d // 2, gru_layers, batch_first=True,
                            bidirectional=True, dropout=drop if gru_layers > 1 else 0.0)
        self.gnorm = nn.LayerNorm(d)
        self.pe    = PosEnc(d)
        el         = nn.TransformerEncoderLayer(d, n_head, d * 2, drop, "gelu", batch_first=True)
        self.trans = nn.TransformerEncoder(el, n_trans, enable_nested_tensor=False)
        self.tnorm = nn.LayerNorm(d)
        self.t1    = TCNN(d, d,     drop=drop)
        self.t2    = TCNN(d, d * 2, drop=drop)
        self.ctc_head = nn.Sequential(
            nn.Conv1d(d * 2, d * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(d * 2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv1d(d * 2, num_classes + 1, kernel_size=1)
        )

    def forward(self, x, lengths=None):
        if hasattr(self.gru, "flatten_parameters"):
            self.gru.flatten_parameters()
        h = self.proj(x)
        g, _ = self.gru(h)
        h = self.gnorm(h + g)
        h = self.pe(h)
        h = self.trans(h)
        h = self.tnorm(h)
        t = self.t1(h.transpose(1, 2))
        t = self.t2(t)                       # (B, 256, T)
        logits = self.ctc_head(t).transpose(1, 2)  # (B, T, C+1)
        return F.log_softmax(logits, dim=-1)

# ---------------------------------------------------------------------------
# CTC GREEDY DECODER
# ---------------------------------------------------------------------------
def ctc_greedy_decode(log_probs, length, blank_id):
    """Returns list of word indices decoded from (T, C+1) log-prob tensor."""
    preds = log_probs[:length].argmax(dim=-1).cpu().tolist()
    out = []
    prev = blank_id
    for tok in preds:
        if tok != blank_id and tok != prev:
            out.append(tok)
        prev = tok
    return out

# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------
BACKEND_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR    = os.path.join(BACKEND_DIR, "models", "islsentence_synthetic")
WEIGHTS_FILE = os.path.join(MODEL_DIR, "phase1_sentence_best.pth")
CLASSES_FILE = os.path.join(MODEL_DIR, "classes.json")

def load_model(device):
    if not os.path.exists(WEIGHTS_FILE) or not os.path.exists(CLASSES_FILE):
        raise FileNotFoundError(
            f"Missing model files!\n  Weights: {WEIGHTS_FILE}\n  Classes: {CLASSES_FILE}")

    with open(CLASSES_FILE, "r") as f:
        meta = json.load(f)
    classes      = meta["classes"]
    num_classes  = meta["num_classes"]
    ctc_blank_id = meta["ctc_blank_id"]
    idx_to_word  = {i: c for i, c in enumerate(classes)}

    model = Phase1SentenceModel(num_classes=num_classes).to(device)
    sd    = torch.load(WEIGHTS_FILE, map_location=device, weights_only=True)
    # Handle checkpoint dict wrapping
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd)
    model.eval()

    print(f"[LOAD] Phase 1 CTC Sentence Model — {num_classes} classes | Blank ID: {ctc_blank_id}")
    print(f"[LOAD] Weights: {WEIGHTS_FILE}")
    return model, classes, idx_to_word, ctc_blank_id

# ---------------------------------------------------------------------------
# LIVE WEBCAM INFERENCE
# ---------------------------------------------------------------------------
def run_live_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, idx_to_word, BLANK = load_model(device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    print("Webcam started — sign a sequence of words then lower your hands. Press 'q' to quit.")

    # --- State ---
    last_lh, last_rh         = np.zeros((21, 3)), np.zeros((21, 3))
    last_arms, last_face     = np.zeros((4, 3)),  np.zeros((7, 3))
    prev_lh_wrist            = np.zeros(2)
    prev_rh_wrist            = np.zeros(2)
    frame_buffer             = []          # rolling accumulation for CTC
    no_hand_counter          = 0
    still_counter            = 0
    sentence_display         = ""          # last decoded sentence
    decoded_words_display    = []          # token list from last decode
    display_hold             = 0           # frames to keep sentence visible
    infer_cooldown           = 0           # frames between consecutive decodes
    start_ms                 = int(time.time() * 1000)

    # Tuning parameters
    MAX_BUFFER    = 400    # hard cap on rolling buffer
    INFER_EVERY   = 8      # re-run CTC every N frames while signing
    MIN_FRAMES    = 20     # minimum frames before first inference
    STILL_FLUSH   = 45     # stationary frames before flushing (more generous)
    IDLE_FLUSH    = 20     # missing-hand frames before final decode+flush
    HOLD_FRAMES   = 150    # display for ~5 s
    COOLDOWN      = 6      # skip inference frames after a decode

    with mp.tasks.vision.HandLandmarker.create_from_options(hand_options) as hand_lm, \
         mp.tasks.vision.PoseLandmarker.create_from_options(pose_options) as pose_lm, \
         mp.tasks.vision.FaceLandmarker.create_from_options(face_options) as face_lm:

        while True:
            ret, frame = cap.read()
            if not ret: break

            f_rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=f_rgb)
            ts_ms     = int(time.time() * 1000) - start_ms

            hands = hand_lm.detect_for_video(mp_image, ts_ms)
            pose  = pose_lm.detect_for_video(mp_image, ts_ms)
            face  = face_lm.detect_for_video(mp_image, ts_ms)

            hands_lh, hands_rh = np.zeros((21, 3)), np.zeros((21, 3))
            lh_flag, rh_flag   = False, False
            if hands and hands.hand_landmarks:
                for idx, handedness in enumerate(hands.handedness):
                    pts = np.array([[lm.x, lm.y, lm.z] for lm in hands.hand_landmarks[idx]])
                    if handedness[0].category_name == "Left":
                        hands_lh, lh_flag = pts, True
                    else:
                        hands_rh, rh_flag = pts, True

            pose_lms = pose.pose_landmarks if pose and pose.pose_landmarks else None
            face_lms = face.face_landmarks if face and face.face_landmarks else None

            feat, last_lh, last_rh, last_arms, last_face = extract_frame_features_video_mode(
                hands_lh, hands_rh, lh_flag, rh_flag,
                pose_lms, face_lms,
                last_lh, last_rh, last_arms, last_face)

            # Wrist velocity for motion gating
            lh_spd = np.linalg.norm(hands_lh[0, :2] - prev_lh_wrist) if (lh_flag and np.any(prev_lh_wrist)) else 0.0
            rh_spd = np.linalg.norm(hands_rh[0, :2] - prev_rh_wrist) if (rh_flag and np.any(prev_rh_wrist)) else 0.0
            hand_speed = max(lh_spd, rh_spd)
            if lh_flag: prev_lh_wrist = hands_lh[0, :2].copy()
            if rh_flag: prev_rh_wrist = hands_rh[0, :2].copy()

            if infer_cooldown > 0:
                infer_cooldown -= 1
            if display_hold > 0:
                display_hold -= 1

            # ------------------------------------------------------------------
            # BUFFER ACCUMULATION
            # ------------------------------------------------------------------
            hand_active = hands and (lh_flag or rh_flag)
            if hand_active:
                no_hand_counter = 0
                still_counter   = still_counter + 1 if hand_speed < 0.008 else 0
                frame_buffer.append(feat)
                if len(frame_buffer) > MAX_BUFFER:
                    frame_buffer = frame_buffer[-MAX_BUFFER:]
                # Static hands held too long — flush (not signing)
                if still_counter > STILL_FLUSH:
                    frame_buffer.clear()
                    still_counter = 0
            else:
                no_hand_counter += 1
                still_counter    = 0
                if no_hand_counter >= IDLE_FLUSH:
                    # Hands fully lowered: run final decode then flush
                    if len(frame_buffer) >= MIN_FRAMES:
                        sent, words = _decode_with_fallback(
                            model, frame_buffer, BLANK, idx_to_word, device, tag="FINAL")
                        if sent:
                            sentence_display      = sent
                            decoded_words_display = words
                            display_hold          = HOLD_FRAMES
                    frame_buffer.clear()
                    prev_lh_wrist = np.zeros(2)
                    prev_rh_wrist = np.zeros(2)
                    last_lh, last_rh = np.zeros((21, 3)), np.zeros((21, 3))
                    last_arms, last_face = np.zeros((4, 3)), np.zeros((7, 3))

            # ------------------------------------------------------------------
            # ROLLING CTC INFERENCE (while actively signing)
            # ------------------------------------------------------------------
            if (hand_active and infer_cooldown == 0
                    and len(frame_buffer) >= MIN_FRAMES
                    and len(frame_buffer) % INFER_EVERY == 0):
                sent, words = _decode_with_fallback(
                    model, frame_buffer, BLANK, idx_to_word, device, tag="LIVE")
                if sent:
                    sentence_display      = sent
                    decoded_words_display = words
                    display_hold          = HOLD_FRAMES
                    infer_cooldown        = COOLDOWN

            # ------------------------------------------------------------------
            # DISPLAY
            # ------------------------------------------------------------------
            frame = cv2.resize(frame, (1280, 720))
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (1270, 180), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

            # Status dot
            dot_color  = (0, 220, 0) if hand_active else (60, 60, 200)
            status_txt = f"Signing  [buf:{len(frame_buffer)}]" if hand_active else f"Idle  [buf:{len(frame_buffer)}]"
            cv2.circle(frame, (30, 38), 9, dot_color, -1)
            cv2.putText(frame, f"MookVani ISL | {status_txt}", (50, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)

            # Sentence output
            disp_color = (0, 230, 255) if display_hold > 0 and sentence_display else (100, 100, 100)
            disp_text  = sentence_display.upper() if display_hold > 0 and sentence_display else "[Sign to begin...]"
            # Truncate if too long for one line
            if len(disp_text) > 55:
                disp_text = disp_text[:52] + "..."
            cv2.putText(frame, disp_text, (25, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, disp_color, 2, cv2.LINE_AA)

            # Token-level confidence bar (word list)
            if decoded_words_display and display_hold > 0:
                bar = " → ".join(decoded_words_display[:8])
                cv2.putText(frame, bar, (25, 145),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 210, 180), 1, cv2.LINE_AA)

            cv2.putText(frame, "Press 'q' to quit", (25, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1, cv2.LINE_AA)

            # Draw hand landmarks
            if hands and hands.hand_landmarks:
                for hidx, lms in enumerate(hands.hand_landmarks):
                    clr = (255, 80, 0) if hands.handedness[hidx][0].category_name == "Left" else (0, 80, 255)
                    for lm in lms:
                        cx, cy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                        cv2.circle(frame, (cx, cy), 3, clr, -1)

            cv2.imshow("MookVani — Live ISL Sentence Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def _decode(model, frame_buffer, blank_id, idx_to_word, device):
    """Runs CTC greedy decode. Returns (sentence_str, word_list, log_probs_tensor)."""
    arr = torch.tensor(np.array(frame_buffer), dtype=torch.float32).unsqueeze(0).to(device)
    T   = arr.size(1)
    with torch.no_grad():
        log_probs = model(arr)  # (1, T, C+1)
    tokens = ctc_greedy_decode(log_probs.squeeze(0), T, blank_id)
    words  = [idx_to_word.get(t, "?") for t in tokens]
    return " ".join(words), words, log_probs.squeeze(0)

def _vote_fallback(log_probs, blank_id, idx_to_word, top_k=5):
    """
    Frame-voting fallback: when CTC collapses to all blanks,
    compute per-frame argmax, ignore blanks, and return the
    most frequently voted non-blank words.
    """
    argmax = log_probs.argmax(dim=-1).cpu().tolist()  # (T,)
    counts = {}
    for tok in argmax:
        if tok != blank_id:
            counts[tok] = counts.get(tok, 0) + 1
    if not counts:
        return "", []
    ranked = sorted(counts.items(), key=lambda x: -x[1])[:top_k]
    words  = [idx_to_word.get(t, "?") for t, _ in ranked]
    # Show with vote % for debug
    total  = len(argmax)
    info   = [(idx_to_word.get(t, "?"), round(c / total * 100)) for t, c in ranked]
    return " | ".join([f"{w}({p}%)" for w, p in info]), words

def _decode_with_fallback(model, frame_buffer, blank_id, idx_to_word, device, tag=""):
    """
    Tries CTC greedy decode first; if result is empty, falls back to frame voting.
    Returns (display_str, word_list).
    """
    sent, words, lp = _decode(model, frame_buffer, blank_id, idx_to_word, device)

    # Debug: blank ratio
    argmax     = lp.argmax(dim=-1).cpu()
    blank_pct  = (argmax == blank_id).float().mean().item() * 100
    top1_words = [idx_to_word.get(t.item(), "?") for t in argmax[:5]]
    print(f"[{tag}] buf={len(frame_buffer)} blank={blank_pct:.0f}% "
          f"ctc='{sent}' first5={top1_words}")

    if words:
        return sent, words

    # Fallback: frame-vote top candidates
    fb_sent, fb_words = _vote_fallback(lp, blank_id, idx_to_word, top_k=4)
    if fb_words:
        print(f"[{tag}] CTC empty — using vote fallback: {fb_sent}")
        return f"[~] {fb_sent}", fb_words

    return "", []

# ---------------------------------------------------------------------------
# SIMULATION MODE — streams saved .pt tensors through the CTC decoder
# ---------------------------------------------------------------------------
def run_simulated_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, idx_to_word, BLANK = load_model(device)

    DATA_DIR   = os.path.join(BACKEND_DIR, "data", "tensors_include50_word_level")
    test_words = ["brother", "hospital", "time", "mother", "boat", "actor"]
    available  = [w for w in test_words if os.path.isdir(os.path.join(DATA_DIR, w))]
    if not available:
        available = [d for d in os.listdir(DATA_DIR)
                     if os.path.isdir(os.path.join(DATA_DIR, d)) and d != "NONE"][:5]

    print("\n" + "=" * 70)
    print("MOOKVANI — SYNTHETIC SENTENCE CTC SIMULATION")
    print(f"Building a synthetic sentence with: {', '.join(available)}")
    print("=" * 70)

    import random
    n = random.randint(2, min(5, len(available)))
    chosen = random.sample(available, n)
    print(f"\nSynthetic sentence: [{' | '.join(chosen)}]")

    # Concatenate tensors with resting frames between
    tensors = []
    rest = torch.zeros(10, 163)
    rest[:, :63]   = torch.tensor([-1.5, 3.0, 0.0]).repeat(21).unsqueeze(0).expand(10, -1)
    rest[:, 63:126] = rest[:, :63]
    tensors.append(rest)

    for w in chosen:
        files = [os.path.join(DATA_DIR, w, f)
                 for f in os.listdir(os.path.join(DATA_DIR, w)) if f.endswith(".pt")]
        if not files: continue
        t = torch.load(random.choice(files), weights_only=True).float()
        tensors.append(t)
        tensors.append(rest[:6])

    full = torch.cat(tensors, dim=0)
    print(f"Total frames: {full.shape[0]}\n")

    # Stream frame-by-frame and decode every 15 frames
    buffer = []
    print(f"{'FRAME':<8} | {'DECODED SEQUENCE'}")
    print("-" * 60)
    for i in range(full.shape[0]):
        buffer.append(full[i].numpy())
        if len(buffer) >= 30 and i % 15 == 0:
            sent, words = _decode(model, buffer, BLANK, idx_to_word, device)
            print(f"{i+1:<8} | {sent}")

    print("\n>>> FINAL DECODE:")
    sent, words = _decode(model, buffer, BLANK, idx_to_word, device)
    print(f"    Decoded sentence: '{sent}'")
    print(f"    Expected:         '{' '.join(chosen)}'")
    print("=" * 70)

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--simulate", "--test", "-t"):
        run_simulated_inference()
    else:
        run_live_inference()
