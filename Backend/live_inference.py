"""
MookVani Live ISL Inference: Conformer-CTC (real-sentence model)
================================================================
Webcam (or video file) -> MediaPipe -> 163-dim features -> ISLHybridConformerCTC.

Also outputs YAP-compatible raw coordinates per frame:
  pose  (J=42, 3)  float32   left-hand (21 pts) + right-hand (21 pts) in image-normalised units
  conf  (J=42,)    float32   MediaPipe confidence [0,1]; 1.0 if not available
  valid (J=42,)    bool      False = landmark absent / below min_conf this frame
  t     float               monotonic seconds

These four values are produced by extract_yap_frame() and are passed to the optional
on_yap_frame(pose, conf, valid, t) callback in run_camera(), which is the entry point
for YAPStream.push() once YAP is trained.

Behaviour
---------
* A sentence segment STARTS when your hands appear and move. A short lead-in of rest context
  (--lead_sec) is kept, like the training clips.
* While you sign, the model re-decodes the buffer every --infer_every seconds. Partial
  words are STABILISED by local agreement: a word is only "committed" (shown solid) once the
  last --agree_n partial decodes agree on it. Committed words never change mid-sentence.
  Words that are not yet agreed are shown dim as the tentative tail.
* The segment FLUSHES as soon as your hands are gone for --idle_sec (default 0.3 s), or your
  hands stay still in view for --still_sec. Trailing idle frames are trimmed, one last full
  decode runs, and that result is the FINAL sentence. Buffer and extractor state are reset
  so the next sentence starts clean.
* Frames are resampled by their real timestamps to --fps before decoding, so a slow webcam
  or a busy CPU does not change signing speed as the model sees it.

Usage (from the repo root or Backend/)
    python Backend/live_inference_conformer.py                       # webcam 0
    python Backend/live_inference_conformer.py --source clip.mp4     # video file
    python Backend/live_inference_conformer.py --simulate t.pt       # saved (T,163) tensor, no camera
    python Backend/live_inference_conformer.py --ckpt Backend/models/isl_real_sentences/real_sentence_ema_best.pth

The decision logic (LiveRecognizer) has no camera or MediaPipe dependency, so it can be
tested offline. cv2 and mediapipe are imported only when the camera/video loop runs.
"""

import os, sys, json, time, argparse, itertools
from collections import deque
from dataclasses import dataclass
import numpy as np
import torch

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BACKEND_DIR)
if BACKEND_DIR not in sys.path: sys.path.insert(0, BACKEND_DIR)

from training.train_conformer_ctc import ISLHybridConformerCTC
from utils.checkpoint_utils import extract_state_dict, load_flexible

DEFAULT_CKPT = os.path.join(BACKEND_DIR, "models", "isl_real_sentences", "real_sentence_best.pth")
CLASSES_PATH = os.path.join(BACKEND_DIR, "models", "word_vocabulary", "classes.json")


# =============================================================================
# CONFIG
# =============================================================================
@dataclass
class Config:
    fps: float = 30.0            # frame rate the model sees (frames are resampled to this)
    idle_sec: float = 0.30       # no hands for this long -> flush ("hands off the screen")
    still_sec: float = 1.5       # hands visible but motionless this long -> flush (0 = off)
    still_speed: float = 0.25    # mean landmark speed (image-width units/s) below this = still
    lead_sec: float = 0.20       # rest context kept before the first moving frame (seconds, so it
    tail_sec: float = 0.14       # and after the last one, independent of camera speed)
    min_sign_sec: float = 0.5    # shorter hand activity is discarded (hands passing through)
    max_sec: float = 15.0        # force a flush on very long segments
    infer_every_sec: float = 0.30
    agree_n: int = 3             # partial decodes that must agree before a word is committed
    min_conf: float = 0.5        # drop words whose peak probability is below this
    min_dur: int = 1             # min frames per emitted word (keep 1 for peaky CTC)


# =============================================================================
# PURE HELPERS (unit-testable)
# =============================================================================
def resample_nearest(times, feats, fps):
    """Zero-order-hold resample irregularly timed frames onto a uniform `fps` grid."""
    times = np.asarray(times, dtype=np.float64)
    feats = np.asarray(feats, dtype=np.float32)
    if len(times) <= 1:
        return feats
    n = max(1, int(round((times[-1] - times[0]) * fps)) + 1)
    grid = times[0] + np.arange(n) / fps
    idx = np.clip(np.searchsorted(times, grid, side="left"), 1, len(times) - 1)
    left, right = times[idx - 1], times[idx]
    idx = np.where(grid - left < right - grid, idx - 1, idx)
    return feats[idx]


def decode_words(log_probs, blank_id, min_conf=0.5, min_dur=1):
    """Greedy CTC -> [(token, peak_prob, start_frame, end_frame)]. Blank-separated repeats are kept."""
    am = log_probs.argmax(-1).tolist()
    probs = log_probs.exp()
    out, pos = [], 0
    for tok, grp in itertools.groupby(am):
        n = sum(1 for _ in grp)
        if tok != blank_id and n >= min_dur:
            conf = probs[pos:pos + n, tok].max().item()
            if conf >= min_conf:
                out.append((tok, conf, pos, pos + n))
        pos += n
    return out


def common_prefix(seqs):
    out = []
    for items in zip(*seqs):
        if all(x == items[0] for x in items): out.append(items[0])
        else: break
    return out


class AgreementStabilizer:
    """
    Local-agreement stabiliser for streaming hypotheses. `committed` only ever grows within a
    segment; the tentative tail is whatever the latest hypothesis adds after it.
    """
    def __init__(self, n=3):
        self.n = max(1, n)
        self.reset()

    def reset(self):
        self.hist = deque(maxlen=self.n)
        self.committed = []
        self.tentative = []

    def update(self, hyp):
        hyp = list(hyp)
        self.hist.append(hyp)
        if len(self.hist) == self.n:
            lcp = common_prefix(list(self.hist))
            if len(lcp) > len(self.committed) and lcp[:len(self.committed)] == self.committed:
                self.committed = lcp
        c = len(self.committed)
        self.tentative = hyp[c:] if hyp[:c] == self.committed else []
        return self.committed, self.tentative


# =============================================================================
# RECOGNISER (segmentation + decoding + stabilising; no camera code)
# =============================================================================
class LiveRecognizer:
    """
    Feed one frame at a time with push(). Returns a list of events:
      ("start", t)                       a sentence segment began
      ("partial", committed, tentative)  stabilised partial words (strings)
      ("final", words, confs, reason)    sentence finished (reason: hands_down / still / max_len)
      ("discard", reason)                too little hand activity to be a sentence
      ("reset", None)                    caller should reset its feature-extractor carry-over state
    """
    def __init__(self, model, classes, device, cfg=None, blank_penalty=0.0):
        self.model, self.classes, self.device = model, classes, device
        self.cfg = cfg or Config()
        self.blank = len(classes)
        self.blank_penalty = blank_penalty
        self.stab = AgreementStabilizer(self.cfg.agree_n)
        self.lead = deque()           # recent idle frames, trimmed by time in push()
        self.history = deque(maxlen=4)
        self.final = []
        self.last_infer_ms = 0.0
        self._reset_segment()

    # ---- state ---------------------------------------------------------------
    def _reset_segment(self):
        self.active = False
        self.buf = []                 # (t, feat, hands, moving)
        self.first_move_t = self.last_move_t = self.last_hand_t = None
        self.last_infer_t = 0.0
        self.committed, self.tentative = [], []
        self._last_view = None
        self.stab.reset()
        self.lead.clear()

    @property
    def status(self):
        return "signing" if self.active else "idle"

    # ---- model ---------------------------------------------------------------
    @torch.no_grad()
    def _infer(self, seg):
        t0 = time.perf_counter()
        times = [s[0] for s in seg]
        feats = np.stack([s[1] for s in seg])
        x = torch.from_numpy(resample_nearest(times, feats, self.cfg.fps)).float().unsqueeze(0).to(self.device)
        self.model.eval()
        lp = self.model(x)[0].float()
        if self.blank_penalty:
            lp = lp.clone(); lp[:, self.blank] -= self.blank_penalty
            lp = torch.log_softmax(lp, dim=-1)
        words = decode_words(lp, self.blank, self.cfg.min_conf, self.cfg.min_dur)
        self.last_infer_ms = 1000 * (time.perf_counter() - t0)
        return words

    def _names(self, toks):
        return [self.classes[t] for t in toks]

    # ---- main entry ----------------------------------------------------------
    def push(self, t, feat, hands, speed=None):
        """
        t: seconds (monotonic). feat: (163,) features. hands: any hand detected.
        speed: mean landmark speed in image-width units/s, or None if unknown (then any
        detected hand counts as moving).
        """
        cfg, events = self.cfg, []
        moving = bool(hands) and (speed is None or speed >= cfg.still_speed)

        if not self.active:
            self.lead.append((t, feat, bool(hands), moving))
            while self.lead and t - self.lead[0][0] > cfg.lead_sec:
                self.lead.popleft()
            if moving:
                self.active = True
                self.buf = list(self.lead)
                self.lead.clear()
                self.first_move_t = self.last_move_t = self.last_hand_t = t
                self.last_infer_t = t
                self.stab.reset()
                if self.final:
                    self.history.appendleft(self.final)
                    self.final = []
                events.append(("start", t))
            return events

        self.buf.append((t, feat, bool(hands), moving))
        if hands: self.last_hand_t = t
        if moving: self.last_move_t = t

        reason = None
        if t - self.last_hand_t >= cfg.idle_sec:
            reason = "hands_down"
        elif cfg.still_sec > 0 and t - self.last_move_t >= cfg.still_sec:
            reason = "still"
        elif t - self.buf[0][0] >= cfg.max_sec:
            reason = "max_len"
        if reason:
            return events + self._flush(reason)

        if (t - self.last_infer_t >= cfg.infer_every_sec
                and self.last_move_t - self.first_move_t >= cfg.min_sign_sec):
            self.last_infer_t = t
            words = self._infer(self.buf)
            committed, tentative = self.stab.update([w[0] for w in words])
            self.committed, self.tentative = self._names(committed), self._names(tentative)
            view = (tuple(self.committed), tuple(self.tentative))
            if view != self._last_view:
                self._last_view = view
                events.append(("partial", self.committed, self.tentative))
        return events

    def _flush(self, reason):
        events = []
        if self.last_move_t - self.first_move_t >= self.cfg.min_sign_sec:
            last_idx = max(i for i, s in enumerate(self.buf) if s[3])
            end = last_idx + 1                       # keep tail_sec of context after the last moving frame
            while end < len(self.buf) and self.buf[end][0] - self.buf[last_idx][0] <= self.cfg.tail_sec:
                end += 1
            seg = self.buf[:end]
            words = self._infer(seg)
            self.final = self._names([w[0] for w in words])
            events.append(("final", list(self.final), [w[1] for w in words], reason))
        else:
            events.append(("discard", reason))
        self._reset_segment()
        events.append(("reset", None))
        return events


# =============================================================================
# MODEL LOADING
# =============================================================================
def load_model(ckpt_path, classes_path, device):
    with open(classes_path) as f: classes = json.load(f)
    if isinstance(classes, dict): classes = classes["classes"]
    model = ISLHybridConformerCTC(len(classes)).to(device)
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = extract_state_dict(raw)
    # The CTC head depends on the vocabulary size: a checkpoint trained on the
    # old class list cannot be loaded verbatim.  The mismatching head is
    # dropped (and reported), the backbone is kept.
    missing, unexpected = load_flexible(model, sd, label="LIVE")
    bad = [k for k in missing if not k.endswith("pe")] + list(unexpected)
    if bad:
        raise RuntimeError(f"Checkpoint does not match the model ({len(bad)} mismatched keys, e.g. {bad[:3]}). "
                           f"Is {ckpt_path} a Conformer-CTC checkpoint with {len(classes)} classes?")
    model.eval()
    return model, classes


# =============================================================================
# SIMULATION (no camera): drive the recogniser from tensors
# =============================================================================
def run_stream(rec, feats, hands_mask, fps=30.0, speeds=None, t0=0.0):
    """Push frames through a recogniser. Returns [(t, event), ...]."""
    out = []
    for i, f in enumerate(feats):
        t = t0 + i / fps
        sp = None if speeds is None else speeds[i]
        for ev in rec.push(t, np.asarray(f, dtype=np.float32), bool(hands_mask[i]), sp):
            out.append((t, ev))
    return out

def print_event(ev, t=None):
    stamp = f"[{t:6.2f}s] " if t is not None else ""
    kind = ev[0]
    if kind == "start":     print(f"{stamp}START")
    elif kind == "partial": print(f"{stamp}PARTIAL  {' '.join(ev[1])}  |  {' '.join(ev[2])}")
    elif kind == "final":   print(f"{stamp}FINAL    {' '.join(ev[1]) or '<nothing recognised>'}   ({ev[3]})")
    elif kind == "discard": print(f"{stamp}DISCARD  ({ev[1]})")

def simulate_tensor(rec, path, fps):
    seq = torch.load(path, weights_only=True).float().numpy()
    # The saved tensors are raw extractions; the recogniser was trained on the
    # harmonised frame (see training/train_conformer_ctc.py::harmonize_features).
    from training.train_conformer_ctc import harmonize_features, _reclamp
    seq = _reclamp(harmonize_features(torch.from_numpy(seq)).contiguous()).numpy()
    hands = (seq[:, 159] >= 0.5) | (seq[:, 160] >= 0.5)
    pad = int(rec.cfg.idle_sec * fps) + 15
    rest = np.zeros((pad, seq.shape[1]), dtype=np.float32)
    feats = np.concatenate([rest, seq, rest]); mask = np.concatenate([np.zeros(pad, bool), hands, np.zeros(pad, bool)])
    for t, ev in run_stream(rec, feats, mask, fps):
        print_event(ev, t)


# =============================================================================
# YAP-COMPATIBLE COORDINATE EXTRACTOR
# =============================================================================
# Layout: J=42  ->  left_hand [0:21] + right_hand [21:42], each 21-pt MediaPipe hand.
# conf = MediaPipe hand landmark presence score (approximated as 1.0 when hand detected,
#        0.0 when absent). valid = True only for joints whose hand was detected this frame.
# These match the YAPConfig default (two hand_parts) and feed directly into YAPStream.push().
_YAP_J = 42   # 2 hands × 21 joints
_YAP_MIN_CONF = 0.05  # mirrors YAPConfig.min_conf default


def extract_yap_frame(hands_result) -> tuple:
    """
    Extract YAP-compatible (pose, conf, valid) from a MediaPipe HandLandmarker result.

    Returns
    -------
    pose  : np.ndarray (42, 3) float32  — left_hand[0:21] + right_hand[21:42] in raw
            MediaPipe image-normalised coords (x, y in [0,1], z relative depth).
            Missing hands are zeroed; the corresponding valid flags are False.
    conf  : np.ndarray (42,)   float32  — landmark-level confidence. Set to 1.0 when
            the hand is detected (MediaPipe does not expose per-landmark confidence in
            the Tasks API), 0.0 when the hand is absent.
    valid : np.ndarray (42,)   bool     — True only for joints whose hand was found.

    These arrays are ready to be converted to tensors and passed to YAPStream.push():
        pose_t  = torch.from_numpy(pose)
        conf_t  = torch.from_numpy(conf)
        valid_t = torch.from_numpy(valid)
        entities = yap_stream.push(pose_t, conf_t, valid_t, t_seconds)
    """
    pose  = np.zeros((_YAP_J, 3), dtype=np.float32)
    conf  = np.zeros(_YAP_J,      dtype=np.float32)
    valid = np.zeros(_YAP_J,      dtype=bool)

    if hands_result and hands_result.hand_landmarks:
        for i, handedness in enumerate(hands_result.handedness):
            label = handedness[0].category_name   # "Left" or "Right"
            lms   = hands_result.hand_landmarks[i]
            # Use hand-level presence score if available, else 1.0
            hand_conf = float(getattr(handedness[0], "score", 1.0))
            if hand_conf < _YAP_MIN_CONF:
                continue
            offset = 0 if label == "Left" else 21
            for j, lm in enumerate(lms):
                pose[offset + j]  = [lm.x, lm.y, lm.z]
                conf[offset + j]  = hand_conf
                valid[offset + j] = True

    return pose, conf, valid


def run_camera(rec, source, display=True, mirror_view=True, on_yap_frame=None):
    """
    Main camera/video loop.

    on_yap_frame : callable(pose, conf, valid, t) | None
        If provided, called every frame with YAP-compatible raw coordinates
        (numpy arrays: pose (42,3) float32, conf (42,) float32, valid (42,) bool, t float seconds).
        Use this to pipe frames into a YAPStream for gesture segmentation once YAP is trained.
        Example::

            stream = YAPStream(yap_model)
            def on_frame(pose, conf, valid, t):
                pose_t  = torch.from_numpy(pose)
                conf_t  = torch.from_numpy(conf)
                valid_t = torch.from_numpy(valid)
                entities = stream.push(pose_t, conf_t, valid_t, t)
                for ent in entities:
                    print(\"entity:\", ent.reason, ent.start_t, ent.end_t)
            run_camera(rec, \"0\", on_yap_frame=on_frame)
    """
    import cv2
    import mediapipe as mp
    from features.extract_features import extract_frame_features_video_mode
    from training.train_conformer_ctc import harmonize_frame

    BaseOptions = mp.tasks.BaseOptions
    Mode = mp.tasks.vision.RunningMode.VIDEO
    mk = lambda cls, name, **kw: cls(base_options=BaseOptions(model_asset_path=os.path.join(ROOT_DIR, name)),
                                     running_mode=Mode, **kw)
    hand_opt = mk(mp.tasks.vision.HandLandmarkerOptions, "hand_landmarker.task", num_hands=2)
    pose_opt = mk(mp.tasks.vision.PoseLandmarkerOptions, "pose_landmarker.task")
    face_opt = mk(mp.tasks.vision.FaceLandmarkerOptions, "face_landmarker.task")

    is_file = isinstance(source, str) and not source.isdigit()
    cap = cv2.VideoCapture(source if is_file else int(source))
    if not cap.isOpened():
        print(f"Error: could not open source {source!r}"); return
    file_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    def fresh_state():
        return np.zeros((21, 3)), np.zeros((21, 3)), np.zeros((4, 3)), np.zeros((7, 3))
    last_lh, last_rh, last_arms, last_face = fresh_state()
    prev_pts, prev_t, last_ts_ms, frame_i = {}, None, -1, 0
    wall0 = time.monotonic()
    print("Running. Sign, then lower your hands to get the final sentence. Press 'q' to quit.")

    with mp.tasks.vision.HandLandmarker.create_from_options(hand_opt) as hand_lm, \
         mp.tasks.vision.PoseLandmarker.create_from_options(pose_opt) as pose_lm, \
         mp.tasks.vision.FaceLandmarker.create_from_options(face_opt) as face_lm:
        while True:
            ret, frame = cap.read()
            if not ret: break
            t = frame_i / file_fps if is_file else time.monotonic() - wall0
            ts_ms = max(int(t * 1000), last_ts_ms + 1); last_ts_ms = ts_ms   # MediaPipe needs strictly increasing
            frame_i += 1

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            hands = hand_lm.detect_for_video(mp_image, ts_ms)
            pose = pose_lm.detect_for_video(mp_image, ts_ms)
            face = face_lm.detect_for_video(mp_image, ts_ms)

            lh, rh = np.zeros((21, 3)), np.zeros((21, 3))
            lh_flag = rh_flag = False
            cur_pts = {}
            if hands and hands.hand_landmarks:
                for i, handed in enumerate(hands.handedness):
                    pts = np.array([[p.x, p.y, p.z] for p in hands.hand_landmarks[i]])
                    label = handed[0].category_name
                    cur_pts[label] = pts[:, :2]
                    if label == "Left": lh, lh_flag = pts, True
                    else:               rh, rh_flag = pts, True

            feat, last_lh, last_rh, last_arms, last_face = extract_frame_features_video_mode(
                lh, rh, lh_flag, rh_flag,
                pose.pose_landmarks if pose and pose.pose_landmarks else None,
                face.face_landmarks if face and face.face_landmarks else None,
                last_lh, last_rh, last_arms, last_face)

            # Put the frame in the same geometric frame as the training data, otherwise
            # the recogniser sees a skeleton displaced by more than three shoulder widths
            # and returns nothing (see training/train_conformer_ctc.py::harmonize_frame).
            feat = harmonize_frame(feat)

            # ── YAP-compatible raw coordinates (separate from the 163-dim CTC features) ──
            # pose_yap (42,3), conf_yap (42,), valid_yap (42,) — left_hand[0:21] + right_hand[21:42]
            # in raw MediaPipe image-normalised coords, ready for YAPStream.push().
            pose_yap, conf_yap, valid_yap = extract_yap_frame(hands)
            if on_yap_frame is not None:
                on_yap_frame(pose_yap, conf_yap, valid_yap, t)

            # Hand speed = mean landmark displacement per second (image-width units)
            speed = None
            if cur_pts and prev_t is not None:
                d = [np.abs(p - prev_pts[k]).mean() for k, p in cur_pts.items() if k in prev_pts]
                if d: speed = float(np.mean(d)) / max(t - prev_t, 1e-3)
            prev_pts, prev_t = cur_pts, t

            any_hand = lh_flag or rh_flag
            for ev in rec.push(t, feat, any_hand, speed):
                print_event(ev, t)
                if ev[0] == "reset":     # clean slate for the next sentence, like a fresh training clip
                    last_lh, last_rh, last_arms, last_face = fresh_state()
                    prev_pts, prev_t = {}, None

            if display:
                for pts in cur_pts.values():
                    for x, y in pts:
                        cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 3, (0, 200, 255), -1)
                if mirror_view: frame = cv2.flip(frame, 1)
                draw_hud(cv2, frame, rec)
                cv2.imshow("MookVani Live ISL (Conformer-CTC)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"): break
    cap.release()
    if display: cv2.destroyAllWindows()


def draw_hud(cv2, frame, rec):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 150), (15, 15, 15), -1)
    sign = rec.active
    cv2.circle(frame, (24, 24), 8, (0, 220, 0) if sign else (80, 80, 200), -1)
    cv2.putText(frame, f"{rec.status}  buf {len(rec.buf)}  infer {rec.last_infer_ms:.0f} ms", (42, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    if sign:
        x = 20
        for word in rec.committed:                               # solid = committed
            cv2.putText(frame, word, (x, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 140), 2, cv2.LINE_AA)
            x += cv2.getTextSize(word + " ", cv2.FONT_HERSHEY_SIMPLEX, 0.95, 2)[0][0]
        for word in rec.tentative:                               # dim = tentative
            cv2.putText(frame, word, (x, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (130, 130, 130), 2, cv2.LINE_AA)
            x += cv2.getTextSize(word + " ", cv2.FONT_HERSHEY_SIMPLEX, 0.95, 2)[0][0]
        if not rec.committed and not rec.tentative:
            cv2.putText(frame, "...", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (130, 130, 130), 2, cv2.LINE_AA)
    elif rec.final:
        cv2.putText(frame, " ".join(rec.final), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 230, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Sign, then lower your hands", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (110, 110, 110), 2, cv2.LINE_AA)
    for i, sent in enumerate(list(rec.history)[:2]):
        cv2.putText(frame, " ".join(sent)[:90], (20, 110 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    d = Config()
    p = argparse.ArgumentParser(description="Live ISL recognition with the real-sentence Conformer-CTC")
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--classes", default=CLASSES_PATH)
    p.add_argument("--source", default="0", help="webcam index or video file path")
    p.add_argument("--simulate", default=None, help="run a saved (T,163) tensor through the recogniser instead of a camera")
    p.add_argument("--device", default=None)
    p.add_argument("--no_display", action="store_true")
    p.add_argument("--no_mirror", action="store_true", help="do not mirror the preview window")
    p.add_argument("--blank_penalty", type=float, default=0.0, help="subtract from blank log-prob (>0 = fewer deletions)")
    for name in ["fps", "idle_sec", "still_sec", "still_speed", "lead_sec", "tail_sec", "min_sign_sec",
                 "max_sec", "infer_every_sec", "min_conf"]:
        p.add_argument(f"--{name}", type=float, default=getattr(d, name))
    for name in ["agree_n", "min_dur"]:
        p.add_argument(f"--{name}", type=int, default=getattr(d, name))
    return p.parse_args()

def main():
    a = parse_args()
    cfg = Config(**{k: getattr(a, k) for k in Config.__dataclass_fields__})
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, classes = load_model(a.ckpt, a.classes, device)
    print(f"[LOAD] {os.path.basename(a.ckpt)} | {len(classes)} classes | device {device}")
    rec = LiveRecognizer(model, classes, device, cfg, a.blank_penalty)
    if a.simulate:
        simulate_tensor(rec, a.simulate, cfg.fps)
    else:
        run_camera(rec, a.source, display=not a.no_display, mirror_view=not a.no_mirror)

if __name__ == "__main__":
    main()