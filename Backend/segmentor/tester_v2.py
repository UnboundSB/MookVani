"""
YAP v2 live continuous segmentation + word classification tester.
===================================================================

Uses YAPpingConformer (YAP front-end + Conformer word classifier) for
continuous, non-freezing live signing. Reports every detected segment
with its predicted word in real time.

Loads YAP weights from a Phase A checkpoint (yap_phaseA_best.pth) and
optionally Conformer weights from Phase B (yap_seg_best.pth).

Controls:
  SPACE  : pause / resume inference
  R      : reset stream state (flush + new YAPStream)
  Q/ESC  : quit
  S      : screenshot
  D      : toggle debug panel
  M      : toggle mirror view

Output:
  Appends every segment to <save-dir>/segments_v2.jsonl with:
    segment, reason, start_t, end_t, duration, reported_at,
    predicted_word, word_confidence, all_word_logits
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

BASE = Path(__file__).resolve().parent
for candidate in [BASE, BASE / "segmentor", BASE.parent / "segmentor", Path.cwd() / "segmentor"]:
    if (candidate / "yap_conformer.py").is_file():
        sys.path.insert(0, str(candidate))
        break
else:
    raise SystemExit("Cannot locate segmentor/yap_conformer.py")

from yap_conformer import (
    YAP, YAPpingConformer, YAPConfig, ConformerConfig,
    CORPUS_UPPER_BODY_LANDMARKS, CORPUS_FACE_LANDMARKS,
)

MODELS = {
    "hand": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "face": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}
HAND_EDGES = [(0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8),
              (5,9),(9,10),(10,11),(11,12), (9,13),(13,14),(14,15),(15,16),
              (13,17),(17,18),(18,19),(19,20),(0,17)]
COLORS = {"left": (255, 225, 60), "right": (70, 220, 255), "body": (130, 255, 110), "face": (255, 125, 235)}


def args_parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=BASE / "models/yap_stage2/yap_phaseA_best.pth")
    p.add_argument("--conformer-ckpt", type=Path, default=None,
                   help="Optional Phase B conformer checkpoint (yap_seg_best.pth)")
    p.add_argument("--classes", type=Path, default=BASE.parent / "models" / "word_vocabulary" / "classes.json")
    p.add_argument("--models-dir", type=Path, default=BASE / "models/mediapipe_tasks")
    p.add_argument("--download-models", action="store_true", help="Download MediaPipe Tasks assets if missing")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    p.add_argument("--threshold-on", type=float, default=None)
    p.add_argument("--threshold-cut", type=float, default=None)
    p.add_argument("--save-dir", type=Path, default=Path("yap_live_segments_v2"))
    return p.parse_args()


def load_classes(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_yap_model(checkpoint: Path, classes: list, device, conformer_ckpt: Path | None = None):
    """Build YAPpingConformer, load YAP weights from Phase A checkpoint,
    optionally load Conformer weights from Phase B checkpoint."""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {checkpoint}")
    try:
        obj = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"Could not load checkpoint {checkpoint}: {exc}") from exc
    state = obj.get("yap", obj) if isinstance(obj, dict) else obj
    if not isinstance(state, dict) or not any(k.startswith("core.") for k in state):
        raise ValueError("Expected a Phase A checkpoint with YAP core.* state_dict keys.")

    # Infer config from checkpoint
    in_weight = state["core.lift.0.weight"]
    d_model = in_weight.shape[0]
    n_mem = len({k.split(".")[2] for k in state if k.startswith("core.layers.") and k.split(".")[2].isdigit()})
    cnn_kernel = state["core.stem.dw"].shape[0]
    summary_channels = state["core.summary_head.weight"].shape[0]
    entity_dim = state["readout.0.weight"].shape[0]
    hippo_order = state["readout_norm.weight"].numel() // summary_channels

    ycfg = YAPConfig.corpus(d_model=d_model, cnn_kernel=cnn_kernel, n_mem_layers=n_mem,
                            summary_channels=summary_channels, hippo_order=hippo_order,
                            entity_dim=entity_dim)
    ccfg = ConformerConfig(num_words=len(classes), d_model=d_model,
                           heads=4, layers=4, conv_kernel=7, dropout=0.1)

    model = YAPpingConformer(ycfg, ccfg).to(device).eval()

    # Load YAP weights
    model.yap.load_state_dict(state, strict=True)
    print(f"[V2] Loaded YAP weights from {checkpoint}")

    # Optionally load Conformer + word head from Phase B
    if conformer_ckpt is not None and conformer_ckpt.is_file():
        ck = torch.load(conformer_ckpt, map_location="cpu", weights_only=True)
        ck_state = ck.get("yap", ck) if isinstance(ck, dict) else ck
        if isinstance(ck_state, dict):
            # Try to load conformer + head weights
            conformer_keys = {k: v for k, v in ck_state.items()
                              if k.startswith("ent_proj") or k.startswith("meta_proj")
                              or k.startswith("bos") or k.startswith("blocks") or k.startswith("head")}
            if conformer_keys:
                missing, unexpected = model.load_state_dict(conformer_keys, strict=False)
                print(f"[V2] Loaded Conformer weights from {conformer_ckpt} "
                      f"({len(conformer_keys)} tensors, {len(missing)} missing)")
            else:
                print(f"[V2] No Conformer keys found in {conformer_ckpt}; using random Conformer")
        else:
            print(f"[V2] Unexpected Phase B checkpoint format; using random Conformer")
    else:
        print(f"[V2] No Phase B checkpoint; Conformer initialized randomly")

    return model, ycfg


def make_landmarkers(model_dir):
    import mediapipe as mp
    BaseOptions = mp.tasks.BaseOptions
    vision = mp.tasks.vision
    def mk(kind):
        model_path = str(model_dir / (kind + ".task"))
        common = dict(base_options=BaseOptions(model_asset_path=model_path), running_mode=vision.RunningMode.VIDEO)
        if kind == "hand":
            return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(**common, num_hands=2,
                    min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5))
        if kind == "pose":
            return vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(**common,
                    num_poses=1, min_pose_detection_confidence=0.5, min_tracking_confidence=0.5))
        return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(**common, num_faces=1,
                    min_face_detection_confidence=0.5, min_tracking_confidence=0.5))
    return mp, mk("hand"), mk("pose"), mk("face")


def coords(landmarks, indices=None):
    pts = landmarks if indices is None else [landmarks[i] for i in indices]
    return np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float32)


def extract_parts(hands, pose_result, face_result, width):
    xyz = np.zeros((53, 3), np.float32)
    conf = np.zeros(53, np.float32)
    valid = np.zeros(53, dtype=np.bool_)
    parts = {}
    found_hands = set()
    for pts, handed in zip(hands.hand_landmarks, hands.handedness):
        if not handed:
            continue
        side = handed[0].category_name.lower()
        if side not in ("left", "right") or side in found_hands:
            continue
        found_hands.add(side)
        st = 0 if side == "left" else 21
        xyz[st:st+21] = coords(pts)
        conf[st:st+21] = 1.0
        valid[st:st+21] = True
        parts[side] = xyz[st:st+21].copy()
    if pose_result.pose_landmarks:
        landmarks = pose_result.pose_landmarks[0]
        bp = coords(landmarks, CORPUS_UPPER_BODY_LANDMARKS)
        visibility = np.array([getattr(landmarks[i], "visibility", 1.0) or 0.0
                               for i in CORPUS_UPPER_BODY_LANDMARKS], np.float32)
        xyz[42:46] = bp
        conf[42:46] = visibility
        valid[42:46] = np.isfinite(bp).all(axis=1) & (visibility > 0.15)
        parts["body"] = bp
    if face_result.face_landmarks:
        fp = coords(face_result.face_landmarks[0], CORPUS_FACE_LANDMARKS)
        xyz[46:53] = fp
        conf[46:53] = 1.0
        valid[46:53] = np.isfinite(fp).all(axis=1)
        parts["face"] = fp
    valid &= np.isfinite(xyz).all(axis=1)
    xyz[~valid] = 0.0
    conf[~valid] = 0.0
    return xyz, conf, valid, parts


def point(pt, w, h):
    return int(np.clip(pt[0] * w, 0, w-1)), int(np.clip(pt[1] * h, 0, h-1))


def draw_landmarks(frame, parts):
    h, w = frame.shape[:2]
    for side in ("left", "right"):
        if side not in parts:
            continue
        pts, color = parts[side], COLORS[side]
        for i, j in HAND_EDGES:
            cv2.line(frame, point(pts[i], w, h), point(pts[j], w, h), color, 2, cv2.LINE_AA)
        for i, pt in enumerate(pts):
            cv2.circle(frame, point(pt, w, h), 5 if i in (0, 4, 8, 12, 16, 20) else 3, color, -1, cv2.LINE_AA)
        x, y = point(pts[0], w, h)
        cv2.putText(frame, f"{side.upper()} HAND", (x + 8, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    if "body" in parts:
        pts = parts["body"]; color = COLORS["body"]
        for i, j in ((0, 1), (0, 2), (1, 3)):
            cv2.line(frame, point(pts[i], w, h), point(pts[j], w, h), color, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, point(p, w, h), 5, color, -1)
    if "face" in parts:
        for i, p in enumerate(parts["face"]):
            x, y = point(p, w, h)
            cv2.circle(frame, (x, y), 4, COLORS["face"], -1)
            cv2.putText(frame, str(i + 46), (x + 3, y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, COLORS["face"], 1)


def text(frame, s, x, y, color=(235, 235, 235), size=0.52, thick=1):
    cv2.putText(frame, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thick, cv2.LINE_AA)


def chart(frame, history, x, y, w, h, label, color, threshold):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (35, 37, 45), -1)
    ty = y + h - int(threshold * h)
    cv2.line(frame, (x, ty), (x + w, ty), (80, 85, 90), 1)
    if len(history) > 1:
        arr = list(history)
        coords = [(x + int(i * (w - 1) / max(1, len(arr) - 1)), y + h - int(np.clip(v, 0, 1) * h))
                  for i, v in enumerate(arr)]
        cv2.polylines(frame, [np.array(coords, np.int32)], False, color, 2, cv2.LINE_AA)
    text(frame, label, x + 6, y + 17, color, 0.45, 1)
    text(frame, f"{history[-1]:.3f}" if history else "---", x + w - 75, y + 17, (255, 255, 255), 0.45, 1)


def overlay(frame, stats, pres_hist, end_hist, events, cfg, debug=True, last_word=""):
    h, w = frame.shape[:2]
    panel_w = 400
    canvas = np.zeros((max(h, 540), w + panel_w, 3), np.uint8)
    canvas[:h, :w] = frame
    canvas[:, w:] = (25, 27, 33)
    x = w + 15
    text(canvas, "YAP v2 | LIVE SEGMENTATION + WORDS", x, 33, (245, 245, 245), 0.67, 2)
    text(canvas, "Phase A YAP + Conformer word classifier", x, 56, (160, 180, 195), 0.43)
    is_active = stats.get("active", False)
    status = "SEGMENT ACTIVE" if is_active else "WAITING / IDLE"
    state_color = (45, 205, 130) if is_active else (140, 150, 160)
    cv2.rectangle(canvas, (x, 70), (x + 365, 105), (35, 65, 45) if is_active else (48, 48, 48), -1)
    text(canvas, status, x + 12, 94, state_color, 0.58, 2)
    chart(canvas, pres_hist, x, 122, 365, 82, "GESTURE PRESENCE", (70, 220, 130), cfg.on_thr)
    chart(canvas, end_hist, x, 215, 365, 82, "END / CUT PROBABILITY", (80, 165, 255), cfg.cut_thr)
    text(canvas, f"Hand tracked: {stats.get('hands',0)}/2 | joints: {stats.get('valid',0)}/53", x, 325, (240, 240, 240), 0.50)
    text(canvas, f"Model hand_ok: {stats.get('ok',False)} | speed: {stats.get('speed',0):.3f}", x, 348, (185, 195, 205), 0.49)
    text(canvas, f"Completed segments: {stats.get('count',0)}", x, 375, (90, 230, 170), 0.55, 2)
    text(canvas, f"Processing FPS: {stats.get('fps',0):.1f}", x, 402, (180, 200, 215), 0.5)
    paused = stats.get("paused", False)
    text(canvas, f"PAUSED: {paused}", x, 425, (255, 255, 0) if paused else (240, 240, 240), 0.50)
    text(canvas, f"Last word: {last_word}", x, 450, (255, 220, 100), 0.55, 2)
    text(canvas, "SPACE pause | R reset | Q quit | S screenshot | D panel | M mirror", x, 520, (145, 160, 170), 0.43)
    if stats.get("valid", 0) == 0:
        text(canvas, "NO VALID LANDMARKS", 25, 35, (20, 30, 240), 0.63, 2)
    elif stats.get("hands", 0) == 0:
        text(canvas, "NO HANDS DETECTED", 25, 35, (25, 160, 255), 0.58, 2)
    if stats.get("just_cut"):
        cv2.rectangle(canvas, (2, 2), (w - 3, h - 3), (80, 240, 80), 6)
        text(canvas, "SEGMENT COMPLETED", 25, h - 28, (75, 245, 100), 0.8, 2)
    if not debug:
        return canvas[:h, :w]
    return canvas


def predict_word_from_entities(model, entities_buf, classes, device):
    """Run the Conformer on buffered entities to predict word labels."""
    if len(entities_buf) == 0:
        return "", 0.0, []
    # Stack entity vectors into a batch
    entity_vecs = torch.stack([e["vector"] for e in entities_buf], dim=0).unsqueeze(0).to(device)  # (1, S, D)
    entity_meta = torch.stack([e["meta"] for e in entities_buf], dim=0).unsqueeze(0).to(device)    # (1, S, 3)
    B, S, D = entity_vecs.shape
    d_model = model.ccfg.d_model
    tok = model.ent_proj(entity_vecs) + model.meta_proj(entity_meta)
    tok = tok + torch.sin(torch.arange(S, device=device).float().unsqueeze(1) /
                          torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-np.log(10000.0) / d_model)))
    x = torch.cat([model.bos.expand(B, -1, -1), tok], dim=1)
    mask = torch.ones(B, S + 1, dtype=torch.bool, device=device)
    for blk in model.blocks:
        x = blk(x, mask)
    logits = model.head(x[:, 1:])  # (1, S, V+1)
    probs = F.softmax(logits, dim=-1)
    # Predict word for each entity (skip blank class 0)
    pred_probs, pred_indices = probs[:, :, 1:].max(dim=-1)  # (1, S)
    pred_classes = pred_indices + 1  # shift back to include blank=0 offset
    words = []
    for i in range(S):
        ci = int(pred_classes[0, i].item())
        if ci == 0:
            words.append("[BLANK]")
        elif ci - 1 < len(classes):
            words.append(classes[ci - 1])
        else:
            words.append(f"[UNK:{ci}]")
    # Return the most confident non-blank prediction
    best_idx = int(pred_probs[0].argmax().item())
    best_word = words[best_idx] if best_idx < len(words) else ""
    best_conf = float(pred_probs[0, best_idx].item())
    return best_word, best_conf, words


def main():
    args = args_parse()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")
    device = "cuda" if (args.device == "cuda" or args.device == "auto" and torch.cuda.is_available()) else "cpu"

    classes = load_classes(args.classes)
    print(f"[V2] {len(classes)} classes loaded from {args.classes}")

    try:
        model, ycfg = load_yap_model(args.checkpoint, classes, device, args.conformer_ckpt)
    except Exception as exc:
        raise SystemExit(f"[CHECKPOINT ERROR] {exc}") from exc
    cfg = ycfg
    if args.threshold_on is not None:
        cfg.on_thr = args.threshold_on
    if args.threshold_cut is not None:
        cfg.cut_thr = args.threshold_cut
    print(f"[V2] checkpoint: {args.checkpoint} | device: {device} | J={cfg.total_joints}")
    print(f"[V2] on_thr={cfg.on_thr} off_thr={cfg.off_thr} cut_thr={cfg.cut_thr}")

    args.models_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for kind, url in MODELS.items():
        dest = args.models_dir / f"{kind}.task"
        if not dest.is_file() or dest.stat().st_size == 0:
            if args.download_models:
                print(f"[DOWNLOAD] {kind}: {url}")
                import urllib.request
                urllib.request.urlretrieve(url, dest)
            else:
                missing.append(str(dest))
    if missing:
        raise SystemExit("MediaPipe assets missing:\n  " + "\n  ".join(missing) + "\nRun once with --download-models")

    mp, hand, body, face = make_landmarkers(args.models_dir)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    stream = model.yap.stream(device=device)
    start = time.perf_counter()
    last_ts = -1
    hist_p = collections.deque(maxlen=180)
    hist_q = collections.deque(maxlen=180)
    events = collections.deque(maxlen=50)
    count = 0
    show_panel = True
    mirror = True
    flash_until = 0.0
    fps = 0.0
    paused = False
    args.save_dir.mkdir(parents=True, exist_ok=True)
    segments_path = args.save_dir / "segments_v2.jsonl"

    # Entity buffer for word classification (accumulates between segments)
    entity_buf = []
    last_word = ""
    last_word_conf = 0.0

    with hand, body, face, torch.inference_mode():
        try:
            while True:
                loop_start = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    print("[WARN] Camera frame failed")
                    break
                mirrored = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                ts = max(last_ts + 1, int((time.perf_counter() - start) * 1000))
                last_ts = ts
                hands = hand.detect_for_video(image, ts)
                poses = body.detect_for_video(image, ts)
                faces = face.detect_for_video(image, ts)
                xyz, confidence, valid, parts = extract_parts(hands, poses, faces, mirrored.shape[1])
                t_sec = ts / 1000.0

                if paused:
                    entities_out = []
                    out = stream.last_out
                    if out is None:
                        stream.push(torch.from_numpy(xyz), torch.from_numpy(confidence),
                                    torch.from_numpy(valid), t_sec)
                        out = stream.last_out
                else:
                    entities_out = stream.push(torch.from_numpy(xyz), torch.from_numpy(confidence),
                                               torch.from_numpy(valid), t_sec)

                current = time.perf_counter()

                # Collect entity vectors from buffer for word classification
                if stream.buf:
                    for item in stream.buf:
                        if item not in entity_buf:
                            entity_buf.append(item)

                if entities_out:
                    flash_until = current + 0.65
                    for e in entities_out:
                        count += 1
                        event = f"#{count} {e.reason} [{e.start_t:.2f}s-{e.end_t:.2f}s]"
                        events.append(event)
                        print("[SEGMENT]", event, f"duration={e.end_t - e.start_t:.2f}s")

                        # Predict word for this segment using buffered entities
                        seg_entities = [it for it in entity_buf
                                        if e.start_t - 0.1 <= it["t"] <= e.end_t + 0.1]
                        if seg_entities:
                            word, conf, all_words = predict_word_from_entities(
                                model, seg_entities, classes, device)
                            last_word = word
                            last_word_conf = conf
                        else:
                            word = ""
                            conf = 0.0
                            all_words = []

                        rec = {
                            "segment": count,
                            "reason": e.reason,
                            "start_t": round(float(e.start_t), 4),
                            "end_t": round(float(e.end_t), 4),
                            "duration": round(float(e.end_t - e.start_t), 4),
                            "reported_at": round(float(t_sec), 4),
                            "predicted_word": word,
                            "word_confidence": round(conf, 4),
                            "all_words": all_words,
                        }
                        with open(segments_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec) + "\n")

                        # Clear buffer after processing a segment
                        entity_buf = [it for it in entity_buf if it["t"] > e.end_t]

                out = stream.last_out
                if out is None:
                    stream.push(torch.from_numpy(xyz), torch.from_numpy(confidence),
                                torch.from_numpy(valid), t_sec)
                    out = stream.last_out
                if out is None:
                    continue

                p = float(torch.sigmoid(out["presence"][0]).item())
                q = float(torch.sigmoid(out["end"][0]).item())
                hist_p.append(p)
                hist_q.append(q)
                process_elapsed = max(time.perf_counter() - loop_start, 1e-6)
                fps = 0.85 * fps + 0.15 * (1 / process_elapsed) if fps else 1 / process_elapsed

                display = mirrored.copy()
                draw_landmarks(display, parts)
                stats = {"active": stream.tracker.active,
                         "hands": int("left" in parts) + int("right" in parts),
                         "valid": int(valid.sum()),
                         "ok": bool(out["hand_ok"][0].item() > 0.5),
                         "speed": float(out["speed"][0].item()),
                         "count": count,
                         "fps": fps,
                         "just_cut": current < flash_until,
                         "paused": paused}
                viz = overlay(display, stats, hist_p, hist_q, events, cfg, show_panel, last_word)
                if not mirror:
                    viz[:display.shape[0], :display.shape[1]] = cv2.flip(viz[:display.shape[0], :display.shape[1]], 1)
                cv2.imshow("YAP v2 | Live Segmentation + Words", viz)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    paused = not paused
                    print("[PAUSE]" if paused else "[RESUME]")
                if key == ord("d"):
                    show_panel = not show_panel
                if key == ord("m"):
                    mirror = not mirror
                if key == ord("s"):
                    filename = args.save_dir / f"yap_v2_{int(time.time())}.png"
                    cv2.imwrite(str(filename), viz)
                    print("[SCREENSHOT]", filename)
                if key == ord("r"):
                    for e in stream.flush():
                        print("[FLUSH]", e.reason, e.start_t, e.end_t)
                    stream = model.yap.stream(device=device)
                    hist_p.clear(); hist_q.clear(); events.clear()
                    count = 0; flash_until = 0.0; paused = False
                    entity_buf.clear()
                    last_word = ""
                    print("[RESET] Stream state cleared")
        finally:
            cap.release()
            cv2.destroyAllWindows()
    print("[SUMMARY] completed segments:", count)
    print(f"[SUMMARY] segments written to {segments_path}")


if __name__ == "__main__":
    main()
