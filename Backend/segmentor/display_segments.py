"""
Real-time word-by-word segmentor display.
Opens video, runs YAP segmentor, displays each detected word segment sequentially.
Press SPACE to advance to next word, Q to quit.
"""

import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from yap_conformer import YAPConfig, YAPpingConformer, ConformerConfig, hand_part

FPS = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
END_THRESHOLD = 0.45
MIN_SEGMENT_FRAMES = 5  # Minimum frames per segment

# ────────────────────────── MediaPipe Feature Extractor ──────────────────────────
class MediaPipeFeatureExtractor:
    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        model_path = Path(__file__).resolve().parent / "holistic_landmarker.task"
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.HolisticLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_face_detection_confidence=0.5,
            min_face_landmarks_confidence=0.5,
            min_pose_detection_confidence=0.5,
            min_pose_landmarks_confidence=0.5,
            min_hand_landmarks_confidence=0.5,
        )
        self.landmarker = vision.HolisticLandmarker.create_from_options(options)
    
    def extract_video(self, video_path: Path) -> np.ndarray:
        import mediapipe as mp
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_interval_ms = int(1000 / fps)
        features_list = []
        if not hasattr(self, '_global_frame_idx'):
            self._global_frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = self._global_frame_idx * frame_interval_ms
            self._global_frame_idx += 1
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
            feat = self._extract_frame(result)
            features_list.append(feat)
        cap.release()
        return np.stack(features_list, axis=0).astype(np.float32)
    
    def _extract_frame(self, result):
        lh = np.zeros((21, 3)); rh = np.zeros((21, 3))
        lh_flag = 0.0; rh_flag = 0.0
        if result.left_hand_landmarks:
            lh = np.array([[lm.x, lm.y, lm.z] for lm in result.left_hand_landmarks])
            lh_flag = 1.0
        if result.right_hand_landmarks:
            rh = np.array([[lm.x, lm.y, lm.z] for lm in result.right_hand_landmarks])
            rh_flag = 1.0
        arms = np.zeros((4, 3)); arms_flag = 0.0
        if result.pose_landmarks:
            p = result.pose_landmarks
            arms = np.array([
                [p[11].x, p[11].y, p[11].z], [p[12].x, p[12].y, p[12].z],
                [p[13].x, p[13].y, p[13].z], [p[14].x, p[14].y, p[14].z],
            ])
            arms_flag = 1.0
        face_pts = np.zeros((7, 3)); face_flag = 0.0
        if result.face_landmarks:
            f = result.face_landmarks
            face_pts = np.array([
                [f[159].x, f[159].y, f[159].z], [f[386].x, f[386].y, f[386].z],
                [f[4].x, f[4].y, f[4].z],
                [f[234].x, f[234].y, f[234].z], [f[454].x, f[454].y, f[454].z],
                [f[61].x, f[61].y, f[61].z], [f[291].x, f[291].y, f[291].z],
            ])
            face_flag = 1.0
        anchor = np.array([0.5, 0.5, 0.0])
        if face_flag: anchor = face_pts[2]
        elif arms_flag: anchor = (arms[0] + arms[1]) / 2.0
        elif lh_flag: anchor = lh[0]
        elif rh_flag: anchor = rh[0]
        scale = 0.3
        if arms_flag:
            pose_scale = np.linalg.norm(arms[0] - arms[1])
            if pose_scale > 0.05: scale = pose_scale
        elif face_flag:
            face_scale = np.linalg.norm(face_pts[3] - face_pts[4]) * 3.0
            if face_scale > 0.05: scale = face_scale
        lh_out = (lh - anchor) / scale if lh_flag else np.ones((21, 3)) * np.array([-1.5 * scale, 3.0 * scale, 0.0])
        rh_out = (rh - anchor) / scale if rh_flag else np.ones((21, 3)) * np.array([1.5 * scale, 3.0 * scale, 0.0])
        arms_out = (arms - anchor) / scale if arms_flag else np.zeros((4, 3))
        face_out = (face_pts - anchor) / scale if face_flag else np.zeros((7, 3))
        return np.concatenate([lh_out.flatten(), rh_out.flatten(), arms_out.flatten(), face_out.flatten(),
                               [lh_flag, rh_flag, arms_flag, face_flag]]).astype(np.float32)
    
    def close(self):
        self.landmarker.close()


# ────────────────────────── Skeleton Visualization ──────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8), (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16), (0,17),(17,18),(18,19),(19,20)
]
ARM_CONNECTIONS = [(0,1), (1,3), (0,2), (2,3)]
FACE_CONNECTIONS = [(0,2), (1,2), (2,3), (2,4), (3,4), (5,6)]

def extract_keypoints_2d(features: np.ndarray, frame_idx: int, img_w: int = 1280, img_h: int = 720):
    f = features[frame_idx]
    lh = f[0:63].reshape(21, 3)
    rh = f[63:126].reshape(21, 3)
    arms = f[126:138].reshape(4, 3)
    face = f[138:159].reshape(7, 3)
    flags = f[159:163]
    
    def to_pixel(pts_3d):
        if len(pts_3d) == 0:
            return np.zeros((0, 2), dtype=int)
        x = np.clip((pts_3d[:, 0] + 2.0) / 4.0 * img_w, 0, img_w - 1).astype(int)
        y = np.clip((pts_3d[:, 1] + 1.0) / 6.0 * img_h, 0, img_h - 1).astype(int)
        return np.stack([x, y], axis=1)
    
    return {
        'lh': to_pixel(lh), 'rh': to_pixel(rh),
        'arms': to_pixel(arms), 'face': to_pixel(face),
        'flags': flags
    }

def draw_skeleton(frame: np.ndarray, keypoints: dict, color: tuple = (0, 255, 0)):
    flags = keypoints['flags']
    lh_flag, rh_flag, arms_flag, face_flag = flags
    
    for name, pts, flag in [('lh', keypoints['lh'], lh_flag), ('rh', keypoints['rh'], rh_flag)]:
        if flag > 0.5 and len(pts) == 21:
            for a, b in HAND_CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    cv2.line(frame, tuple(pts[a]), tuple(pts[b]), color, 2)
            for pt in pts:
                cv2.circle(frame, tuple(pt), 3, color, -1)
    
    arms = keypoints['arms']
    if arms_flag > 0.5 and len(arms) == 4:
        for a, b in ARM_CONNECTIONS:
            cv2.line(frame, tuple(arms[a]), tuple(arms[b]), color, 3)
        for pt in arms:
            cv2.circle(frame, tuple(pt), 5, color, -1)
        lh = keypoints['lh']
        rh = keypoints['rh']
        if lh_flag > 0.5 and len(lh) == 21:
            cv2.line(frame, tuple(lh[0]), tuple(arms[0]), color, 2)
        if rh_flag > 0.5 and len(rh) == 21:
            cv2.line(frame, tuple(rh[0]), tuple(arms[1]), color, 2)
    
    face = keypoints['face']
    if face_flag > 0.5 and len(face) == 7:
        for a, b in FACE_CONNECTIONS:
            cv2.line(frame, tuple(face[a]), tuple(face[b]), color, 2)
        for pt in face:
            cv2.circle(frame, tuple(pt), 3, color, -1)


# ────────────────────────── Word-by-Word Display ──────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8), (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16), (0,17),(17,18),(18,19),(19,20)
]
ARM_CONNECTIONS = [(0,1), (1,3), (0,2), (2,3)]
FACE_CONNECTIONS = [(0,2), (1,2), (2,3), (2,4), (3,4), (5,6)]

def display_word_segments(video_path: Path):
    """Display video segmented word-by-word. SPACE=next word, Q=quit."""
    
    # Load model
    cfg = YAPConfig(parts=[hand_part("left_hand"), hand_part("right_hand")])
    conformer_cfg = ConformerConfig(num_words=228)
    model = YAPpingConformer(
        YAPConfig(parts=[hand_part("left_hand"), hand_part("right_hand")]),
        ConformerConfig(num_words=228)
    ).to(DEVICE)
    ckpt_path = Path(r"D:\MookVani\Backend\models\yap_segmentor\yap_segmentor_final.pth")
    model = YAPpingConformer(
        YAPConfig(parts=[hand_part("left_hand"), hand_part("right_hand")]),
        ConformerConfig(num_words=228)
    ).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    
    # Extract features
    extractor = MediaPipeFeatureExtractor()
    features = extractor.extract_video(video_path)
    T = len(features)
    extractor.close()
    
    if T == 0:
        print("No frames extracted")
        return
    
    features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(DEVICE)
    B, T, _ = features_tensor.shape
    
    # Inference
    with torch.no_grad():
        yap_out = model.yap.forward_frames(
            pose=features_tensor[:, :, :126].reshape(1, T, 42, 3),
            conf=torch.ones(1, T, 42, device=DEVICE),
            valid=torch.ones(1, T, 42, dtype=torch.bool, device=DEVICE),
            t=torch.arange(T, device=DEVICE).float().unsqueeze(0) / FPS,
            reset_mask=None,
        )
    
    presence = torch.sigmoid(yap_out['presence'][0]).cpu().numpy()
    end = torch.sigmoid(yap_out['end'][0]).cpu().numpy()
    pred_boundaries = np.where(end > 0.45)[0]
    
    # Filter boundaries: minimum segment length
    filtered_boundaries = []
    last_boundary = 0
    for b in pred_boundaries:
        if b - last_boundary >= 5:  # min 5 frames per segment
            filtered_boundaries.append(b)
            last_boundary = b
    
    print(f"Video: {video_path.name}")
    print(f"Frames: {T} ({T/FPS:.2f}s)")
    print(f"Segments detected: {len(filtered_boundaries)}")
    print(f"Boundaries at frames: {filtered_boundaries}")
    
    # Extract original video frames for display
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(frame, (1280, 720)))
    cap.release()
    T_orig = len(frames)
    
    # Build segments: each segment is [start_frame, end_frame)
    segments = []
    start = 0
    for b in filtered_boundaries:
        if b < T:
            segments.append((start, min(b, T_orig - 1)))
            start = b
    if start < T_orig:
        segments.append((start, T_orig - 1))
    
    print(f"\nSegments ({len(segments)}):")
    for i, (s, e) in enumerate(segments):
        print(f"  Word {i+1}: frames {s}-{e} ({(e-s)/FPS:.2f}s)")
    
    # ────────────────────────── Display Each Word ──────────────────────────
    print("\nPress SPACE for next word, Q to quit")
    
    for word_idx, (seg_start, seg_end) in enumerate(segments):
        if seg_end <= seg_start:
            continue
        
        print(f"\n{'='*60}")
        print(f"WORD {word_idx+1}/{len(segments)}: frames {seg_start}-{seg_end} ({(seg_end-seg_start)/FPS:.2f}s)")
        print(f"Press SPACE for next word, Q to quit")
        
        # Play this segment
        for frame_idx in range(seg_start, min(seg_end + 1, len(frames))):
            frame = frames[frame_idx].copy()
            
            # Draw skeleton from features (if available)
            if frame_idx < len(features):
                f = features[frame_idx]
                lh = f[0:63].reshape(21, 3)
                rh = f[63:126].reshape(21, 3)
                arms = f[126:138].reshape(4, 3)
                face = f[138:159].reshape(7, 3)
                flags = f[159:163]
                
                def to_pixel(pts_3d, img_w=1280, img_h=720):
                    if len(pts_3d) == 0:
                        return np.zeros((0, 2), dtype=int)
                    x = np.clip((pts_3d[:, 0] + 2.0) / 4.0 * 1280, 0, 1279).astype(int)
                    y = np.clip((pts_3d[:, 1] + 1.0) / 6.0 * 720, 0, 719).astype(int)
                    return np.stack([x, y], axis=1)
                
                lh_pts = to_pixel(lh)
                rh_pts = to_pixel(rh)
                arms_pts = to_pixel(arms)
                face_pts = to_pixel(face)
                
                # Draw skeleton
                for pts, flag in [(lh_pts, f[159]), (rh_pts, f[160])]:
                    if flag > 0.5 and len(pts) == 21:
                        for a, b in HAND_CONNECTIONS:
                            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (0, 255, 0), 2)
                        for pt in pts:
                            cv2.circle(frame, tuple(pt), 3, (0, 255, 0), -1)
                
                if f[162] > 0.5 and len(arms_pts) == 4:
                    arms_pts = to_pixel(f[126:138].reshape(4, 3))
                    for a, b in ARM_CONNECTIONS:
                        cv2.line(frame, tuple(arms_pts[a]), tuple(arms_pts[b]), (0, 255, 0), 3)
                    for pt in arms_pts:
                        cv2.circle(frame, tuple(pt), 5, (0, 255, 0), -1)
                    if features[frame_idx][159] > 0.5:
                        lh_pts = to_pixel(features[frame_idx][0:63].reshape(21, 3))
                        cv2.line(frame, tuple(lh_pts[0]), tuple(arms_pts[0]), (0, 255, 0), 2)
                    if features[frame_idx][160] > 0.5:
                        rh_pts = to_pixel(features[frame_idx][63:126].reshape(21, 3))
                        cv2.line(frame, tuple(rh_pts[0]), tuple(arms_pts[1]), (0, 255, 0), 2)
            
            # Overlay
            cv2.putText(frame, f"WORD {word_idx+1}/{len(segments)}", (30, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(frame, f"Frames: {seg_start}-{seg_end} ({(seg_end-seg_start)/FPS:.2f}s)", (30, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, f"SPACE=next word  Q=quit", (30, 680), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
            
            cv2.imshow("YAP Segmentor - Word by Word", frame)
            
            key = cv2.waitKey(40) & 0xFF  # ~25 FPS
            if key == ord('q') or key == 27:
                cv2.destroyAllWindows()
                return False
            elif key == 32:  # SPACE
                break  # Next word
        
        cv2.destroyAllWindows()
    
    print("\n[DONE] All words displayed")
    return True


if __name__ == "__main__":
    VIDEO_DIR = Path(r"D:\MookVani\backend\data\ISL_CSLRT_Corpus\Videos_Sentence_Level")
    test_videos = [
        VIDEO_DIR / "are you free today" / "are you free today (2).mp4",
        VIDEO_DIR / "can i help you" / "can i help you.mp4",
    ]
    
    for video_path in test_videos:
        if not video_path.exists():
            print(f"[SKIP] Not found: {video_path}")
            continue
        
        extractor = MediaPipeFeatureExtractor()
        continue_test = display_word_segments(video_path)
        if not continue_test:
            break
    
    print("\n[DONE]")