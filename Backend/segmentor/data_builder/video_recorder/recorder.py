"""
Sentence Video Recorder — Data Builder Module
Records user signing via webcam, extracts MediaPipe coordinates, stores only .npy features.
Videos are deleted after processing.
"""

import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import cv2
import json
import time
import random
import threading
import queue
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import mediapipe as mp

# ────────────────────────── paths ──────────────────────────
PKG_DIR = Path(__file__).resolve().parent
APP_DIR = PKG_DIR.parent

CLASSES_JSON = APP_DIR / "classes.json"
WORD_INVENTORY_JSON = APP_DIR / "word_inventory.json"
CLASSES_JSON_FALLBACK = Path(r"D:\MookVani\Backend\models\word_vocabulary\classes.json")
WORD_INVENTORY_JSON_FALLBACK = Path(r"D:\MookVani\Backend\segmentor\data_builder\word_inventory.json")

# Archive zip (source of archive_extracted)
ARCHIVE_ZIP = Path(r"F:\dataset\archive.zip")

# Output: organized by sentence words, only .npy kept
FEATURES_DIR = APP_DIR / "conformer_features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

FREQUENCY_JSON = PKG_DIR / "frequency_tracker.json"
CHECKPOINT_JSON = PKG_DIR / "recorder_checkpoint.json"

# ────────────────────────── constants ──────────────────────────
TARGET_CYCLES = 150
WORDS_PER_SENTENCE = (2, 5)
PREP_SECONDS = 5
FPS = 30
FOURCC = cv2.VideoWriter_fourcc(*'mp4v')
TEMP_VIDEO_DIR = PKG_DIR / "temp_videos"
TEMP_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# ────────────────────────── load resources ──────────────────────────
def load_classes() -> List[str]:
    for p in (CLASSES_JSON, CLASSES_JSON_FALLBACK):
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError("classes.json not found")

def load_word_inventory() -> Dict:
    for p in (WORD_INVENTORY_JSON, WORD_INVENTORY_JSON_FALLBACK):
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError("word_inventory.json not found")

def load_frequencies() -> Dict[str, int]:
    if FREQUENCY_JSON.exists():
        return json.loads(FREQUENCY_JSON.read_text(encoding="utf-8"))
    return {}

def save_frequencies(freq: Dict[str, int]):
    FREQUENCY_JSON.write_text(json.dumps(freq, indent=2), encoding="utf-8")

def load_checkpoint() -> int:
    if CHECKPOINT_JSON.exists():
        try:
            data = json.loads(CHECKPOINT_JSON.read_text(encoding="utf-8"))
            return int(data.get("cycle", 0))
        except Exception:
            return 0
    return 0

def save_checkpoint(cycle: int):
    CHECKPOINT_JSON.write_text(json.dumps({"cycle": cycle, "updated_at": datetime.now().isoformat()}, indent=2), encoding="utf-8")

# ────────────────────────── word selection ──────────────────────────
class WordSelector:
    def __init__(self, classes: List[str], inventory: Dict):
        self.classes = classes
        self.inventory = inventory
        self.freq = load_frequencies()
        for c in classes:
            self.freq.setdefault(c, 0)
        save_frequencies(self.freq)
    
    def _available_words(self) -> List[str]:
        return [w for w in self.classes if w in self.inventory and self.inventory[w]["videos"]]
    
    def select_words(self) -> List[str]:
        available = self._available_words()
        if not available:
            raise RuntimeError("No words with videos available")
        
        n = random.randint(*WORDS_PER_SENTENCE)
        weights = [1.0 / (self.freq[w] + 1) for w in available]
        selected = random.choices(available, weights=weights, k=min(n, len(available)))
        # Deduplicate preserving order
        seen = set()
        unique = []
        for w in selected:
            if w not in seen:
                unique.append(w)
                seen.add(w)
        return unique
    
    def increment(self, words: List[str]):
        for w in words:
            self.freq[w] = self.freq.get(w, 0) + 1
        save_frequencies(self.freq)

# ────────────────────────── MediaPipe extractor (Tasks API v1.0+) ──────────────────────────
class MediaPipeExtractor:
    def __init__(self):
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        # Download model if needed
        import urllib.request
        model_url = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task"
        model_path = PKG_DIR / "holistic_landmarker.task"
        if not model_path.exists():
            print(f"[EXTRACT] Downloading MediaPipe holistic model...")
            urllib.request.urlretrieve(model_url, model_path)
        
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.HolisticLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_face_detection_confidence=0.5,
            min_face_landmarks_confidence=0.5,
            min_pose_detection_confidence=0.5,
            min_pose_landmarks_confidence=0.5,
            min_hand_detection_confidence=0.5,
            min_hand_landmarks_confidence=0.5,
            output_face_blendshapes=False,
            output_segmentation_mask=False,
        )
        self.landmarker = vision.HolisticLandmarker.create_from_options(options)
    
    def extract_video(self, video_path: Path, output_path: Path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[EXTRACT] Cannot open {video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval_ms = int(1000 / fps)
        
        features_list = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = frame_idx * frame_interval_ms
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
            feat = self._extract_frame(result)
            features_list.append(feat)
            frame_idx += 1
        
        cap.release()
        if features_list:
            arr = np.stack(features_list, axis=0).astype(np.float32)  # (T, 163)
            np.save(output_path, arr)
            print(f"[EXTRACT] Saved {output_path} shape={arr.shape}")
    
    def _extract_frame(self, result) -> np.ndarray:
        # result: HolisticLandmarkerResult
        # pose_landmarks: List[List[NormalizedLandmark]]
        # left_hand_landmarks: List[List[NormalizedLandmark]]
        # right_hand_landmarks: List[List[NormalizedLandmark]]
        # face_landmarks: List[List[NormalizedLandmark]]
        
        lh = np.zeros((21, 3)); rh = np.zeros((21, 3))
        lh_flag = 0.0; rh_flag = 0.0
        
        if result.left_hand_landmarks:
            lh = np.array([[lm.x, lm.y, lm.z] for lm in result.left_hand_landmarks[0]])
            lh_flag = 1.0
        if result.right_hand_landmarks:
            rh = np.array([[lm.x, lm.y, lm.z] for lm in result.right_hand_landmarks[0]])
            rh_flag = 1.0
        
        arms = np.zeros((4, 3)); arms_flag = 0.0
        if result.pose_landmarks:
            p = result.pose_landmarks[0]
            arms = np.array([
                [p[11].x, p[11].y, p[11].z], [p[12].x, p[12].y, p[12].z],
                [p[13].x, p[13].y, p[13].z], [p[14].x, p[14].y, p[14].z],
            ])
            arms_flag = 1.0
        
        face_pts = np.zeros((7, 3)); face_flag = 0.0
        if result.face_landmarks:
            f = result.face_landmarks[0]
            face_pts = np.array([
                [f[159].x, f[159].y, f[159].z], [f[386].x, f[386].y, f[386].z],
                [f[4].x, f[4].y, f[4].z],
                [f[234].x, f[234].y, f[234].z], [f[454].x, f[454].y, f[454].z],
                [f[61].x, f[61].y, f[61].z],    [f[291].x, f[291].y, f[291].z],
            ])
            face_flag = 1.0
        
        lh_out, rh_out, arms_out, face_out = self._normalize(lh, rh, arms, face_pts, lh_flag, rh_flag, arms_flag, face_flag)
        return np.concatenate([lh_out.flatten(), rh_out.flatten(), arms_out.flatten(), face_out.flatten(),
                               [lh_flag, rh_flag, arms_flag, face_flag]]).astype(np.float32)
    
    def _normalize(self, lh, rh, arms, face_pts, lh_flag, rh_flag, arms_flag, face_flag):
        anchor = np.array([0.5, 0.5, 0.0])
        if face_flag:
            anchor = face_pts[2]
        elif arms_flag:
            anchor = (arms[0] + arms[1]) / 2.0
        elif lh_flag:
            anchor = lh[0]
        elif rh_flag:
            anchor = rh[0]
        
        scale = 0.3
        if arms_flag:
            pose_scale = np.linalg.norm(arms[0] - arms[1])
            if pose_scale > 0.05:
                scale = pose_scale
        elif face_flag:
            face_scale = np.linalg.norm(face_pts[3] - face_pts[4]) * 3.0
            if face_scale > 0.05:
                scale = face_scale
        
        lh_out = (lh - anchor) / scale if lh_flag else np.ones((21, 3)) * np.array([-1.5 * scale, 3.0 * scale, 0.0])
        rh_out = (rh - anchor) / scale if rh_flag else np.ones((21, 3)) * np.array([1.5 * scale, 3.0 * scale, 0.0])
        arms_out = (arms - anchor) / scale if arms_flag else np.zeros((4, 3))
        face_out = (face_pts - anchor) / scale if face_flag else np.zeros((7, 3))
        return lh_out, rh_out, arms_out, face_out
    
    def close(self):
        self.landmarker.close()

# ────────────────────────── background extraction worker ──────────────────────────
class ExtractionWorker:
    def __init__(self):
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
    
    def _run(self):
        extractor = MediaPipeExtractor()
        while True:
            item = self.queue.get()
            if item is None:
                break
            video_path, npy_path = item
            try:
                extractor.extract_video(video_path, npy_path)
                # Delete video after successful extraction
                video_path.unlink(missing_ok=True)
            except Exception as e:
                print(f"[EXTRACT ERROR] {e}")
            self.queue.task_done()
        extractor.close()
    
    def submit(self, video_path: Path, npy_path: Path):
        self.queue.put((video_path, npy_path))
    
    def shutdown(self):
        self.queue.put(None)
        self.thread.join()

# ────────────────────────── main recorder UI ──────────────────────────
class SentenceRecorder:
    def __init__(self):
        self.classes = load_classes()
        self.inventory = load_word_inventory()
        self.selector = WordSelector(self.classes, self.inventory)
        self.extractor = ExtractionWorker()
        self.cycle = load_checkpoint()
        self.window = "Sentence Recorder"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, 1280, 720)
        if self.cycle > 0:
            print(f"[RESUME] Starting from cycle {self.cycle + 1}")
    
    def _make_feature_path(self, words: List[str]) -> Path:
        """Create organized path: conformer_features/Hello_My/Hello_My_001.npy"""
        sentence_key = "_".join(words).replace(" ", "_")
        sentence_dir = FEATURES_DIR / sentence_key
        sentence_dir.mkdir(parents=True, exist_ok=True)
        existing = list(sentence_dir.glob("*.npy"))
        next_num = len(existing) + 1
        return sentence_dir / f"{sentence_key}_{next_num:03d}.npy"
    
    def _play_word_videos(self, words: List[str]) -> bool:
        for word in words:
            vids = self.inventory[word]["videos"]
            if not vids:
                continue
            cap = cv2.VideoCapture(vids[0])
            if not cap.isOpened():
                continue
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            delay = int(1000 / fps)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.putText(frame, f"WORD: {word}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.putText(frame, f"Sentence: {' '.join(words)}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(frame, "SPACE = Ready to record   R = Rewatch", (30, 660), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
                cv2.imshow(self.window, frame)
                key = cv2.waitKey(delay) & 0xFF
                if key == ord('r') or key == ord('R'):
                    cap.release()
                    return False
                if key == 32:
                    cap.release()
                    return True
            cap.release()
        return True
    
    def _countdown(self, cap) -> bool:
        for i in range(PREP_SECONDS, 0, -1):
            ret, frame = cap.read()
            if not ret:
                return False
            cv2.putText(frame, f"GET READY: {i}", (300, 360), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
            cv2.putText(frame, "Position yourself in frame", (200, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow(self.window, frame)
            if cv2.waitKey(1000) & 0xFF in (ord('q'), ord('Q'), 27):
                return False
        return True
    
    def _record(self, cap, words: List[str]) -> Optional[Path]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sentence_key = "_".join(words).replace(" ", "_")
        temp_name = f"temp_{self.cycle:04d}_{sentence_key}_{timestamp}.mp4"
        video_path = TEMP_VIDEO_DIR / temp_name
        
        writer = cv2.VideoWriter(str(video_path), FOURCC, FPS, (1280, 720))
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            elapsed = time.time() - start_time
            cv2.putText(frame, f"RECORDING: {elapsed:.1f}s", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(frame, f"Sentence: {' '.join(words)}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, "SPACE = Stop recording", (30, 660), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
            
            writer.write(frame)
            cv2.imshow(self.window, frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 32:
                break
            if key in (ord('q'), ord('Q'), 27):
                writer.release()
                video_path.unlink(missing_ok=True)
                return None
        
        writer.release()
        return video_path
    
    def _post_record_prompt(self) -> str:
        while True:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(frame, "RECORDING COMPLETE", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
            cv2.putText(frame, "R = Re-record this sentence", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            cv2.putText(frame, "SPACE = Next sentence", (200, 380), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, "Q = Quit", (200, 460), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.imshow(self.window, frame)
            key = cv2.waitKey(100) & 0xFF
            if key in (ord('r'), ord('R')):
                return "rerecord"
            if key == 32:
                return "next"
            if key in (ord('q'), ord('Q'), 27):
                return "quit"
    
    def run(self):
        print(f"[INFO] Starting Sentence Recorder — target {TARGET_CYCLES} cycles")
        print(f"[INFO] Features dir: {FEATURES_DIR}")
        print(f"[INFO] Classes: {len(self.classes)} | Inventory words: {len([w for w in self.classes if w in self.inventory])}")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Cannot open webcam")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        
        try:
            while self.cycle < TARGET_CYCLES:
                self.cycle += 1
                words = self.selector.select_words()
                print(f"\n=== CYCLE {self.cycle}/{TARGET_CYCLES} ===")
                print(f"Words: {words}")
                print(f"Frequencies: {{w: self.selector.freq[w] for w in words}}")
                
                # Preview loop
                while True:
                    if self._play_word_videos(words):
                        break
                
                # Countdown
                if not self._countdown(cap):
                    break
                
                # Record to temp
                video_path = self._record(cap, words)
                if video_path is None:
                    break
                
                # Queue extraction (will delete video after)
                npy_path = self._make_feature_path(words)
                self.extractor.submit(video_path, npy_path)
                
                # Post-record prompt
                action = self._post_record_prompt()
                if action == "rerecord":
                    self.cycle -= 1
                    continue
                elif action == "quit":
                    break
                
                # Commit frequencies
                self.selector.increment(words)
                save_checkpoint(self.cycle)
                
                freq_sorted = sorted(self.selector.freq.items(), key=lambda x: x[1])
                print(f"[FREQ] Min: {freq_sorted[0]}, Max: {freq_sorted[-1]}, Avg: {sum(self.selector.freq.values())/len(self.selector.freq):.1f}")
        
        finally:
            cap.release()
            self.extractor.shutdown()
            cv2.destroyAllWindows()
            print("[INFO] Recorder stopped")

def main():
    print("=" * 60)
    print("MookVani Sentence Video Recorder")
    print("=" * 60)
    recorder = SentenceRecorder()
    recorder.run()

if __name__ == "__main__":
    main()