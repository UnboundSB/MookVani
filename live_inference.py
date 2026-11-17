"""
Live ISL Inference UI
=====================

Runs the ISL Reconformer engine against the default webcam.

Features
--------
- Local MJPEG stream at http://127.0.0.1:5000
- Optional local OpenCV HUD window
- Stable 1-second voting inside ISLReconformer
- Clean camera/shutdown handling

Run:
    .venv\\Scripts\\python.exe live_inference.py

Close with Q or Esc.
"""

from __future__ import annotations

import http.server
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2

from isl_reconformer import ALPHABET, ISLReconformer


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 17), (5, 9), (9, 13), (13, 17),
]

_JOINT_COLOR = (0, 230, 128)
_CONN_COLOR = (180, 180, 180)
_TARGET_FPS = 15

latest_jpeg: Optional[bytes] = None
_stream_lock = threading.Lock()

HTTP_HOST = os.environ.get("ISL_HTTP_HOST", "127.0.0.1")
HTTP_PORT = int(os.environ.get("ISL_HTTP_PORT", "5000"))
SHOW_CV_WINDOW = os.environ.get("SHOW_CV_WINDOW", "1") == "1"


def draw_skeleton_from_normalized(image, normalized_points: List[List[float]]) -> None:
    """Draw MediaPipe hand connections from normalized landmark coordinates."""
    if not normalized_points:
        return

    h, w = image.shape[:2]
    points = []
    for lm in normalized_points:
        x = max(0, min(w - 1, int(float(lm[0]) * w)))
        y = max(0, min(h - 1, int(float(lm[1]) * h)))
        points.append((x, y))

    for a, b in HAND_CONNECTIONS:
        if a < len(points) and b < len(points):
            cv2.line(image, points[a], points[b], _CONN_COLOR, 2, cv2.LINE_AA)

    for x, y in points:
        cv2.circle(image, (x, y), 5, _JOINT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(image, (x, y), 5, (30, 30, 30), 1, cv2.LINE_AA)


def draw_hud(image, label: str, confidence: float, fps: float, hand_count: int) -> None:
    H, W = image.shape[:2]
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (W, 90), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    cv2.putText(
        image, str(label), (25, 68),
        cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0, 240, 130), 4, cv2.LINE_AA,
    )

    conf = max(0.0, min(1.0, float(confidence)))
    bar_w = int((W - 190) * conf)
    cv2.rectangle(image, (180, 25), (180 + bar_w, 46), (0, 210, 110), -1)
    cv2.rectangle(image, (180, 25), (W - 15, 46), (150, 150, 150), 1)
    cv2.putText(
        image, f"{conf * 100:.1f}%", (W - 80, 42),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA,
    )

    cv2.putText(
        image,
        f"FPS: {fps:.1f} | Hands: {hand_count} | classes: {len(ALPHABET)}",
        (15, H - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )


class StreamingHandler(http.server.BaseHTTPRequestHandler):
    """Minimal localhost-only MJPEG stream handler."""

    protocol_version = "HTTP/1.0"

    def log_message(self, fmt, *args):
        # Keep the PowerShell console readable.
        return

    def do_GET(self):
        global latest_jpeg

        if self.path == "/":
            html = """<!DOCTYPE html>
<html>
<head>
  <title>ISL Live Inference</title>
  <style>
    body { background:#121212; color:#fff; font-family:sans-serif; text-align:center; margin:0; padding:20px; }
    h2 { color:#00e676; margin:10px 0; }
    .stream-box { display:inline-block; border:3px solid #333; border-radius:8px; overflow:hidden; background:#000; }
    img { display:block; width:640px; height:auto; }
    p { color:#aaa; font-size:14px; }
  </style>
</head>
<body>
  <h2>ISL Live Prediction Stream</h2>
  <div class="stream-box"><img src="/video_feed"></div>
  <p>Stable prediction updates every second. Press Q or Esc in the OpenCV window to exit.</p>
</body>
</html>"""
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path != "/video_feed":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
        self.end_headers()

        try:
            while True:
                with _stream_lock:
                    frame = latest_jpeg

                if frame is not None:
                    self.wfile.write(b"--FRAME\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")

                time.sleep(1.0 / 18.0)
        except Exception:
            # Client disconnected; that is normal for MJPEG.
            return


class _ThreadingHTTPServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_server() -> Tuple[Optional[_ThreadingHTTPServer], Optional[threading.Thread]]:
    try:
        server = _ThreadingHTTPServer((HTTP_HOST, HTTP_PORT), StreamingHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f"[INFO] Live web stream: http://{HTTP_HOST}:{HTTP_PORT}")
        return server, thread
    except OSError as exc:
        print(f"[WARN] Web stream disabled: {exc}")
        return None, None


def open_camera() -> cv2.VideoCapture:
    """Open the default camera with fallbacks and useful errors."""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open camera 0. Close other apps that may use the webcam, "
            "check Windows camera privacy permissions, then retry."
        )

    cap.set(cv2.CAP_PROP_FPS, _TARGET_FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


def safe_show_window(enabled: bool, window_name: str, image) -> bool:
    """Try OpenCV GUI without killing headless runs."""
    if not enabled:
        return False

    try:
        cv2.imshow(window_name, image)
        return True
    except Exception as exc:
        print(f"[WARN] OpenCV window disabled: {exc}")
        return False


def run() -> None:
    global latest_jpeg

    engine = ISLReconformer()
    cap = open_camera()
    server, _ = start_server()

    window_enabled = SHOW_CV_WINDOW
    if window_enabled:
        cv2.namedWindow("ISL Reconformer Live", cv2.WINDOW_AUTOSIZE)

    fps_times: List[float] = []
    last_time = time.time()

    print("[INFO] Pipeline ready. Press Q or Esc to exit.")

    try:
        while cap.isOpened():
            frame_start = time.time()

            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            # Selfie view for intuitive interaction.
            frame = cv2.flip(frame, 1)

            try:
                result = engine.process_frame(frame)
            except Exception as exc:
                print(f"[WARN] frame skipped: {exc}")
                continue

            if result.get("prediction_updated"):
                label = result.get("prediction", "-")
                conf = float(result.get("confidence", 0.0)) * 100.0
                print(f"[PREDICTION] {label} | confidence={conf:.1f}%")

            canvas = frame.copy()

            for hand in result.get("hands", []):
                draw_skeleton_from_normalized(canvas, hand.get("landmarks", []))

            fps_times.append(time.time())
            if len(fps_times) > 30:
                fps_times = fps_times[-30:]

            fps = (
                (len(fps_times) - 1) / (fps_times[-1] - fps_times[0])
                if len(fps_times) >= 2
                else 0.0
            )

            draw_hud(
                canvas,
                result.get("prediction", "-"),
                result.get("confidence", 0.0),
                fps,
                result.get("hands_count", 0),
            )

            try:
                ok_jpeg, jpeg = cv2.imencode(
                    ".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                )
                if ok_jpeg:
                    with _stream_lock:
                        latest_jpeg = jpeg.tobytes()
            except Exception as exc:
                print(f"[WARN] JPEG encode skipped: {exc}")

            if window_enabled:
                still_enabled = safe_show_window(True, "ISL Reconformer Live", canvas)
                if not still_enabled:
                    window_enabled = False
                    time.sleep(1.0 / _TARGET_FPS)
                    continue

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
            else:
                time.sleep(1.0 / _TARGET_FPS)

            elapsed = time.time() - frame_start
            wait = (1.0 / _TARGET_FPS) - elapsed
            if window_enabled and wait > 0:
                time.sleep(wait)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        cap.release()
        if window_enabled:
            cv2.destroyAllWindows()
        if server is not None:
            server.shutdown()
            server.server_close()
        engine.close()
        print("[INFO] Pipeline stopped cleanly.")


if __name__ == "__main__":
    run()