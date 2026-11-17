"""
Compatibility entry point.

The original standalone implementation used the removed MediaPipe Solutions API.
The active pipeline now lives in `live_inference.py` and `isl_reconformer.py`.

Run:
    .venv\\Scripts\\python.exe isl_detection.py
"""

from live_inference import run

if __name__ == "__main__":
    run()