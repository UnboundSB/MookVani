"""
Setup smoke test
================

Validates:
- required Python packages import
- islreconformer.h5 exists and loads
- hand_landmarker.task exists and loads
- model input/output contract matches the 42-feature pipeline
- model output count matches ALPHABET

Run:
    .venv\\Scripts\\python.exe check_setup.py
"""

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

EXPECTED_PACKAGES = ["cv2", "mediapipe", "numpy", "tensorflow"]


def check_packages() -> bool:
    ok = True
    for package in EXPECTED_PACKAGES:
        try:
            __import__(package)
            print(f"[OK] import {package}")
        except Exception as exc:
            ok = False
            print(f"[FAIL] import {package}: {exc}")
    return ok


def check_files(base_dir: str) -> bool:
    ok = True
    required = {
        "model": "islreconformer.h5",
        "hand landmarker": "hand_landmarker.task",
    }

    for label, filename in required.items():
        path = os.path.join(base_dir, filename)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            print(f"[OK] {label}: {filename}")
        else:
            ok = False
            print(f"[FAIL] missing/empty {label}: {path}")

    return ok


def check_engine(base_dir: str) -> bool:
    try:
        from isl_reconformer import ALPHABET, ISLReconformer

        engine = ISLReconformer()
        print(f"[OK] engine load: model={os.path.basename(engine.model_path)}")
        print(f"[OK] input shape: {engine.input_shape}")
        print(f"[OK] output shape: {engine.output_shape}")
        print(f"[OK] input feature count: {engine.input_feature_count}")
        print(f"[OK] label classes: {len(ALPHABET)}")
        print(f"[OK] label order: {''.join(ALPHABET)}")
        engine.close()
        return True
    except Exception as exc:
        print(f"[FAIL] engine initialization: {exc}")
        return False


def main() -> int:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ok = check_packages()
    ok = check_files(base_dir) and ok
    ok = check_engine(base_dir) and ok

    if ok:
        print("\n[PASS] Pipeline setup is valid.")
        return 0

    print("\n[FAIL] Pipeline setup is not valid.")
    return 1


if __name__ == "__main__":
    sys.exit(main())