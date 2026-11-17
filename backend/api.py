"""
ISL Reconformer Backend API
===========================

FastAPI backend for sending webcam/image frames from a React frontend.

Pipeline:
    camera frame
    -> MediaPipe hand landmarks
    -> ISL model prediction
    -> cache predicted characters while hands stay visible
    -> on no-hand gap: SymSpell segmentation/correction
    -> local text-to-speech sentence output

Run from the project root:
    .venv\\Scripts\\python.exe -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import base64
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Use the project-root inference engine without duplicating it.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isl_reconformer import ISLReconformer  # noqa: E402
from .session import ISLSession  # noqa: E402
from .spelling import SpellCorrector  # noqa: E402
from .speech import SpeechEngine  # noqa: E402


API_HOST = os.environ.get("ISL_API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("ISL_API_PORT", "8000"))
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "ISL_ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]


_engine: Optional[ISLReconformer] = None
_spelling: Optional[SpellCorrector] = None
_speech: Optional[SpeechEngine] = None
_session: Optional[ISLSession] = None
_engine_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _spelling, _speech, _session

    try:
        _engine = ISLReconformer()
        print("[STARTUP] ISL landmark/model engine loaded.")
    except Exception as exc:
        _engine = None
        print(f"[ERROR] Failed to load ISL engine: {exc}")

    try:
        _spelling = SpellCorrector()
    except Exception as exc:
        _spelling = None
        print(f"[ERROR] Failed to initialize spelling: {exc}")

    # The React frontend speaks through browser SpeechSynthesis.
    _speech = None
    print("[SPEECH] Backend speech disabled; frontend will use browser TTS.")

    if _engine is not None:
        _session = ISLSession(
            engine=_engine,
            spelling=_spelling,
            speech=_speech,
        )
        print("[STARTUP] ISL session/state pipeline ready.")

    yield

    if _speech is not None:
        _speech.stop()

    if _session is not None:
        _session = None

    if _engine is not None:
        _engine.close()
        _engine = None


app = FastAPI(
    title="ISL Reconformer API",
    description="React-to-model backend with cached sign spell correction and speech.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FrameRequest(BaseModel):
    image_base64: str = Field(..., min_length=10)


def decode_base64_frame(image_base64: str) -> np.ndarray:
    """Decode a data-URL/raw base64 JPEG into an OpenCV BGR frame."""
    payload = image_base64
    if "," in payload:
        payload = payload.split(",", 1)[1]

    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 frame: {exc}")

    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)

    if frame is None or frame.size == 0:
        raise HTTPException(status_code=400, detail="Unable to decode image frame")

    return frame


def api_payload(result: dict, session_state: dict | None = None) -> dict:
    response = {
        "ok": True,
        "prediction": result.get("prediction", "-"),
        "confidence": result.get("confidence", 0.0),
        "hands_count": result.get("hands_count", 0),
        "prediction_updated": result.get("prediction_updated", False),
        "hands": result.get("hands", []),
    }

    if session_state is not None:
        response["session"] = session_state

    return response


def require_session() -> ISLSession:
    if _session is None or _engine is None:
        raise HTTPException(
            status_code=503,
            detail="ISL engine/session is not ready",
        )
    return _session


@app.get("/")
async def root():
    return {
        "service": "ISL Reconformer API",
        "status": "ready" if _session is not None else "loading",
        "endpoints": [
            "/health",
            "/classes",
            "/predict/base64",
            "/predict",
            "/finalize",
            "/flush_spoken",
            "/reset",
            "/sentence",
            "/speech/state",
        ],
    }


@app.get("/health")
async def health():
    if _session is None or _engine is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Engine/session not ready"},
        )

    return {
        "status": "ready",
        "model": os.path.basename(_engine.model_path),
        "classes": int(_engine.model.output_shape[-1]),
        "input_mode": (
            "two_hand_combined"
            if _engine.input_feature_count == _engine.TWO_HAND_FEATURES
            else "single_hand"
        ),
        "input_shape": _engine.input_shape,
        "output_shape": _engine.output_shape,
        "input_feature_count": _engine.input_feature_count,
        "spelling": _spelling.state() if _spelling else {"enabled": False},
        "speech": _speech.state() if _speech else {"enabled": False},
        "session": _session.snapshot(),
    }


@app.get("/classes")
async def classes():
    from isl_reconformer import ALPHABET

    return {"classes": ALPHABET, "count": len(ALPHABET)}


@app.post("/predict/base64")
async def predict_base64(body: FrameRequest):
    require_session()
    frame = decode_base64_frame(body.image_base64)

    with _engine_lock:
        result = _engine.process_frame(frame)
        session_state = _session.process(result)

    return api_payload(result, session_state)


@app.post("/predict")
async def predict_file(file: UploadFile = File(...)):
    require_session()
    raw = await file.read()

    try:
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to decode upload: {exc}")

    if frame is None or frame.size == 0:
        raise HTTPException(status_code=400, detail="Empty or invalid image upload")

    with _engine_lock:
        result = _engine.process_frame(frame)
        session_state = _session.process(result)

    return api_payload(result, session_state)


@app.post("/finalize")
async def finalize_session():
    require_session()

    with _engine_lock:
        finalized = _session.finalize(force=True)

    return {"ok": True, "finalize": finalized}


@app.post("/flush_spoken")
async def flush_spoken_session(count: int = 0):
    require_session()

    with _engine_lock:
        snapshot = _session.flush_spoken(count if count > 0 else None)

    return {"ok": True, "session": snapshot}


@app.post("/reset")
async def reset_session():
    require_session()

    with _engine_lock:
        snapshot = _session.reset()

    return {"ok": True, "session": snapshot}


@app.get("/sentence")
async def sentence():
    require_session()

    with _engine_lock:
        snapshot = _session.snapshot()

    return {"ok": True, "session": snapshot}


@app.get("/speech/state")
async def speech_state():
    return {
        "ok": True,
        "speech": _speech.state() if _speech else {"enabled": False},
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.api:app", host=API_HOST, port=API_PORT, reload=False)