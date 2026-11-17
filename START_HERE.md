# ISL Reconformer — Start Here

The project now has a React frontend and a FastAPI backend.

## 1. Create/use the virtual environment

Run from the project root:

```powershell
cd D:\islcsltr\Indian-Sign-Language-Detection
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
.\.venv\Scripts\python.exe -m pip install -r .\backend\requirements.txt
```

## 2. Start the backend

Run from the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.api:app --host 127.0.0.1 --port 8000 --reload
```

Backend root:

```text
http://127.0.0.1:8000
```

## 3. Start the React frontend

Run from the frontend folder:

```powershell
cd D:\islcsltr\Indian-Sign-Language-Detection\frontend
npm install
npm run dev
```

Frontend:

```text
http://127.0.0.1:5173
```

The frontend sends JPEG frames to:

```text
http://127.0.0.1:8000/predict/base64
```

## Existing local pipeline

The native OpenCV pipeline is still available:

```powershell
.\.venv\Scripts\python.exe live_inference.py
```

## Setup validation

Before camera/frontend use, validate the model and MediaPipe engine:

```powershell
.\.venv\Scripts\python.exe check_setup.py
```

## Project layout

```text
backend/
    FastAPI API for React frame inference.
frontend/
    React + Vite camera app.
isl_reconformer.py
    Core engine: landmark extraction, normalization, prediction, voting.
live_inference.py
    Native OpenCV fallback pipeline.
check_setup.py
    Fast preflight validation.
dataset_keypoint_generation.py
    Keypoint regeneration for training data.
```