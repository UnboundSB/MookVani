# Backend

FastAPI backend that receives webcam frames from the React frontend and returns predictions from `islreconformer.h5`.

## Run

From the project root:

```powershell
cd D:\islcsltr\Indian-Sign-Language-Detection
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
.\.venv\Scripts\python.exe -m pip install -r .\backend\requirements.txt
.\.venv\Scripts\python.exe -m uvicorn backend.api:app --host 127.0.0.1 --port 8000 --reload
```

Server root:

```text
http://127.0.0.1:8000
```

## Sign-to-sentence pipeline

1. React sends JPEG frames to `/predict/base64`.
2. The model returns a stable letter prediction every second.
3. The backend caches letters while any hand remains visible.
4. When hands disappear for `ISL_FINALIZE_IDLE_SECONDS` (default: 1.0 second), the cached letters are finalized.
5. SymSpell segments/corrects the letter run into a word.
6. Corrected words are added to the sentence.
7. The returned `session` payload is sent back to React.
8. React browser SpeechSynthesis speaks the corrected sentence/word.

Example: cached `H I A M G O O D` → corrected sentence `HI AM GOOD` → spoken sentence.

Digit runs are preserved exactly and are never sent to SymSpell.

## Endpoints

- `GET /health`
- `GET /classes`
- `POST /predict/base64`
- `POST /predict`
- `POST /finalize`
- `POST /flush_spoken`
- `POST /reset`
- `GET /sentence`
- `GET /speech/state`

Example request body:

```json
{
  "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

Example response:

```json
{
  "ok": true,
  "prediction": "A",
  "confidence": 0.91,
  "hands_count": 1,
  "prediction_updated": true,
  "hands": [],
  "session": {
    "letters": ["A"],
    "word": "A",
    "sentence": "HI AM GOOD",
    "corrected_word": "HI",
    "hands_count": 1,
    "finalized": { "finalized": false }
  }
}
```

## Environment settings

- `ISL_FINALIZE_IDLE_SECONDS` — no-hand delay before finalizing a word
- `ISL_SPEECH_ENABLED` — `1` enables speech, `0` disables it
- `ISL_SPEECH_RATE` — speech words per minute
- `ISL_SPEECH_VOLUME` — speech volume from `0.0` to `1.0`
- `ISL_SPEECH_MODE` — `sentence` or `word`
- `ISL_ALLOWED_ORIGINS` — comma-separated frontend origins