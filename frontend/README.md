# Frontend

React + Vite camera client for the FastAPI backend.

It captures local webcam frames, encodes them as JPEG, posts them to the backend, and displays the stable letter prediction returned by `islreconformer.h5`.

## Run

Install dependencies once:

```powershell
cd D:\islcsltr\Indian-Sign-Language-Detection\frontend
npm install
```

Start the dev server:

```powershell
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

The frontend expects the backend at:

```text
http://127.0.0.1:8000
```

That value is defined in `frontend/.env.development`.

## Backend health

The UI shows `Backend Ready` only after `/health` confirms the ISL engine is loaded.