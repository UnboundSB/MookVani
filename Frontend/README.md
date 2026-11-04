# MookVani Frontend

A professional, easy-on-the-eyes React frontend for the MookVani ISL Recognition
and Translation engine. Runs MediaPipe entirely in the browser — no frames leave
your device.

## Tech stack

- React 18 + Vite
- Tailwind CSS
- MediaPipe Tasks Vision (`@mediapipe/tasks-vision`)
- Framer Motion, Lucide React

## Getting started

```bash
cd Frontend
npm install
npm run dev        # http://localhost:5173
npm run build      # production build
npm run preview    # preview production build
```

## Features

- Live webcam capture with MediaPipe holistic landmark detection
- Pose, face, and hand landmark overlays drawn on canvas
- Real-time ISL translation display with text-to-speech
- Local-only processing (privacy-first)
- Fullscreen mode and session stats
- Soft, warm theme designed for extended viewing

## Project structure

```
Frontend/
├── src/
│   ├── main.jsx
│   ├── App.jsx
│   ├── index.css
│   ├── components/
│   │   ├── Navbar.jsx
│   │   ├── CameraView.jsx
│   │   └── TranslationPanel.jsx
│   ├── hooks/
│   │   └── useMediaPipe.js
│   └── lib/
│       └── mediapipe.js
├── index.html
├── vite.config.js
├── tailwind.config.js
└── package.json
```
