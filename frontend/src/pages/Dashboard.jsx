import { useEffect, useState } from 'react'
import Button from '../components/Button.jsx'
import "./Dashboard.css"

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'

export default function Dashboard() {
  const [backend, setBackend] = useState('checking')

  useEffect(() => {
    let alive = true

    async function checkHealth() {
      try {
        const res = await fetch(`${API_BASE}/health`)
        const data = await res.json()
        if (!alive) return
        setBackend(res.ok && data.status === 'ready' ? 'ready' : 'offline')
      } catch {
        if (alive) setBackend('offline')
      }
    }

    checkHealth()
    const timer = setInterval(checkHealth, 5000)

    return () => {
      alive = false
      clearInterval(timer)
    }
  }, [])

  const statusLabel =
    backend === 'ready' ? 'Backend Ready' : backend === 'offline' ? 'Backend Offline' : 'Checking Backend'

  return (
    <section className="dashboard">
      <div className="hero-card">
        <p className="eyebrow">Welcome to</p>
        <h1>
          Mookvani <span>·</span> <em>Lightweight</em> Indian Sign Language Translator
        </h1>
        <p className="hero-copy">
          Mookvani turns hand landmarks into readable signs using the trained local model.
          Open the camera, hold a sign steady, and Mookvani settles on one stable letter every second.
        </p>

        <div className="hero-actions">
          <Button as="a" href="#/live" variant="primary">
            Start Live Recognition
          </Button>
          <Button as="a" href={`${API_BASE}/docs`} target="_blank" rel="noreferrer" variant="ghost">
            Backend API Docs
          </Button>
        </div>

        <div className="hero-status">
          <span className={backend === 'ready' ? 'dot ready' : 'dot offline'} />
          {statusLabel}
        </div>
      </div>

      <section className="model-stack">
        <div className="stack-head">
          <p className="stack-kicker">Pipeline</p>
          <h2>Skeleton → Model → Labels → Correction Model → Speech</h2>
        </div>

        <div className="stack-grid">
          <article className="stack-card skeleton">
            <p className="stack-label">Skeleton</p>
            <h3>MediaPipe HandLandmarker</h3>
            <div className="stack-flow">
              <span>Hand</span>
              <span className="arrow">→</span>
              <span>21 Joints</span>
              <span className="arrow">→</span>
              <span>42 Features</span>
            </div>
            <p className="stack-note">
              MediaPipe detects up to 21 hand joints per hand. Each hand becomes 42 x/y values; two hands combine into 84.
            </p>
          </article>

          <article className="stack-card model">
            <p className="stack-label">Model</p>
            <h3>Keras sequence classifier</h3>
            <div className="chip-row">
              <span className="chip">RNN Decoder Stack</span>
              <span className="chip">Transformer</span>
              <span className="chip">TCNN Head</span>
            </div>
            <p className="stack-note">
              The trained classifier converts one-hand 42-feature or two-hand 84-feature vectors into class scores.
            </p>
          </article>

          <article className="stack-card labels">
            <p className="stack-label">Labels</p>
            <h3 className="good-label">Digits + Letters</h3>
            <div className="chip-row">
              <span className="chip">35 Classes</span>
              <span className="chip">1–9</span>
              <span className="chip">A–Z</span>
            </div>
            <p className="stack-note">
              Prediction labels cover digits 1–9 and uppercase letters A–Z.
            </p>
          </article>

          <article className="stack-card correction">
            <p className="stack-label">Correction Model</p>
            <h3>SymSpell</h3>
            <div className="chip-row">
              <span className="chip">Word Segmentation</span>
              <span className="chip">Spell Correction</span>
              <span className="chip">Digit-safe</span>
            </div>
            <p className="stack-note">
              After hands disappear, cached letters are segmented and corrected without removing digits.
            </p>
          </article>

          <article className="stack-card speech">
            <p className="stack-label">Speech</p>
            <h3>Browser SpeechSynthesis</h3>
            <div className="chip-row">
              <span className="chip">React TTS</span>
              <span className="chip">Auto Speak</span>
              <span className="chip">Sentence / Word</span>
            </div>
            <p className="stack-note">
              The corrected sentence is returned to React and spoken directly in the browser.
            </p>
          </article>
        </div>
      </section>

      <div className="workflow-card">
        <h2>How Mookvani works</h2>
        <ol className="workflow-list">
          <li>
            <span>1</span>
            <div>
              <strong>Capture</strong>
              <p>The browser sends one JPEG webcam frame to FastAPI.</p>
            </div>
          </li>
          <li>
            <span>2</span>
            <div>
              <strong>Landmark</strong>
              <p>MediaPipe finds up to 21 hand joints per hand and converts each hand into 42 features.</p>
            </div>
          </li>
          <li>
            <span>3</span>
            <div>
              <strong>Predict</strong>
              <p>The Keras model scores the combined feature vector across 35 sign classes.</p>
            </div>
          </li>
          <li>
            <span>4</span>
            <div>
              <strong>Stabilize</strong>
              <p>One second of votes removes flickering labels during live translation.</p>
            </div>
          </li>
        </ol>
      </div>
    </section>
  )
}