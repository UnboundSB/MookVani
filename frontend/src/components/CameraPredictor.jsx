import { useCallback, useEffect, useRef, useState } from 'react'
import Button from './Button.jsx'
import "./CameraPredictor.css"

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'
const DEFAULT_SEND_INTERVAL_MS = 500

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
]

export default function CameraPredictor() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const overlayCanvasRef = useRef(null)
  const streamRef = useRef(null)
  const timerRef = useRef(null)
  const mirrorRef = useRef(false)
  const autoStartRef = useRef(false)

  const [running, setRunning] = useState(false)
  const [busy, setBusy] = useState(false)
  const [prediction, setPrediction] = useState('-')
  const [confidence, setConfidence] = useState(0)
  const [handsCount, setHandsCount] = useState(0)
  const [lastUpdate, setLastUpdate] = useState('never')
  const [backendReady, setBackendReady] = useState(false)
  const [status, setStatus] = useState('Idle')
  const [error, setError] = useState('')
  const [sendIntervalMs, setSendIntervalMs] = useState(DEFAULT_SEND_INTERVAL_MS)
  const [letters, setLetters] = useState([])
  const [word, setWord] = useState('')
  const [correctedWord, setCorrectedWord] = useState('')
  const [sentence, setSentence] = useState('')
  const [handsData, setHandsData] = useState([])
  const [mirrorInput, setMirrorInput] = useState(false)
  const [autoSpeak, setAutoSpeak] = useState(true)
  const [speakMode, setSpeakMode] = useState('sentence')
  const [speechSupported, setSpeechSupported] = useState(
    () => typeof window !== 'undefined' && 'speechSynthesis' in window
  )

  const speakText = useCallback((text, onEnd) => {
    if (!speechSupported || !text || !String(text).trim()) return

    const utterance = new SpeechSynthesisUtterance(String(text))
    utterance.rate = 0.95
    utterance.pitch = 1
    utterance.volume = 1

    const finish = () => {
      if (onEnd) onEnd()
    }

    utterance.onend = finish
    utterance.onerror = finish

    window.speechSynthesis.cancel()
    window.speechSynthesis.speak(utterance)
  }, [speechSupported])

  useEffect(() => {
    mirrorRef.current = mirrorInput
  }, [mirrorInput])

  useEffect(() => {
    const canvas = overlayCanvasRef.current
    if (!canvas) return

    const width = 320
    const height = 240
    canvas.width = width
    canvas.height = height

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.clearRect(0, 0, width, height)

    const handGroups = handsData
      .map((hand) => (Array.isArray(hand.landmarks) ? hand.landmarks : []))
      .filter((group) => group.length > 0)

    if (!handGroups.length) return

    ctx.save()

    if (mirrorRef.current) {
      ctx.translate(width, 0)
      ctx.scale(-1, 1)
    }

    ctx.lineWidth = 2
    ctx.strokeStyle = 'rgba(254, 215, 102, 0.85)'
    ctx.fillStyle = 'rgba(0, 159, 183, 0.95)'

    for (const handGroup of handGroups) {
      const points = handGroup.map((lm) => [
        Number(lm[0]) * width,
        Number(lm[1]) * height,
      ])

      for (const [a, b] of HAND_CONNECTIONS) {
        if (points[a] && points[b]) {
          ctx.beginPath()
          ctx.moveTo(points[a][0], points[a][1])
          ctx.lineTo(points[b][0], points[b][1])
          ctx.stroke()
        }
      }

      points.forEach((point) => {
        ctx.beginPath()
        ctx.arc(point[0], point[1], 3.2, 0, Math.PI * 2)
        ctx.fill()
      })
    }

    ctx.restore()
  }, [handsData, mirrorInput])

  useEffect(() => {
    if (speechSupported) {
      // Warm available voices on browsers where voice loading is async.
      window.speechSynthesis.getVoices()
    }
  }, [speechSupported])

  useEffect(() => {
    return () => {
      if (typeof window !== 'undefined' && window.speechSynthesis) {
        window.speechSynthesis.cancel()
      }
    }
  }, [])

  useEffect(() => {
    let alive = true

    async function checkHealth() {
      try {
        const res = await fetch(`${API_BASE}/health`)
        const data = await res.json()
        if (alive) {
          setBackendReady(res.ok && data.status === 'ready')
        }
      } catch {
        if (alive) setBackendReady(false)
      }
    }

    checkHealth()
    const healthTimer = setInterval(checkHealth, 5000)

    return () => {
      alive = false
      clearInterval(healthTimer)
    }
  }, [])

  const stop = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }

    if (streamRef.current) {
      for (const track of streamRef.current.getTracks()) {
        track.stop()
      }
      streamRef.current = null
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null
    }

    setRunning(false)
    setStatus('Stopped')
  }, [])

  const resetPrediction = useCallback(async () => {
    try {
      await fetch(`${API_BASE}/reset`, { method: 'POST' })
      setPrediction('-')
      setConfidence(0)
      setHandsCount(0)
      setLetters([])
      setWord('')
      setCorrectedWord('')
      setSentence('')
      setHandsData([])
    } catch {
      // Ignore reset failures; the UI still stops locally.
    }
  }, [])

  const flushSpoken = useCallback(async (count = 0) => {
    try {
      const query = count > 0 ? `?count=${encodeURIComponent(count)}` : ''
      const res = await fetch(`${API_BASE}/flush_spoken${query}`, {
        method: 'POST'
      })

      if (!res.ok) return

      const data = await res.json()
      if (data.ok && data.session) {
        setSentence(data.session.sentence || '')
        setCorrectedWord(data.session.corrected_word || '')
      }
    } catch {
      // Ignore flush failures; speech queue should still be able to continue.
    }
  }, [])

  const stopAndReset = useCallback(() => {
    stop()
    resetPrediction()
  }, [stop, resetPrediction])

  const sendFrame = useCallback(async () => {
    const video = videoRef.current
    const canvas = canvasRef.current

    if (!video || !canvas || video.readyState < 2) {
      return
    }

    const width = video.videoWidth || 640
    const height = video.videoHeight || 480

    canvas.width = width
    canvas.height = height

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    if (mirrorRef.current) {
      // Match the older mirrored camera pipeline for model input.
      ctx.save()
      ctx.translate(width, 0)
      ctx.scale(-1, 1)
      ctx.drawImage(video, 0, 0, width, height)
      ctx.restore()
    } else {
      ctx.drawImage(video, 0, 0, width, height)
    }

    const imageBase64 = canvas.toDataURL('image/jpeg', 0.7)

    try {
      const res = await fetch(`${API_BASE}/predict/base64`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_base64: imageBase64 })
      })

      if (!res.ok) {
        throw new Error(`Backend returned ${res.status}`)
      }

      const data = await res.json()

      setBackendReady(true)
      setError('')
      setHandsCount(data.hands_count || 0)
      setPrediction(data.prediction || '-')
      setConfidence(data.confidence || 0)
      setLastUpdate(new Date().toLocaleTimeString())

      const session = data.session || {}
      setLetters(session.letters || [])
      setWord(session.word || '')
      setCorrectedWord(session.corrected_word || '')
      setSentence(session.sentence || '')

      // Clear stale hand data as soon as the backend reports no hands.
      const incomingHands = data.hands_count ? data.hands || [] : []
      setHandsData(incomingHands)

      const finalized = session.finalized || {}
      if (autoSpeak && finalized.finalized) {
        const text =
          speakMode === 'word'
            ? finalized.corrected_word || correctedWord || word
            : finalized.sentence || sentence

        const flushCount =
          Number(finalized.parts_count || 0) ||
          String(text || '')
            .trim()
            .split(/\s+/)
            .filter(Boolean).length

        // TTS is intentionally async; inference keeps running while it speaks.
        speakText(text, () => flushSpoken(flushCount))
      }
    } catch (err) {
      setError(`Frame send failed: ${err.message}`)
      setBackendReady(false)
      stop()
    }
  }, [autoSpeak, correctedWord, flushSpoken, sentence, speakMode, speakText, stop, word])

  const start = useCallback(async () => {
    if (!backendReady) {
      setError('Backend is not ready. Start the FastAPI backend first.')
      return
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      setError('This browser does not expose camera access.')
      return
    }

    try {
      setError('')
      setStatus('Requesting camera...')

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false
      })

      streamRef.current = stream

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }

      setRunning(true)
      setStatus('Running')

      if (timerRef.current) clearInterval(timerRef.current)

      timerRef.current = setInterval(() => {
        setBusy(true)
        sendFrame().finally(() => setBusy(false))
      }, sendIntervalMs)
    } catch (err) {
      setError(`Camera start failed: ${err.message}`)
      setStatus('Error')
      stop()
    }
  }, [backendReady, sendFrame, sendIntervalMs, stop])

  useEffect(() => {
    if (!backendReady || autoStartRef.current) return

    autoStartRef.current = true
    start()
  }, [backendReady, start])

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)

      if (streamRef.current) {
        for (const track of streamRef.current.getTracks()) {
          track.stop()
        }
      }
    }
  }, [])

  return (
    <section className="panel">
      <div className="panel-header">
        <h2>Live Camera Inference</h2>
        <span className={backendReady ? 'badge ok' : 'badge bad'}>
          {backendReady ? 'Backend Ready' : 'Backend Offline'}
        </span>
      </div>

      <div className="camera-layout">
        <div className={running ? 'video-wrap live' : 'video-wrap'}>
          <video
            ref={videoRef}
            playsInline
            muted
            className={mirrorInput ? 'camera mirror' : 'camera'}
          />

          <canvas
            ref={overlayCanvasRef}
            className="overlay-canvas"
            aria-hidden="true"
          />

          {running && (
            <div className="camera-overlay-tags">
              <span className="tag">{mirrorInput ? 'Mirror: On' : 'Mirror: Off'}</span>
              <span className="tag">
                {speakMode === 'word' ? 'Word Mode' : 'Sentence Mode'}
              </span>
              {handsCount === 0 && <span className="tag warn">No hands detected</span>}
            </div>
          )}

          {!running && (
            <div className="camera-placeholder">
              <span className="pulse-dot" />
              <p>{backendReady ? 'Waiting for camera permission' : 'Waiting for backend connection'}</p>
              <small>
                {backendReady
                  ? 'Your browser will ask once for webcam access.'
                  : 'Start the FastAPI backend first; camera controls enable automatically.'}
              </small>

              <Button
                className="camera-enable"
                variant="primary"
                onClick={start}
                disabled={!backendReady || status === 'Requesting camera...'}
              >
                {status === 'Requesting camera...' ? 'Requesting...' : 'Enable Camera'}
              </Button>
            </div>
          )}
        </div>

        
      </div>

      <canvas ref={canvasRef} className="capture-canvas" />

      <div className="hud">
        <div className="prediction" aria-live="polite">{prediction}</div>
        <div className="confidence">{(confidence * 100).toFixed(1)}%</div>
        <div className="meta">Hands: {handsCount}</div>
        <div className="meta">Status: {status}</div>
        <div className="meta">Last update: {lastUpdate}</div>
      </div>

      <div className="session-strip">
        <div className="session-block">
          <span className="session-label">Letters</span>
          <strong>{word || '—'}</strong>
        </div>

        <div className="session-block">
          <span className="session-label">Corrected</span>
          <strong>{correctedWord || '—'}</strong>
        </div>

        <div className="session-block wide">
          <span className="session-label">Sentence</span>
          <strong aria-live="polite">{sentence || '—'}</strong>
        </div>

        <div className="session-block wide">
          <span className="session-label">Top Predictions</span>
          {handsData.length ? (
            <div className="debug-list">
              {handsData.map((hand, handIndex) => (
                <div className="debug-hand" key={`debug-hand-${handIndex}`}>
                  <strong>Hand {handIndex + 1}</strong>
                  {(hand.top_predictions || []).map((item) => (
                    <span key={`${handIndex}-${item.class_index}-${item.label}`}>
                      {item.label} [{item.class_index}]{' '}
                      {(item.probability * 100).toFixed(1)}%
                    </span>
                  ))}
                </div>
              ))}
            </div>
          ) : (
            <strong>—</strong>
          )}
        </div>
      </div>

      <div className="controls">
        <Button
          variant="primary"
          onClick={start}
          disabled={!backendReady || running || status === 'Requesting camera...'}
        >
          Start Camera
        </Button>
        <Button variant="secondary" onClick={stopAndReset} disabled={!running}>
          Stop
        </Button>

        <label>
          Send interval:
          <input
            type="range"
            min="200"
            max="1500"
            step="50"
            value={sendIntervalMs}
            onChange={(e) => setSendIntervalMs(Number(e.target.value))}
          />
          {sendIntervalMs}ms
        </label>
      </div>

      <div className="controls">
        <Button
          variant={mirrorInput ? 'primary' : 'secondary'}
          onClick={() => {
            const next = !mirrorInput
            setMirrorInput(next)
            mirrorRef.current = next
            resetPrediction()
          }}
        >
          {mirrorInput ? 'Mirror: On' : 'Mirror: Off'}
        </Button>

        <Button
          variant={autoSpeak ? 'primary' : 'secondary'}
          onClick={() => setAutoSpeak((value) => !value)}
          disabled={!speechSupported}
        >
          {autoSpeak ? 'Auto Speak: On' : 'Auto Speak: Off'}
        </Button>

        <Button
          variant="secondary"
          onClick={() => {
            const text =
              speakMode === 'word'
                ? correctedWord || word
                : sentence || correctedWord || word

            const flushCount = String(text || '')
              .trim()
              .split(/\s+/)
              .filter(Boolean).length

            speakText(text, () => flushSpoken(flushCount))
          }}
          disabled={!sentence && !correctedWord && !word}
        >
          Speak Now
        </Button>

        <label>
          Speak mode:
          <select
            value={speakMode}
            onChange={(e) => setSpeakMode(e.target.value)}
          >
            <option value="sentence">Sentence</option>
            <option value="word">Word</option>
          </select>
        </label>
      </div>

      {!speechSupported && (
        <p className="error">
          This browser does not support speech synthesis. Use Chrome/Edge for spoken output.
        </p>
      )}

      {error && <p className="error">{error}</p>}
    </section>
  )
}