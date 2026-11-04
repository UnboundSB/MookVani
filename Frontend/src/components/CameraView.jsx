import { useRef, useEffect, useCallback } from 'react'
import { Activity } from 'lucide-react'

const CONNECTIONS = {
  POSE: [
    [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8],
    [9, 10], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
    [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
    [11, 12], [11, 23], [12, 24], [23, 24], [23, 25], [24, 26], [25, 27], [26, 28],
  ],
  LEFT_HAND: [
    [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
  ],
  RIGHT_HAND: [
    [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
  ],
  FACE: [
    [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
    [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15],
  ],
}

export const LandmarksOverlay = ({ results, width, height }) => {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !results) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, width, height)

    const drawLandmarks = (landmarks, connections, color, radius = 1.5) => {
      if (!landmarks) return
      ctx.strokeStyle = color
      ctx.lineWidth = 1.5
      ctx.fillStyle = color

      connections.forEach(([a, b]) => {
        const p1 = landmarks[a]
        const p2 = landmarks[b]
        if (p1 && p2) {
          ctx.beginPath()
          ctx.moveTo(p1.x * width, p1.y * height)
          ctx.lineTo(p2.x * width, p2.y * height)
          ctx.stroke()
        }
      })

      landmarks.forEach((lm) => {
        if (lm) {
          ctx.beginPath()
          ctx.arc(lm.x * width, lm.y * height, radius, 0, Math.PI * 2)
          ctx.fill()
        }
      })
    }

    const poseColor = '#3b82f6'
    const leftHandColor = '#10b981'
    const rightHandColor = '#f59e0b'
    const faceColor = '#8b5cf6'

    if (results.faceLandmarks) drawLandmarks(results.faceLandmarks, CONNECTIONS.FACE, faceColor, 0.8)
    if (results.poseLandmarks) drawLandmarks(results.poseLandmarks, CONNECTIONS.POSE, poseColor, 1.5)
    if (results.leftHandLandmarks) drawLandmarks(results.leftHandLandmarks, CONNECTIONS.LEFT_HAND, leftHandColor, 1.5)
    if (results.rightHandLandmarks) drawLandmarks(results.rightHandLandmarks, CONNECTIONS.RIGHT_HAND, rightHandColor, 1.5)
  }, [results, width, height])

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none"
      width={width}
      height={height}
    />
  )
}

export const CameraView = ({ results, onFrame }) => {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const animationRef = useRef(null)

  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    const start = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 1280, height: 720, facingMode: 'user' },
        })
        video.srcObject = stream
        await video.play()
      } catch (err) {
        console.error('Camera access error:', err)
      }
    }

    start()

    return () => {
      const stream = video.srcObject
      if (stream) {
        stream.getTracks().forEach((track) => track.stop())
      }
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
    }
  }, [])

  useEffect(() => {
    const video = videoRef.current
    if (!video || !onFrame) return

    const loop = () => {
      if (video.readyState === 4) {
        onFrame(video)
      }
      animationRef.current = requestAnimationFrame(loop)
    }
    animationRef.current = requestAnimationFrame(loop)

    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
    }
  }, [onFrame])

  return (
    <div className="relative w-full aspect-video rounded-3xl overflow-hidden border border-soft-border bg-white shadow-soft">
      <video
        ref={videoRef}
        className="w-full h-full object-cover"
        autoPlay
        muted
        playsInline
      />
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />
      <div className="absolute inset-0 flex items-end justify-between p-6 pointer-events-none">
        <div className="flex items-center gap-2 bg-black/50 px-3 py-2 rounded-full backdrop-blur-sm text-white text-sm font-medium">
          <Activity className="w-3.5 h-3.5 text-berry-400 animate-pulse" />
          <span>Live</span>
        </div>
      </div>
    </div>
  )
}
