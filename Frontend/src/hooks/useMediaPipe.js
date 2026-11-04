import { useEffect, useRef, useState, useCallback } from 'react'
import { initMediaPipe, detectVideo } from '../lib/mediapipe'

export const useMediaPipe = (onResult, enabled = true) => {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const animationRef = useRef(null)
  const landmarkerRef = useRef(null)
  const [isReady, setIsReady] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState(null)

  const processVideo = useCallback(
    (videoTime) => {
      if (!videoRef.current || !landmarkerRef.current || !enabled) return
      try {
        const results = detectVideo(landmarkerRef.current, videoRef.current, videoTime)
        if (results && onResult) {
          onResult(results)
        }
        animationRef.current = requestAnimationFrame(() => {
          processVideo(videoRef.current ? Date.now() : 0)
        })
      } catch (err) {
        console.error('Detection error:', err)
      }
    },
    [enabled, onResult]
  )

  useEffect(() => {
    let mounted = true

    const setup = async () => {
      try {
        setIsLoading(true)
        const landmarker = await initMediaPipe()
        if (!mounted) return
        landmarkerRef.current = landmarker
        setIsReady(true)
        setIsLoading(false)
      } catch (err) {
        if (!mounted) return
        setError(err.message || 'Failed to initialize MediaPipe')
        setIsLoading(false)
      }
    }

    setup()

    return () => {
      mounted = false
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
    }
  }, [enabled])

  useEffect(() => {
    if (!isReady || !enabled || !videoRef.current) return
    animationRef.current = requestAnimationFrame(() => {
      processVideo(videoRef.current ? Date.now() : 0)
    })
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
    }
  }, [isReady, enabled, processVideo])

  return {
    videoRef,
    canvasRef,
    isReady,
    isLoading,
    error,
  }
}
