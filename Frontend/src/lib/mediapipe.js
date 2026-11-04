import {
  HolisticLandmarker,
  FilesetResolver,
} from '@mediapipe/tasks-vision'

let holisticLandmarker = null
let isInitialized = false
let initPromise = null

const MODEL_BASE_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm/'

export const initMediaPipe = async () => {
  if (isInitialized && holisticLandmarker) return holisticLandmarker
  if (initPromise) return initPromise

  initPromise = (async () => {
    const vision = await FilesetResolver.forVisionTasks(
      `${MODEL_BASE_URL}holistic_landmarker/`
    )
    holisticLandmarker = await HolisticLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: `${MODEL_BASE_URL}holistic_landmarker.task`,
        delegate: 'CPU',
      },
      runningMode: 'VIDEO',
      numFaces: 1,
      minFaceDetectionConfidence: 0.5,
      minFacePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      outputSegmentationMasks: false,
    })
    isInitialized = true
    return holisticLandmarker
  })()

  return initPromise
}

export const detectVideo = async (landmarker, video, videoTime) => {
  if (!landmarker) return null
  return landmarker.detectForVideo(video, videoTime)
}

export { HolisticLandmarker }
