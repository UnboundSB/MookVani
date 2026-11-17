import PageHeader from '../components/PageHeader.jsx'
import CameraPredictor from '../components/CameraPredictor.jsx'
import "./LiveRecognition.css"

export default function LiveRecognition() {
  return (
    <section className="page">
      <PageHeader
        title="Live Recognition"
        description="Send camera frames to the Mookvani model and watch a stable sign prediction settle every second."
      />

      <CameraPredictor />
    </section>
  )
}