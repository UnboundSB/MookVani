import Button from '../components/Button.jsx'
import "./Welcome.css"

export default function Welcome({ onEnter }) {
  return (
    <section className="welcome-screen">
      <div className="welcome-bg" aria-hidden="true">
        <span className="aura" />
        <span className="aura aura-2" />

        <div className="ring-set left">
          <div className="ring-assembly left-a">
            <div className="ring-spin spin-cw">
              <span className="ring ring-outer" />
              <span className="ring-gradient g1 up" />
            </div>
          </div>

          <div className="ring-assembly left-b">
            <div className="ring-spin spin-ccw">
              <span className="ring ring-semi semi-top" />
              <span className="ring-gradient g2 down" />
            </div>
          </div>
        </div>

        <div className="ring-set right">
          <div className="ring-assembly right-a">
            <div className="ring-spin spin-ccw">
              <span className="ring ring-inner" />
              <span className="ring-gradient g3 up" />
            </div>
          </div>

          <div className="ring-assembly right-b">
            <div className="ring-spin spin-cw">
              <span className="ring ring-semi semi-bottom" />
              <span className="ring-gradient g4 down" />
            </div>
          </div>
        </div>

        <div className="floating-shapes">
          <span className="shape-orbit o1">
            <span className="shape circle" />
          </span>

          <span className="shape-orbit o2">
            <span className="shape circle" />
          </span>

          <span className="shape-orbit o3">
            <span className="shape circle" />
          </span>

          <span className="shape-orbit o4">
            <span className="shape square" />
          </span>

          <span className="shape-orbit o5">
            <span className="shape square" />
          </span>

          <span className="shape-orbit o6">
            <span className="shape square" />
          </span>
        </div>

        <span className="top-glow" />
      </div>

      <div className="welcome-content">
        <h1 className="welcome-title">Mookvani</h1>
        <p className="welcome-subtitle">
          The Indian Sign Language Translator
        </p>

        <Button className="welcome-enter" variant="primary" onClick={onEnter}>
          Enter Mookvani
        </Button>
      </div>
    </section>
  )
}