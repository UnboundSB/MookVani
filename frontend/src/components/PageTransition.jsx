import { useEffect, useState } from 'react'
import "./PageTransition.css"

const ROUTE_ORDER = {
  welcome: 0,
  dashboard: 1,
  live: 2
}

function routeIndex(key) {
  return ROUTE_ORDER[key] ?? Number.MAX_SAFE_INTEGER
}

export default function PageTransition({ routeKey, children }) {
  const [current, setCurrent] = useState({
    key: routeKey,
    element: children,
  })
  const [phase, setPhase] = useState('enter')
  const [direction, setDirection] = useState('forward')

  useEffect(() => {
    if (routeKey === current.key) return

    setDirection(routeIndex(routeKey) >= routeIndex(current.key) ? 'forward' : 'backward')

    setPhase('leaving')

    const timer = setTimeout(() => {
      setCurrent({ key: routeKey, element: children })
      setPhase('enter')
    }, 480)

    return () => clearTimeout(timer)
  }, [routeKey, current.key, children])

  const isWelcomeRoute = current.key === 'welcome'
  const className = [
    'page-transition',
    isWelcomeRoute ? 'welcome-route' : `slide-${direction}`,
    phase === 'leaving' ? 'is-leaving' : ''
  ].filter(Boolean).join(' ')

  return (
    <div key={current.key} className={className}>
      {current.element}
    </div>
  )
}