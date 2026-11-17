import { useEffect, useState } from 'react'
import Dashboard from '../pages/Dashboard.jsx'
import LiveRecognition from '../pages/LiveRecognition.jsx'
import Welcome from '../pages/Welcome.jsx'
import PageTransition from './PageTransition.jsx'
import "./AppShell.css"

export const ROUTES = {
  welcome: {
    path: '#/welcome',
    label: 'Welcome',
    showInNav: false
  },
  dashboard: {
    path: '#/dashboard',
    label: 'Dashboard',
    showInNav: true
  },
  live: {
    path: '#/live',
    label: 'Live Recognition',
    showInNav: true
  }
}

const WELCOME_SEEN_KEY = 'mookvani.welcome.seen'

function welcomeWasSeen() {
  try {
    return window.sessionStorage.getItem(WELCOME_SEEN_KEY) === '1'
  } catch {
    return false
  }
}

function markWelcomeSeen() {
  try {
    window.sessionStorage.setItem(WELCOME_SEEN_KEY, '1')
  } catch {
    // Session storage can fail in private mode; navigation still works.
  }
}

function getInitialRoute() {
  const key = window.location.hash.replace('#/', '')

  if (ROUTES[key]) {
    return key
  }

  return welcomeWasSeen() ? 'dashboard' : 'welcome'
}

function getRouteFromHash() {
  const key = window.location.hash.replace('#/', '')
  return ROUTES[key] ? key : 'dashboard'
}

export default function AppShell() {
  const [route, setRoute] = useState(getInitialRoute)

  useEffect(() => {
    const onHashChange = () => setRoute(getRouteFromHash())
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  const enterApp = () => {
    markWelcomeSeen()
    window.location.hash = '#/dashboard'
  }

  const renderRoute = () => {
    switch (route) {
      case 'welcome':
        return <Welcome onEnter={enterApp} />
      case 'live':
        return <LiveRecognition />
      default:
        return <Dashboard />
    }
  }

  return (
    <main className="app-shell">
      {route !== 'welcome' && (
        <header className="topbar">
          <div className="brand">
            <span className="brand-mark">M</span>
            <div>
              <strong>Mookvani</strong>
              <small>Lightweight ISL Translator</small>
            </div>
          </div>

          <nav className="nav" aria-label="Primary">
            {Object.entries(ROUTES)
              .filter(([, routeInfo]) => routeInfo.showInNav)
              .map(([key, routeInfo]) => (
                <a
                  key={key}
                  href={routeInfo.path}
                  className={route === key ? 'nav-link active' : 'nav-link'}
                >
                  {routeInfo.label}
                </a>
              ))}
          </nav>
        </header>
      )}

      <PageTransition routeKey={route}>
        {renderRoute()}
      </PageTransition>
    </main>
  )
}