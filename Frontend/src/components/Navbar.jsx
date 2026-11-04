import { Framer, Home, Video, Mic, Send, CheckCircle2, Activity } from 'lucide-react'

export const Navbar = () => (
  <nav className="fixed top-0 left-0 right-0 z-50 border-b border-soft-border bg-white/80 backdrop-blur-md">
    <div className="flex items-center justify-between px-6 py-3.5 max-w-7xl mx-auto">
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-xl bg-berry-600 flex items-center justify-center shadow-sm shadow-berry-500/30">
          <Framer className="w-5 h-5 text-white" />
        </div>
        <div>
          <div className="text-sm font-bold text-soft-ink tracking-tight">MookVani</div>
          <div className="text-xs text-soft-ink/60 tracking-wider">ISL Translation</div>
        </div>
      </div>

      <div className="hidden md:flex items-center gap-6 text-sm text-soft-ink/70 font-medium">
        <button className="flex items-center gap-2 hover:text-berry-600 transition-colors">
          <Home className="w-4 h-4" /> Dashboard
        </button>
        <button className="flex items-center gap-2 hover:text-berry-600 transition-colors">
          <Video className="w-4 h-4" /> Live Capture
        </button>
        <button className="flex items-center gap-2 hover:text-berry-600 transition-colors">
          <Mic className="w-4 h-4" /> Voice
        </button>
      </div>

      <div className="flex items-center gap-3">
        <button className="btn-ghost text-xs">
          <Send className="w-3 h-3" /> Export
        </button>
        <button className="btn-primary text-xs px-4">
          Sign In
        </button>
      </div>
    </div>
  </nav>
)

export const StatusBadge = ({ label, variant = 'default' }) => {
  const variants = {
    default: 'bg-white text-soft-ink border border-soft-border',
    active: 'bg-berry-50 text-berry-700 border border-berry-100',
    success: 'bg-[#dcfce7] text-[#166534] border border-[#bbf7d0]',
    idle: 'bg-[#fef2f2] text-[#991b1b] border border-[#fee2e2]',
  }

  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${variants[variant]}`}>
      <span className="w-1.5 h-1.5 rounded-full bg-current opacity-70 animate-pulse" />
      {label}
    </span>
  )
}
