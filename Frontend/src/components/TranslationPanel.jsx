import { ArrowLeft, Clock, MessageCircle, ShieldCheck } from 'lucide-react'

export const TranslationPanel = ({ text, confidence, isSpeaking, onSpeak, onClear }) => {
  return (
    <div className="flex flex-col gap-5">
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-berry-50 flex items-center justify-center text-berry-600 shadow-sm">
            <MessageCircle className="w-5 h-5" />
          </div>
          <div>
            <div className="text-sm font-semibold text-soft-ink">Translation</div>
            <div className="text-xs text-soft-ink/50">Indian Sign Language → Text</div>
          </div>
        </div>
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white border border-soft-border text-xs font-medium text-soft-ink shadow-sm">
          <ShieldCheck className="w-3 h-3 fill-current" />
          <span>Local processing</span>
        </div>
      </div>

      <div className="relative flex-1 min-h-[140px] rounded-2xl border border-soft-border bg-white/70 backdrop-blur-sm p-6 shadow-sm transition-all duration-300 hover:border-berry-200/50">
        {text ? (
          <div className="flex flex-col gap-3">
            <div className="text-2xl font-display font-semibold text-soft-ink leading-relaxed animate-[fade-in_0.4s_ease-out]">
              {text}
            </div>
            <div className="flex items-center justify-between pt-2 border-t border-soft-border/50">
              <div className="flex items-center gap-1.5 text-xs text-soft-ink/50">
                <Clock className="w-3 h-3" />
                <span>Just now</span>
              </div>
              <div className="flex items-center gap-1.5 text-xs font-medium text-berry-600">
                {Math.round((confidence || 0) * 100)}% confidence
              </div>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-center gap-3">
            <div className="w-12 h-12 rounded-full bg-berry-50/50 flex items-center justify-center text-berry-400 mx-auto">
              <MessageCircle className="w-5 h-5 opacity-50" />
            </div>
            <div className="text-sm text-soft-ink/50 font-medium">No signs detected yet</div>
            <div className="text-xs text-soft-ink/40">Start signing in front of the camera</div>
          </div>
        )}
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={onSpeak}
          disabled={!text || isSpeaking}
          className="btn-primary flex-1"
        >
          {isSpeaking ? (
            <>
              <span className="relative animate-pulse">Speaking...</span>
            </>
          ) : (
            <span>Speak translation</span>
          )}
        </button>
        <button onClick={onClear} className="btn-ghost">
          <ArrowLeft className="w-4 h-4" /> Clear
        </button>
      </div>
    </div>
  )
}

export const StatsCard = ({ label, value, icon: Icon, color = 'berry' }) => {
  const colorMap = {
    berry: 'bg-berry-50 text-berry-600',
    cyan: 'bg-cyan-50 text-cyan-600',
    neutral: 'bg-neutral-50 text-neutral-600',
  }

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-3">
        <div className={`w-8 h-8 rounded-xl flex items-center justify-center ${colorMap[color] || colorMap.accent}`}>
          <Icon className="w-4 h-4" />
        </div>
      </div>
      <div className="text-2xl font-display font-bold text-soft-ink">{value}</div>
      <div className="text-sm text-soft-ink/60 font-medium">{label}</div>
    </div>
  )
}
