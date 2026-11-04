import { useState, useEffect } from 'react'
import { Upload, CheckCircle2, FileText, Play, Video as VideoIcon, AlertCircle } from 'lucide-react'
import { loadTrainedWords, getWordsForVideo } from '../lib/csvParser'

export const VideoUpload = ({ onWordsFound }) => {
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [trainedWords, setTrainedWords] = useState([])
  const [sentence, setSentence] = useState('')
  const [category, setCategory] = useState('')
  const [error, setError] = useState(null)
  const [rows, setRows] = useState([])

  useEffect(() => {
    loadTrainedWords().then((data) => setRows(data))
  }, [])

  const handleFileChange = (e) => {
    const selected = e.target.files[0]
    if (!selected) return
    if (!selected.type.startsWith('video/')) {
      setError('Please upload a video file.')
      return
    }
    setFile(selected)
    setError(null)
    setTrainedWords([])
    setSentence('')
    setCategory('')
  }

  const handleUpload = async () => {
    if (!file) return
    setUploading(true)
    setError(null)

    // Simulate upload processing
    await new Promise((r) => setTimeout(r, 800))

    const result = getWordsForVideo(rows, file.name)
    if (result) {
      setTrainedWords(result.words)
      setSentence(result.sentence)
      setCategory(result.category)
      if (onWordsFound) onWordsFound(result.words)
    } else {
      setError(`No trained words found for "${file.name}".`)
    }

    setUploading(false)
  }

  const reset = () => {
    setFile(null)
    setTrainedWords([])
    setSentence('')
    setCategory('')
    setError(null)
  }

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-xl bg-berry-50 flex items-center justify-center text-berry-600">
            <Upload className="w-4 h-4" />
          </div>
          <div>
            <div className="font-semibold text-soft-ink">Upload Training Video</div>
            <div className="text-xs text-soft-ink/50">Match against trained-words.csv</div>
          </div>
        </div>
        <FileText className="w-4 h-4 text-soft-ink/40" />
      </div>

      <div className="border-2 border-dashed border-soft-border rounded-2xl bg-neutral-50/50 p-8 text-center hover:border-berry-200 hover:bg-neutral-100 transition-colors relative overflow-hidden group">
        <input
          type="file"
          accept="video/*"
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
          onChange={handleFileChange}
        />
        <div className="relative z-0 flex flex-col items-center gap-3 pointer-events-none">
          {!file ? (
            <>
              <Upload className="w-10 h-10 text-berry-400 group-hover:text-berry-600 transition-colors" />
              <div className="text-sm font-medium text-soft-ink group-hover:text-soft-ink transition-colors">
                Click to upload or drag a video
              </div>
              <div className="text-xs text-soft-ink/40">
                Supported: MP4, MOV, AVI, WEBM
              </div>
            </>
          ) : (
            <>
              <div className="flex items-center gap-2 text-soft-ink font-medium text-sm">
                <VideoIcon className="w-4 h-4 text-berry-500" />
                <span>{file.name}</span>
              </div>
              <button onClick={handleUpload} disabled={uploading} className="btn-primary">
                {uploading ? (
                  <>
                    <span className="relative animate-pulse">Processing...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" /> Match Trained Words
                  </>
                )}
              </button>
              <button onClick={reset} className="text-xs text-soft-ink/50 hover:text-berry-600 underline underline-offset-4">
                Change file
              </button>
            </>
          )}
        </div>
      </div>

      {error && (
        <div className="mt-3 p-3 rounded-xl bg-[#fef2f2] border border-[#fee2e2] text-[#991b1b] text-sm flex items-center gap-2">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          {error}
        </div>
      )}

      {trainedWords.length > 0 && (
        <div className="mt-4 pt-4 border-t border-soft-border">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-[#166534]" />
              <div className="font-semibold text-soft-ink">Trained Words Found</div>
            </div>
            <span className="text-xs bg-neutral-100 text-neutral-600 px-2 py-1 rounded border border-neutral-200">
              {category}
            </span>
          </div>
          <div className="flex flex-wrap gap-2">
            {trainedWords.map((word, i) => (
              <span
                key={i}
                className="px-3 py-1.5 rounded-full bg-berry-50 border border-berry-100 text-berry-700 text-xs font-medium shadow-sm"
              >
                {word}
              </span>
            ))}
          </div>
          {sentence && (
            <div className="mt-3 flex items-center gap-2 text-xs text-soft-ink/60">
              <FileText className="w-3 h-3" />
              <span>From sentence: "{sentence}"</span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
