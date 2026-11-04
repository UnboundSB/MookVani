export const parseCSV = (text) => {
  const lines = text.split(/\r?\n/).filter((line) => line.trim().length > 0)
  if (lines.length === 0) return []

  const headers = lines[0].split(',').map((h) => h.trim())
  const rows = []

  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(',')
    const row = {}
    headers.forEach((header, index) => {
      row[header] = values[index] ? values[index].trim() : ''
    })
    rows.push(row)
  }

  return rows
}

export const loadTrainedWords = async () => {
  try {
    const response = await fetch('/trained_words.csv')
    if (!response.ok) throw new Error('Failed to load CSV')
    const text = await response.text()
    return parseCSV(text)
  } catch (error) {
    console.error('CSV load error:', error)
    return []
  }
}

export const getWordsForVideo = (rows, filename) => {
  const match = rows.find((row) => row.video_filename === filename)
  if (!match) return null
  const words = match.words.split(',').map((w) => w.trim()).filter(Boolean)
  return {
    filename: match.video_filename,
    words,
    sentence: match.sentence,
    category: match.category,
  }
}
