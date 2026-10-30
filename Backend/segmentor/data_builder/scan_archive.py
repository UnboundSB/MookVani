import os
from pathlib import Path
import re

ae = Path(r"F:\dataset\archive_extracted")
word_entries = []

for cat_dir in sorted(ae.iterdir()):
    if not cat_dir.is_dir():
        continue
    for wd in sorted(cat_dir.iterdir()):
        if not wd.is_dir():
            continue
        # Folder name like "1. loud" or "83. big large"
        m = re.match(r"^(\d+)\.\s+(.+)$", wd.name.strip())
        if m:
            word = m.group(2).strip().lower()
            idx = int(m.group(1))
        else:
            word = wd.name.strip().lower()
            idx = 0
        # Count videos
        videos = [f for f in wd.iterdir() if f.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}]
        word_entries.append({
            "word": word,
            "folder": wd.name,
            "category": cat_dir.name,
            "index": idx,
            "video_count": len(videos),
            "source": "archive_extracted"
        })

print(f"Total word entries from archive_extracted: {len(word_entries)}")
print(f"Unique words: {len(set(e['word'] for e in word_entries))}")

# Show some examples
for e in word_entries[:10]:
    print(f"  {e['index']:3d}. {e['word']:20s} [{e['category']}] {e['video_count']} videos")
