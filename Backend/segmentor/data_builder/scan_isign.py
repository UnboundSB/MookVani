import csv
from pathlib import Path
from collections import Counter

ie = Path(r"F:\dataset\isign_extracted")
csv_path = ie / "word-presence-dataset_v1.1.csv"

# Read the CSV to map word_id -> word
word_id_to_word = {}
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        wid = row['word_id'].strip()
        word = row['word'].strip().lower()
        if wid and word:
            word_id_to_word[wid] = word

print(f"Unique word IDs in CSV: {len(word_id_to_word)}")
print(f"Sample: {list(word_id_to_word.items())[:5]}")

# Now scan the video files
videos_dir = ie / "iSign-videos_v1.1"
word_entries = []
for vf in sorted(videos_dir.iterdir()):
    if vf.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}:
        stem = vf.stem
        # Extract word_id from filename like "scJSFdl6vqk_w" or "_QxIMNNlDiw--11"
        if "_w" in stem:
            wid = stem.split("_w")[0]
        elif "--" in stem:
            wid = stem.split("--")[0]
        else:
            wid = stem
        
        word = word_id_to_word.get(wid, stem.lower())
        word_entries.append({
            "word": word,
            "file": vf.name,
            "source": "isign_extracted"
        })

print(f"Total isign_extracted video files: {len(word_entries)}")
unique_words = set(e['word'] for e in word_entries)
print(f"Unique words found in isign_extracted videos: {len(unique_words)}")

# Show words that are in archive_extracted but NOT in isign, and vice versa
ae_words = set()
from pathlib import Path
import re
ae_root = Path(r"F:\dataset\archive_extracted")
for cat_dir in sorted(ae_root.iterdir()):
    if not cat_dir.is_dir():
        continue
    for wd in sorted(cat_dir.iterdir()):
        if not wd.is_dir():
            continue
        m = re.match(r"^(\d+)\.\s+(.+)$", wd.name.strip())
        if m:
            ae_words.add(m.group(2).strip().lower())

isign_words = unique_words

print(f"\nWords in archive_extracted only: {len(ae_words - isign_words)}")
print(f"Examples: {sorted(ae_words - isign_words)[:20]}")
print(f"\nWords in isign_extracted only: {len(isign_words - ae_words)}")
print(f"Examples: {sorted(isign_words - ae_words)[:20]}")
print(f"\nWords in BOTH: {len(ae_words & isign_words)}")
