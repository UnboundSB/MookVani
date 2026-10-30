import csv
import re
from pathlib import Path
from collections import defaultdict

# Build complete word inventory from ALL sources
ae = Path(r"F:\dataset\archive_extracted")
ie = Path(r"F:\dataset\isign_extracted")

# 1. Scan archive_extracted
archive_words = {}
for cat_dir in sorted(ae.iterdir()):
    if not cat_dir.is_dir():
        continue
    for wd in sorted(cat_dir.iterdir()):
        if not wd.is_dir():
            continue
        m = re.match(r"^(\d+)\.\s+(.+)$", wd.name.strip())
        if m:
            word = m.group(2).strip().lower()
            idx = int(m.group(1))
        else:
            word = wd.name.strip().lower()
            idx = 0
        videos = [str(f) for f in wd.iterdir() if f.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}]
        archive_words[word] = {
            "videos": videos,
            "source": "archive_extracted",
            "category": cat_dir.name,
            "index": idx
        }

print(f"archive_extracted: {len(archive_words)} words")

# 2. Scan isign_extracted - match filenames to word_ids from CSVs
word_ids = set()
word_id_to_word = {}
for csv_name in ["word-description-dataset_v1.1.csv", "word-presence-dataset_v1.1.csv"]:
    csv_path = ie / csv_name
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            wid = row['word_id'].strip()
            word = row['word'].strip().lower()
            if wid:
                word_ids.add(wid)
                word_id_to_word[wid] = word

videos_dir = ie / "iSign-videos_v1.1"
isign_words = {}
for vf in sorted(videos_dir.iterdir()):
    if vf.suffix.lower() not in {".mp4", ".mov", ".mkv", ".avi"}:
        continue
    stem = vf.stem
    
    # Try direct match
    if stem in word_ids:
        word = word_id_to_word[stem]
        isign_words.setdefault(word, []).append(str(vf))
        continue
    
    # Strip _w and match
    if stem.endswith('_w'):
        base = stem[:-2]
        if base in word_ids:
            word = word_id_to_word[base]
            isign_words.setdefault(word, []).append(str(vf))
            continue
    
    # Strip --N and match
    if '--' in stem:
        base = stem.split('--')[0]
        if base in word_ids:
            word = word_id_to_word[base]
            isign_words.setdefault(word, []).append(str(vf))
            continue

print(f"isign_extracted: {len(isign_words)} words")

# Merge: archive_extracted takes priority for same word
all_words = {}
for word, info in archive_words.items():
    all_words[word] = info

for word, videos in isign_words.items():
    if word not in all_words:
        all_words[word] = {
            "videos": videos,
            "source": "isign_extracted",
            "category": "isign",
            "index": 0
        }
    else:
        # Add isign videos to existing word
        all_words[word]["videos"].extend(videos)

print(f"\nTotal unique words with videos: {len(all_words)}")

# Show words that are in archive but not isign, and vice versa
ae_only = set(archive_words.keys()) - set(isign_words.keys())
isign_only = set(isign_words.keys()) - set(archive_words.keys())
both = set(archive_words.keys()) & set(isign_words.keys())

print(f"\nWords in archive_extracted only: {len(ae_only)}")
print(f"Words in isign_extracted only: {len(isign_only)}")
print(f"Words in BOTH: {len(both)}")

# Show sample of isign-only words
print(f"\nSample isign-only words: {sorted(isign_only)[:30]}")

# Save inventory for use in building CSVs
import json
inventory = {}
for word, info in all_words.items():
    inventory[word] = {
        "videos": info["videos"],
        "source": info["source"],
        "category": info["category"],
        "index": info.get("index", 0)
    }

with open(r"D:\MookVani\backend\segmentor\data_builder\word_inventory.json", 'w') as f:
    json.dump(inventory, f, indent=2)

print("\nSaved word inventory to word_inventory.json")
