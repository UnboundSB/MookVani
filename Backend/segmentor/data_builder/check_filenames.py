from pathlib import Path
import re

ie = Path(r"F:\dataset\isign_extracted")
videos_dir = ie / "iSign-videos_v1.1"

# Get sample filenames to understand pattern
files = sorted(videos_dir.iterdir())[:100]
print("First 100 filenames:")
for f in files:
    print(f"  {f.name}")

# Try to match word_ids from CSVs against filenames
csv_path = ie / "word-presence-dataset_v1.1.csv"
word_ids = set()
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        wid = row['word_id'].strip()
        if wid:
            word_ids.add(wid)

print(f"\nTotal unique word_ids in CSV: {len(word_ids)}")
print(f"Sample word_ids: {list(word_ids)[:10]}")

# Check how many video filenames match word_ids
video_files = list(videos_dir.iterdir())
print(f"\nTotal video files: {len(video_files)}")

matches = 0
no_match = 0
for vf in video_files:
    stem = vf.stem
    if stem in word_ids:
        matches += 1
    else:
        no_match += 1

print(f"Videos matching word_id: {matches}")
print(f"Videos NOT matching word_id: {no_match}")

# Show some non-matching stems to understand pattern
non_matching = []
for vf in video_files:
    stem = vf.stem
    if stem not in word_ids:
        non_matching.append(stem)
        if len(non_matching) >= 20:
            break
print(f"\nSample non-matching stems:")
for s in non_matching:
    print(f"  {s}")
