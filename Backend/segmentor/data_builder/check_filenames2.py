import csv
from pathlib import Path

ie = Path(r"F:\dataset\isign_extracted")
videos_dir = ie / "iSign-videos_v1.1"

# Load word_ids from both CSVs
word_ids = set()
for csv_name in ["word-description-dataset_v1.1.csv", "word-presence-dataset_v1.1.csv"]:
    csv_path = ie / csv_name
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            wid = row['word_id'].strip()
            if wid:
                word_ids.add(wid)

print(f"Total unique word_ids in CSVs: {len(word_ids)}")
print(f"Sample word_ids: {list(word_ids)[:10]}")

# Check video filenames
video_files = list(videos_dir.iterdir())
print(f"\nTotal video files: {len(video_files)}")

# Try matching: strip _w from word_id, compare to part before -- in filename
csv_stems = set()
for wid in word_ids:
    if wid.endswith('_w'):
        csv_stems.add(wid[:-2])  # remove _w
    else:
        csv_stems.add(wid)

print(f"CSV stems (without _w): {len(csv_stems)}")

matches = 0
no_match = 0
matched_words = set()
for vf in video_files:
    stem = vf.stem
    # Extract part before --
    if '--' in stem:
        base = stem.split('--')[0]
    else:
        base = stem
    
    if base in csv_stems:
        matches += 1
        matched_words.add(base)
    else:
        no_match += 1

print(f"\nVideos matching word_id base: {matches}")
print(f"Videos NOT matching: {no_match}")
print(f"Unique matched words: {len(matched_words)}")

# Show non-matching stems
non_matching = set()
for vf in video_files:
    stem = vf.stem
    if '--' in stem:
        base = stem.split('--')[0]
    else:
        base = stem
    if base not in csv_stems:
        non_matching.add(base)
        if len(non_matching) >= 20:
            break

print(f"\nSample non-matching stems:")
for s in sorted(non_matching)[:20]:
    print(f"  {s}")
