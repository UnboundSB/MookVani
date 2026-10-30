import csv
from pathlib import Path

ie = Path(r"F:\dataset\isign_extracted")
videos_dir = ie / "iSign-videos_v1.1"

# Load word_ids from both CSVs
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

print("Total unique word_ids in CSVs:", len(word_ids))

# Try matching video filenames to word_ids
video_files = list(videos_dir.iterdir())
print("Total video files:", len(video_files))

matches = 0
no_match = 0
matched_words = set()

for vf in video_files:
    stem = vf.stem
    
    # Direct match: filename stem == word_id
    if stem in word_ids:
        matches += 1
        matched_words.add(stem)
        continue
    
    # Strip _w from stem and check
    if stem.endswith('_w'):
        base = stem[:-2]
        if base in word_ids:
            matches += 1
            matched_words.add(base)
            continue
    
    # Strip --N from stem and check
    if '--' in stem:
        base = stem.split('--')[0]
        if base in word_ids:
            matches += 1
            matched_words.add(base)
            continue
    
    no_match += 1

print("\nVideos matching word_id:", matches)
print("Videos NOT matching:", no_match)
print("Unique matched words:", len(matched_words))

# Show matched words
print("\nSample matched words:")
for w in sorted(matched_words)[:20]:
    print("  ", w, "->", word_id_to_word.get(w, w + '_w'))
