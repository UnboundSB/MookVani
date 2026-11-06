"""
Clean segmentation CSV:
- Deduplicate boundary frames within tolerance (e.g., 17,17,17,18,19 → single frame per boundary)
- Normalize multiple lines per sentence
- Save cleaned version
"""

import csv
import numpy as np
from pathlib import Path

SEG_CSV = Path(r"D:\MookVani\backend\segmentor\data_builder\segmentation.csv")
CLEAN_CSV = Path(r"D:\MookVani\backend\segmentor\data_builder\segmentation_clean.csv")

WINDOW = 5  # frames tolerance for deduplication

with open(SEG_CSV, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Original rows: {len(rows)}")

# Group by sentence_id
by_sentence = {}
for r in rows:
    by_sentence.setdefault(r['sentence_id'], []).append(r)

cleaned = []
for sent_id, sent_rows in by_sentence.items():
    # Sort by frame
    sent_rows.sort(key=lambda x: int(x['frame_index']))
    
    # Deduplicate within WINDOW frames
    clusters = []
    current = [sent_rows[0]]
    for r in sent_rows[1:]:
        f = int(r['frame_index'])
        last_f = int(current[-1]['frame_index'])
        if f - last_f <= WINDOW:
            current.append(r)
        else:
            clusters.append(current)
            current = [r]
    clusters.append(current)
    
    # Keep middle frame of each cluster
    for cluster in clusters:
        if len(cluster) == 1:
            cleaned.append(cluster[0])
        else:
            mid = len(cluster) // 2
            cleaned.append(cluster[mid])

print(f"Cleaned rows: {len(cleaned)}")

# Write cleaned CSV
with open(CLEAN_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [
        'sentence_hash', 'sentence', 'sentence_id', 'sentence_video',
        'word', 'word_video', 'frame_index', 'timestamp', 'accepted', 'marked_at'
    ])
    writer.writeheader()
    writer.writerows(cleaned)

# Summary
by_sentence_clean = {}
for r in cleaned:
    by_sentence_clean.setdefault(r['sentence_id'], 0)
    by_sentence_clean[r['sentence_id']] += 1

print(f"\nUnique sentences: {len(by_sentence_clean)}")
for sid, cnt in sorted(by_sentence_clean.items(), key=lambda x: -x[1]):
    print(f"  {sid}: {cnt} boundaries")