import csv
from pathlib import Path

csv_path = Path(r'D:\MookVani\backend\segmentor\data_builder\segmentation.csv')

with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f'Original: {len(rows)} marks')

# Group by sentence_id
by_sentence = {}
for r in rows:
    by_sentence.setdefault(r['sentence_id'], []).append(r)

cleaned = []
WINDOW = 5  # frames

for sent_id, sent_rows in by_sentence.items():
    # Sort by frame
    sent_rows.sort(key=lambda x: int(x['frame_index']))
    
    clusters = []
    current_cluster = [sent_rows[0]]
    
    for r in sent_rows[1:]:
        f = int(r['frame_index'])
        last_f = int(current_cluster[-1]['frame_index'])
        if f - last_f <= WINDOW:
            current_cluster.append(r)
        else:
            clusters.append(current_cluster)
            current_cluster = [r]
    clusters.append(current_cluster)
    
    # For each cluster, keep middle frame (or last if even)
    for cluster in clusters:
        if len(cluster) == 1:
            cleaned.append(cluster[0])
        else:
            # Keep middle (floor for even)
            idx = len(cluster) // 2
            cleaned.append(cluster[idx])

print(f'Cleaned: {len(cleaned)} marks')

# Write back
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [
        'sentence_hash', 'sentence', 'sentence_id', 'sentence_video',
        'word', 'word_video', 'frame_index', 'timestamp', 'accepted', 'marked_at'
    ])
    writer.writeheader()
    writer.writerows(cleaned)

# Summary
by_sent = {}
for r in cleaned:
    by_sent.setdefault(r['sentence'], 0)
    by_sent[r['sentence']] += 1

print(f'\nUnique sentences: {len(by_sent)}')
for sent, cnt in sorted(by_sent.items(), key=lambda x: -x[1]):
    print(f'  {cnt:2d} marks: {sent}')