import csv
from pathlib import Path
csv_path = Path(r'D:\MookVani\backend\segmentor\data_builder\segmentation.csv')
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
print(f'Total boundary marks: {len(rows)}')
if rows:
    by_sentence = {}
    for r in rows:
        by_sentence.setdefault(r['sentence'], set()).add(r['sentence_id'])
    print(f'Unique sentences marked: {len(by_sentence)}')
    for sent, ids in by_sentence.items():
        sent_rows = [r for r in rows if r['sentence'] == sent]
        frames = sorted([int(r['frame_index']) for r in sent_rows])
        print(f'  "{sent}": {len(sent_rows)} marks, frames={frames[:10]}{"..." if len(frames)>10 else ""}')