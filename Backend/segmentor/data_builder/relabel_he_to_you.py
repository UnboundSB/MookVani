import csv
from pathlib import Path

csv_path = Path(r'D:\MookVani\backend\segmentor\data_builder\sentences.csv')

with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

# Find all "he is going into room" sentences
target = "he is going into room"
indices = [i for i, r in enumerate(rows) if r['sentence'].strip().lower() == target]

print(f"Found {len(indices)} occurrences of '{target}'")
for idx in indices:
    print(f"  Row {idx}: sentence_id={rows[idx]['sentence_id']}")

# Change all except first 3
new_sentence = "you go into room"
changed = 0
for i, idx in enumerate(indices):
    if i >= 3:  # skip first 3
        rows[idx]['sentence'] = new_sentence
        # Also update sentence_hash
        import hashlib
        rows[idx]['sentence_hash'] = hashlib.sha256(new_sentence.strip().encode('utf-8')).hexdigest()
        changed += 1

print(f"Changed {changed} rows to '{new_sentence}'")

with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("Done")