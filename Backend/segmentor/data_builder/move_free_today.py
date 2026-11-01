import csv
from pathlib import Path

csv_path = Path(r'D:\MookVani\backend\segmentor\data_builder\sentences.csv')

with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

# Split: keep non-"are you free today" first, then "are you free today" at bottom
other_rows = [r for r in rows if r['sentence'].strip().lower() != 'are you free today']
free_rows = [r for r in rows if r['sentence'].strip().lower() == 'are you free today']

print(f"Total: {len(rows)}")
print(f"Other: {len(other_rows)}")
print(f"'are you free today': {len(free_rows)}")

# Reorder
new_rows = other_rows + free_rows

with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(new_rows)

print("Done - 'are you free today' moved to bottom")