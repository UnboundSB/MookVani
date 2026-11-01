import csv
with open(r'D:\MookVani\backend\segmentor\data_builder\segmentation.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
print(f'Total rows: {len(rows)}')
for row in rows:
    print(f"  sentence='{row['sentence']}' frame={row['frame_index']} ts={row['timestamp']}")