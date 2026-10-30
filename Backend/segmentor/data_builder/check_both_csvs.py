import csv
from pathlib import Path

ie = Path(r"F:\dataset\isign_extracted")

# Check both CSVs
for csv_name in ["word-description-dataset_v1.1.csv", "word-presence-dataset_v1.1.csv"]:
    csv_path = ie / csv_name
    print(f"\n=== {csv_name} ===")
    if csv_path.exists():
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            print(f"Columns: {list(rows[0].keys()) if rows else 'N/A'}")
            print(f"Total rows: {len(rows)}")
            print(f"First 5 rows:")
            for row in rows[:5]:
                print(f"  {dict(row)}")
    else:
        print("NOT FOUND")
