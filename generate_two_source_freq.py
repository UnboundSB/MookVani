import json, re, csv
from pathlib import Path
from collections import Counter

def clean(w):
    w = str(w).strip().lower()
    w = re.sub(r"^\d+\.\s*", "", w)
    w = re.sub(r"[^\w\s]", "", w)
    return w.strip()

combined = Counter()
sources_detail = {}

classes = json.load(open("D:/mookvani/Backend/models/word_vocabulary/classes.json"))
classes_set = set(classes)

# === SOURCE 1: ISIGN_EXTRACTED CSVs (only column index 1 = 'word') ===
isign = Path("F:/dataset/isign_extracted")
csv_total = 0
for fname in ["word-presence-dataset_v1.1.csv", "word-description-dataset_v1.1.csv"]:
    fpath = isign / fname
    if not fpath.exists():
        print(f"WARNING: {fpath} not found")
        continue
    with open(fpath) as f:
        r = csv.reader(f)
        header = next(r)  # ['word_id', 'word', 'sentence_id', 'sentence']
        # Only column index 1 (word). Ignore sentence_id (col 2) and sentence (col 3).
        for row in r:
            w = row[1].strip().lower()
            combined[w] += 1
            csv_total += 1
            sources_detail.setdefault(w, {})[fname] = sources_detail.get(w, {}).get(fname, 0) + 1

print(f"ISIGN CSVs: {csv_total} word occurrences (column 'word' only)")

# === SOURCE 2: ARCHIVE_EXTRACTED (video folders) ===
archive = Path("F:/dataset/archive_extracted")
archive_total = 0
for cat in archive.iterdir():
    if not cat.is_dir():
        continue
    for w in cat.iterdir():
        if not w.is_dir():
            continue
        cn = clean(w.name)
        vids = [f for f in w.iterdir() if f.is_file() and f.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv")]
        if cn:
            combined[cn] += len(vids)
            archive_total += len(vids)
            sources_detail.setdefault(cn, {})["archive_extracted"] = sources_detail.get(cn, {}).get("archive_extracted", 0) + len(vids)

print(f"Archive extracted: {archive_total} videos")

# Write CSV
out_path = Path("D:/mookvani/isign_archive_word_freq.csv")
with open(out_path, "w", newline="") as f:
    wtr = csv.writer(f)
    wtr.writerow(["word", "total_frequency", "in_classes_json", "isign_csv_count", "archive_extracted_count", "sources_detail"])
    for word, count in combined.most_common():
        isign_count = sum(v for k, v in sources_detail.get(word, {}).items() if k.endswith(".csv"))
        archive_count = sources_detail.get(word, {}).get("archive_extracted", 0)
        in_cj = "YES" if word in classes_set else "NO"
        srcs = "; ".join(f"{k}={v}" for k, v in sorted(sources_detail.get(word, {}).items()))
        wtr.writerow([word, count, in_cj, isign_count, archive_count, srcs])

print(f"\nWritten: {out_path}")
print(f"Total unique words: {len(combined)}")
print(f"Total cumulative count: {sum(combined.values())}")
print(f"In classes.json: {len(set(combined) & classes_set)}")
print(f"NOT in classes.json: {len(set(combined) - classes_set)}")
print()
print("TOP 40 (descending):")
for word, count in combined.most_common(40):
    tag = "IN_CJ" if word in classes_set else "NOT_IN_CJ"
    print(f"  {word:30s} {count:5d}  {tag}")
