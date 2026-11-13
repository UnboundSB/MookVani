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

# 1. Archive extracted
archive = Path("F:/dataset/archive_extracted")
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
            sources_detail.setdefault(cn, {})["archive_extracted"] = sources_detail.get(cn, {}).get("archive_extracted", 0) + len(vids)

# 2. iSign CSVs
for fname in ["word-presence-dataset_v1.1.csv", "word-description-dataset_v1.1.csv"]:
    with open("F:/dataset/isign/" + fname) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            w = row[1].strip().lower()
            combined[w] += 1
            sources_detail.setdefault(w, {})[fname] = sources_detail.get(w, {}).get(fname, 0) + 1

# 3. Tensors F:
ft = Path("F:/dataset/tensors_include50_word_level")
for d in ft.iterdir():
    if not d.is_dir():
        continue
    pts = [f for f in d.iterdir() if f.suffix == ".pt"]
    if pts:
        combined[d.name] += len(pts)
        sources_detail.setdefault(d.name, {})["tensors_F"] = sources_detail.get(d.name, {}).get("tensors_F", 0) + len(pts)

# 4. Tensors Backend
bt = Path("D:/mookvani/Backend/data/tensors_include50_word_level")
for d in bt.iterdir():
    if not d.is_dir() or d.name in ("_smoke_test",):
        continue
    pts = [f for f in d.iterdir() if f.suffix == ".pt"]
    if pts:
        combined[d.name] += len(pts)
        sources_detail.setdefault(d.name, {})["tensors_Backend"] = sources_detail.get(d.name, {}).get("tensors_Backend", 0) + len(pts)

# 5. Cached coords (EXCLUDE isign category - those are video IDs, not words)
cc = Path("F:/dataset/cached_coords")
for cat in cc.iterdir():
    if not cat.is_dir() or cat.name == "isign":
        continue
    for d in cat.iterdir():
        if not d.is_dir():
            continue
        files = [f for f in d.iterdir() if f.is_file()]
        if files:
            cn = clean(d.name)
            if cn:
                combined[cn] += len(files)
                sources_detail.setdefault(cn, {})["cached_coords"] = sources_detail.get(cn, {}).get("cached_coords", 0) + len(files)

# Write CSV
out_path = Path("D:/mookvani/all_words_frequency_report.csv")
with open(out_path, "w", newline="") as f:
    wtr = csv.writer(f)
    wtr.writerow(["word", "total_frequency", "in_classes_json", "sources"])
    for word, count in combined.most_common():
        in_cj = "YES" if word in classes_set else "NO"
        srcs = "; ".join(f"{k}={v}" for k, v in sorted(sources_detail.get(word, {}).items()))
        wtr.writerow([word, count, in_cj, srcs])

print("Written:", out_path)
print("Total unique words:", len(combined))
print("Total combined count:", sum(combined.values()))
print("In classes.json:", len(set(combined) & classes_set))
print("NOT in classes.json:", len(set(combined) - classes_set))
print()
print("TOP 40:")
for word, count in combined.most_common(40):
    tag = "IN_CJ" if word in classes_set else "NOT_IN_CJ"
    print(f"  {word:30s} {count:5d}  {tag}")
