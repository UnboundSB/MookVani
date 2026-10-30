import os
from pathlib import Path

ds = Path(r"F:\dataset")
print("Directories in F:\\dataset:")
for d in sorted(ds.iterdir()):
    if d.is_dir():
        print(f"  DIR: {d.name}")
    else:
        print(f"  FILE: {d.name} ({d.stat().st_size} bytes)")

print("\n--- archive_extracted subdirs ---")
ae = ds / "archive_extracted"
if ae.exists():
    for d in sorted(ae.iterdir()):
        if d.is_dir():
            print(f"  {d.name}")

print("\n--- isign_extracted subdirs ---")
ie = ds / "isign_extracted"
if ie.exists():
    for d in sorted(ie.iterdir()):
        if d.is_dir():
            print(f"  {d.name}")

print("\n--- tensors_include50_word_level subdirs ---")
ti = ds / "tensors_include50_word_level"
if ti.exists():
    count = 0
    for d in sorted(ti.iterdir()):
        if d.is_dir():
            count += 1
            if count <= 20:
                print(f"  {d.name}")
    print(f"  ... total {count} word folders")
