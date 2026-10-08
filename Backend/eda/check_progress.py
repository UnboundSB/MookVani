import os
import sys

target = r"F:\dataset\isign_extracted"
if os.path.exists(target):
    count = sum([len(files) for r, d, files in os.walk(target)])
    print(f"File count: {count}")
else:
    print("Folder does not exist yet.")
