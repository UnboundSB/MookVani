import os
import re
import pandas as pd
from pathlib import Path

def clean_word(w):
    if pd.isna(w):
        return ""
    w = str(w).strip().lower()
    # Remove leading numbers and dots (e.g., '1. dog' -> 'dog')
    w = re.sub(r'^\d+\.\s*', '', w)
    # Remove all punctuation just in case
    w = re.sub(r'[^\w\s]', '', w)
    return w.strip()

# 1. Get INCLUDE50 words
archive_path = Path(r'F:\dataset\archive_extracted')
include50_words = set()

for category_dir in archive_path.iterdir():
    if category_dir.is_dir():
        for word_dir in category_dir.iterdir():
            if word_dir.is_dir():
                cw = clean_word(word_dir.name)
                if cw:
                    include50_words.add(cw)

# 2. Get ALL iSign texts from the main 127k CSV
main_csv = r'd:\MookVani\iSign_v1.1.csv'
df_main = pd.read_csv(main_csv)

isign_main_texts = set()
for text in df_main['text']:
    cw = clean_word(text)
    if cw:
        isign_main_texts.add(cw)

# 3. Calculate collision
collision_all = include50_words.intersection(isign_main_texts)

print(f"INCLUDE50 unique words: {len(include50_words)}")
print(f"iSign unique texts (from entire 127k dataset): {len(isign_main_texts)}")
print(f"Number of colliding words with entire iSign dataset: {len(collision_all)}")

if len(collision_all) > 0:
    print(f"\nColliding words ({len(collision_all)}):")
    # Print them nicely
    cols = sorted(list(collision_all))
    print(', '.join(cols))
