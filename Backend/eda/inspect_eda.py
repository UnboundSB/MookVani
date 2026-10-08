import os
import pandas as pd

def list_dirs(path, depth=1):
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        if level < depth:
            print(f"{'  ' * level}{os.path.basename(root)}/ ({len(dirs)} dirs, {len(files)} files)")
        else:
            del dirs[:] # stop recursion

print("--- archive_extracted ---")
list_dirs(r'F:\dataset\archive_extracted', depth=2)

print("\n--- isign CSVs ---")
csv_paths = [
    r'd:\MookVani\iSign_v1.1.csv',
    r'd:\MookVani\indian-sign-language-video-dataset-metadata.json',
    r'F:\dataset\isign\word-description-dataset_v1.1.csv',
    r'F:\dataset\isign\word-presence-dataset_v1.1.csv'
]
for p in csv_paths:
    if os.path.exists(p):
        if p.endswith('.csv'):
            df = pd.read_csv(p)
            print(f"{os.path.basename(p)}: columns={list(df.columns)}, shape={df.shape}")
            if 'Word' in df.columns or 'Category' in df.columns or 'Class' in df.columns:
                print(df.head(2))
        else:
            print(f"{os.path.basename(p)} exists")
    else:
        print(f"{os.path.basename(p)} DOES NOT EXIST")
