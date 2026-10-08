import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def clean_word(w):
    if pd.isna(w): return ""
    w = str(w).strip().lower()
    w = re.sub(r'^\d+\.\s*', '', w)
    w = re.sub(r'[^\w\s]', '', w)
    return w.strip()

# 1. Get INCLUDE50 counts per word class
archive_path = Path(r'F:\dataset\archive_extracted')
include50_counts = {}

for category_dir in archive_path.iterdir():
    if category_dir.is_dir():
        for word_dir in category_dir.iterdir():
            if word_dir.is_dir():
                cw = clean_word(word_dir.name)
                if cw:
                    # Count number of video files
                    vid_count = len(list(word_dir.glob('*.*')))
                    include50_counts[cw] = include50_counts.get(cw, 0) + vid_count

# 2. Get iSign counts per word class
main_csv = r'd:\MookVani\iSign_v1.1.csv'
df_main = pd.read_csv(main_csv)
df_main['clean_text'] = df_main['text'].apply(clean_word)
isign_counts = df_main.groupby('clean_text').size().to_dict()

# 3. Combine counts for the 262 INCLUDE50 classes
merged_data = []
for word, inc_count in include50_counts.items():
    isign_c = isign_counts.get(word, 0)
    total_c = inc_count + isign_c
    merged_data.append({
        'Word Class': word,
        'INCLUDE50_Count': inc_count,
        'iSign_Bump': isign_c,
        'Total_Merged_Count': total_c
    })

df_merged = pd.DataFrame(merged_data)
# Sort from most populated to least populated after merge
df_merged = df_merged.sort_values(by='Total_Merged_Count', ascending=False).reset_index(drop=True)

# 4. Generate Chunked Stacked Bar Plots
chunk_size = 40
total_chunks = (len(df_merged) // chunk_size) + (1 if len(df_merged) % chunk_size != 0 else 0)

sns.set_theme(style="whitegrid")
output_dir = r'd:\MookVani\Backend\eda\plots\include50'
os.makedirs(output_dir, exist_ok=True)

for i in range(total_chunks):
    chunk = df_merged.iloc[i*chunk_size : (i+1)*chunk_size]
    
    plt.figure(figsize=(18, 8))
    
    # We will plot the Total first (which will act as the background/top bar)
    # Then plot the INCLUDE50 count over it (which will act as the bottom bar)
    
    bar1 = sns.barplot(data=chunk, x='Word Class', y='Total_Merged_Count', color='coral', label='Bump from iSign')
    bar2 = sns.barplot(data=chunk, x='Word Class', y='INCLUDE50_Count', color='skyblue', label='Original INCLUDE50 Samples')
    
    plt.title(f'Merged Dataset Classes (Most to Least Populated) - Part {i+1}/{total_chunks}', fontsize=16, fontweight='bold')
    plt.xlabel('Word Class (Sign)', fontsize=12)
    plt.ylabel('Total Number of Videos (Samples)', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.legend(loc='upper right', fontsize=12)
    
    # Add count labels on top of the bars for the total merged count
    for idx, p in enumerate(bar1.patches[:len(chunk)]):
        height = p.get_height()
        if height > 0:
            bar1.annotate(f"{int(height)}", 
                          (p.get_x() + p.get_width() / 2., height),
                          ha='center', va='bottom', fontsize=9, fontweight='bold')
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'merge_bump_{i+1}.png'), dpi=200)
    plt.close()

print(f"Generated {total_chunks} chunked stacked bar plots for all 262 classes.")
