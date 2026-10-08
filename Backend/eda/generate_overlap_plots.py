import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def clean_word(w):
    if pd.isna(w):
        return ""
    w = str(w).strip().lower()
    w = re.sub(r'^\d+\.\s*', '', w)
    w = re.sub(r'[^\w\s]', '', w)
    return w.strip()

# 1. Load INCLUDE50 and map words to categories
archive_path = Path(r'F:\dataset\archive_extracted')
include50_data = []

for category_dir in archive_path.iterdir():
    if category_dir.is_dir():
        cat_name = category_dir.name
        for word_dir in category_dir.iterdir():
            if word_dir.is_dir():
                cw = clean_word(word_dir.name)
                if cw:
                    include50_data.append({'Category': cat_name, 'Word': cw})

df_include = pd.DataFrame(include50_data)

# 2. Load full iSign texts
main_csv = r'd:\MookVani\iSign_v1.1.csv'
df_main = pd.read_csv(main_csv)
isign_main_texts = set()
for text in df_main['text']:
    cw = clean_word(text)
    if cw:
        isign_main_texts.add(cw)

# 3. Calculate overlap per category
df_include['Overlaps_with_iSign'] = df_include['Word'].apply(lambda w: w in isign_main_texts)

# Group by category
category_stats = df_include.groupby('Category').agg(
    Total_INCLUDE50=('Word', 'count'),
    iSign_Overlap=('Overlaps_with_iSign', 'sum')
).reset_index()

# Sort by Total_INCLUDE50 descending
category_stats = category_stats.sort_values(by='Total_INCLUDE50', ascending=False)

# Prepare data for plotting
plot_data = pd.melt(category_stats, id_vars=['Category'], 
                    value_vars=['Total_INCLUDE50', 'iSign_Overlap'],
                    var_name='Dataset', value_name='Word Count')

plot_data['Dataset'] = plot_data['Dataset'].map({
    'Total_INCLUDE50': 'Total Words in INCLUDE50',
    'iSign_Overlap': 'Overlapping Words found in iSign'
})

# 4. Plot
sns.set_theme(style="whitegrid")
plt.figure(figsize=(16, 8))

ax = sns.barplot(data=plot_data, x='Category', y='Word Count', hue='Dataset', palette=['skyblue', 'coral'])

plt.title('INCLUDE50 Categories: Total Words vs. iSign Overlap', fontsize=18, fontweight='bold')
plt.xlabel('Word Category', fontsize=14)
plt.ylabel('Number of Words', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.legend(title='Legend', fontsize=12, title_fontsize=14)

# Add value labels on bars
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f"{int(p.get_height())}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
output_path = r'd:\MookVani\Backend\eda\plots\include50\overlap_distro.png'
plt.savefig(output_path, dpi=300)
print(f"Plot saved successfully to {output_path}")
