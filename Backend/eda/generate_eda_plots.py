import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 1. Parse INCLUDE50 (archive_extracted)
archive_path = Path(r'F:\dataset\archive_extracted')
categories = []
word_counts = []
samples_per_word_include = []

for category_dir in archive_path.iterdir():
    if category_dir.is_dir():
        cat_name = category_dir.name
        words_in_cat = 0
        for word_dir in category_dir.iterdir():
            if word_dir.is_dir():
                words_in_cat += 1
                videos = list(word_dir.glob('*.*'))
                samples_per_word_include.append({'Word': word_dir.name, 'Category': cat_name, 'Count': len(videos)})
        categories.append(cat_name)
        word_counts.append(words_in_cat)

df_include_cat = pd.DataFrame({'Category': categories, 'Word Count': word_counts})
df_include_cat = df_include_cat.sort_values('Word Count', ascending=False)
df_include_samples = pd.DataFrame(samples_per_word_include)

# 3. Plotting
sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(20, 10))

# Subplot 1: Words per category in INCLUDE50
ax1 = plt.subplot(1, 2, 1)
# Use hue to allow a legend
sns.barplot(data=df_include_cat, x='Category', y='Word Count', hue='Category', ax=ax1, palette='Set2', dodge=False)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=10)
ax1.set_title('Number of Words per Category in INCLUDE50', fontsize=16, fontweight='bold')
ax1.set_xlabel('Category', fontsize=12)
ax1.set_ylabel('Total Number of Words', fontsize=12)
ax1.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')

for p in ax1.patches:
    if p.get_height() > 0:
        ax1.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontsize=10)

# Subplot 2: Samples per class distribution (INCLUDE50)
ax2 = plt.subplot(1, 2, 2)
# Histogram of samples per word, with legend showing mean
sns.histplot(data=df_include_samples, x='Count', bins=30, ax=ax2, color='coral', kde=True, label='Words (Classes)')
mean_count = df_include_samples['Count'].mean()
ax2.axvline(mean_count, color='red', linestyle='--', label=f'Mean = {mean_count:.1f} samples/word')
ax2.set_title('Distribution of Videos per Word (INCLUDE50)', fontsize=16, fontweight='bold')
ax2.set_xlabel('Number of Videos present per Word', fontsize=12)
ax2.set_ylabel('Frequency (Number of Words)', fontsize=12)
ax2.legend(title='Legend', fontsize=12, loc='upper right')

plt.tight_layout()
output_path = r'd:\MookVani\Backend\eda\plots\include50_eda_plots.png'
plt.savefig(output_path, dpi=300)
print(f"Plots saved successfully to {output_path}")
