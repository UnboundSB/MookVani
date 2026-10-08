import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

desc_csv = r'F:\dataset\isign\word-description-dataset_v1.1.csv'
pres_csv = r'F:\dataset\isign\word-presence-dataset_v1.1.csv'

df_desc = pd.read_csv(desc_csv)
df_pres = pd.read_csv(pres_csv)

# Analysis
# 1. Number of sentences per word
desc_counts = df_desc['word'].value_counts()
pres_counts = df_pres['word'].value_counts()

# 2. Sentence Lengths
df_desc['sentence_length'] = df_desc['sentence'].apply(lambda x: len(str(x).split()))
df_pres['sentence_length'] = df_pres['sentence'].apply(lambda x: len(str(x).split()))

# Plotting
sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(20, 15))

# Subplot 1: Distribution of Sentences per Word in Word Presence Dataset
ax1 = plt.subplot(2, 2, 1)
sns.histplot(pres_counts, bins=20, color='skyblue', ax=ax1, kde=False)
ax1.set_title('Number of Sentences per Word\n(Word Presence Dataset)', fontsize=16, fontweight='bold')
ax1.set_xlabel('Number of Example Sentences', fontsize=12)
ax1.set_ylabel('Frequency (Number of Words)', fontsize=12)

# Add text for max and mean
pres_mean = pres_counts.mean()
ax1.axvline(pres_mean, color='red', linestyle='--', label=f'Mean = {pres_mean:.1f}')
ax1.legend(fontsize=12)

# Subplot 2: Top 15 Words with Most Sentences (Word Presence)
ax2 = plt.subplot(2, 2, 2)
top_pres = pres_counts.head(15).reset_index()
top_pres.columns = ['Word', 'Count']
sns.barplot(data=top_pres, x='Word', y='Count', hue='Word', palette='viridis', ax=ax2, dodge=False)
ax2.set_title('Top 15 Words with Most Example Sentences', fontsize=16, fontweight='bold')
ax2.set_xlabel('Word', fontsize=12)
ax2.set_ylabel('Number of Sentences', fontsize=12)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=10)
# Add labels on top of bars
for p in ax2.patches:
    if p.get_height() > 0:
        ax2.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontsize=10)

# Subplot 3: Sentence Length Distribution (Word Description)
ax3 = plt.subplot(2, 2, 3)
sns.histplot(df_desc['sentence_length'], bins=20, color='lightgreen', ax=ax3, kde=True, label='Description Lengths')
desc_len_mean = df_desc['sentence_length'].mean()
ax3.axvline(desc_len_mean, color='darkgreen', linestyle='--', label=f'Mean = {desc_len_mean:.1f} words')
ax3.set_title('Distribution of Description Lengths', fontsize=16, fontweight='bold')
ax3.set_xlabel('Number of Words in Description', fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)
ax3.legend(fontsize=12)

# Subplot 4: Sentence Length Distribution (Word Presence)
ax4 = plt.subplot(2, 2, 4)
sns.histplot(df_pres['sentence_length'], bins=20, color='orchid', ax=ax4, kde=True, label='Example Sentence Lengths')
pres_len_mean = df_pres['sentence_length'].mean()
ax4.axvline(pres_len_mean, color='purple', linestyle='--', label=f'Mean = {pres_len_mean:.1f} words')
ax4.set_title('Distribution of Example Sentence Lengths', fontsize=16, fontweight='bold')
ax4.set_xlabel('Number of Words in Example Sentence', fontsize=12)
ax4.set_ylabel('Frequency', fontsize=12)
ax4.legend(fontsize=12)

plt.tight_layout()
output_path = r'd:\MookVani\Backend\eda\plots\isign_eda_plots.png'
plt.savefig(output_path, dpi=300)
print(f"Plots saved successfully to {output_path}")

print(f"Word Description Dataset: {len(df_desc)} rows, {df_desc['word'].nunique()} unique words.")
print(f"Word Presence Dataset: {len(df_pres)} rows, {df_pres['word'].nunique()} unique words.")
