import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

output_dir = r'd:\MookVani\Backend\eda\plots'
os.makedirs(output_dir, exist_ok=True)

pres_csv = r'F:\dataset\isign\word-presence-dataset_v1.1.csv'
df_pres = pd.read_csv(pres_csv)

# 1. Frequency of words (how many sentences each word appears in)
word_counts = df_pres['word'].value_counts().reset_index()
word_counts.columns = ['Word', 'Sentence_Count']
word_counts = word_counts.sort_values(by='Sentence_Count', ascending=False)

# Chunk words to avoid cramping
chunk_size = 30
total_chunks = (len(word_counts) // chunk_size) + (1 if len(word_counts) % chunk_size != 0 else 0)

sns.set_theme(style="whitegrid")

for i in range(total_chunks):
    chunk = word_counts.iloc[i*chunk_size : (i+1)*chunk_size]
    
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(data=chunk, x='Word', y='Sentence_Count', palette='viridis', dodge=False)
    plt.title(f'Sentence-wise Frequency of Words (Part {i+1}/{total_chunks})', fontsize=16, fontweight='bold')
    plt.xlabel('Word', fontsize=12)
    plt.ylabel('Number of Sentences', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    # Add count labels on bars
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=9)
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'word_sent_distro_{i+1}.png'), dpi=200)
    plt.close()

# 2. Distro of Sentences (Sentence Lengths)
df_pres['sentence_length'] = df_pres['sentence'].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(10, 6))
sns.histplot(df_pres['sentence_length'], bins=25, color='coral', kde=True)
plt.title('Distribution of Sentence Lengths (Words per Sentence)', fontsize=16, fontweight='bold')
plt.xlabel('Number of Words in Sentence', fontsize=12)
plt.ylabel('Frequency (Number of Sentences)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'sentence_length_distro.png'), dpi=200)
plt.close()

print(f"Generated {total_chunks} word_sent_distro plots and 1 sentence_length_distro plot.")
