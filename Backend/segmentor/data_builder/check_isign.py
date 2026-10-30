import csv
from collections import Counter
from pathlib import Path

csv_path = Path(r'F:\dataset\isign\word-presence-dataset_v1.1.csv')

with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print('Total rows:', len(rows))
print('Columns:', list(rows[0].keys()))
print()

sentence_ids = set(r['sentence_id'] for r in rows)
print('Unique sentence_ids:', len(sentence_ids))

words_per_sentence = Counter()
for r in rows:
    words_per_sentence[r['sentence_id']] += 1

print()
print('Words per sentence distribution:')
for n in sorted(set(words_per_sentence.values()), reverse=True)[:10]:
    count = sum(1 for v in words_per_sentence.values() if v == n)
    print('  %d words: %d sentences' % (n, count))

print()
print('Sample rows:')
for r in rows[:5]:
    print('  word_id=%s word=%s sentence_id=%s sentence=%s' % (
        r['word_id'], r['word'], r['sentence_id'], r['sentence'][:60]))
