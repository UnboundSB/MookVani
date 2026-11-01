import csv
with open(r'D:\MookVani\backend\segmentor\data_builder\word_videos.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['sentence_id'] == 'are_you_free_today':
            print(f"  word='{row['word']}' video='{row['word_video_path']}' source={row['source']}")