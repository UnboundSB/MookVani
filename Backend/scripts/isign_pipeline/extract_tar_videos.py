import os
import sys
import subprocess
import json
import re
import csv
from tqdm import tqdm

def create_word_mapping(vocab_path):
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    valid_words = {}
    for key in vocab.keys():
        sub_words = key.split('_')
        for sw in sub_words:
            clean_sw = sw.lower().strip()
            if clean_sw:
                valid_words[clean_sw] = key
    return valid_words

def clean_text(text):
    text = re.sub(r'[^\w\s\']', '', text)
    return text.lower().strip()

def main():
    part_file = r"D:\MookVani\iSign-videos_v1.1_part_ab"
    csv_path = r"E:\channel\mp\data\isign_dataset\iSign_v1.1.csv"
    vocab_path = r"D:\MookVani\Backend\models\word_class_to_idx.json"
    
    out_dir = r"D:\MookVani\Backend\data\extracted_videos"
    os.makedirs(out_dir, exist_ok=True)

    print("1. Identifying Target UIDs from CSV...")
    valid_words_map = create_word_mapping(vocab_path)
    target_uids = set()
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 2: continue
            uid, text = row[0], row[1]
            cleaned = clean_text(text)
            if not cleaned: continue
            
            if cleaned in valid_words_map:
                target_uids.add(uid)
            else:
                words = cleaned.split()
                if len(words) > 1 and all(w in valid_words_map for w in words):
                    target_uids.add(uid)
                    
    print(f"   Found {len(target_uids)} target videos we actually need.")
    
    print("\n2. Scanning massive 30GB archive to find our specific files...")
    # List all files in the archive
    result = subprocess.run(["tar", "-tf", part_file], capture_output=True, text=True)
    all_files = result.stdout.splitlines()
    
    files_to_extract = []
    for f in all_files:
        filename = os.path.basename(f)
        # Check if the filename (without .mp4) is in our target UIDs
        uid_from_file = filename.replace('.mp4', '')
        if uid_from_file in target_uids:
            files_to_extract.append(f)
            
    if not files_to_extract:
        print("   No matching videos found in this specific part file (they might be in part_ab).")
        return
        
    print(f"   Found {len(files_to_extract)} matching videos inside part_aa!")
    
    # Write to a text file for tar to use
    list_file = r"D:\MookVani\Backend\data\files_to_extract.txt"
    with open(list_file, 'w') as f:
        for fname in files_to_extract:
            f.write(f"{fname}\n")
            
    print("\n3. Extracting ONLY our target files (Saving huge amounts of disk space)...")
    subprocess.run(["tar", "-xf", part_file, "-T", list_file, "-C", out_dir])
    
    print(f"\nExtraction Complete! The files are saved in {out_dir}")

if __name__ == "__main__":
    main()
