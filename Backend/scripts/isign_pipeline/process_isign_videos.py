import os
import sys
import glob
import csv
import torch
import json
import re
from tqdm import tqdm
import sys
sys.path.append(r"E:\channel\mp\data")
sys.path.append(r"d:\MookVani\Backend")
from data.extract_video import extract_video_features

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

def process_isign_videos(isign_video_dir):
    vocab_path = r"D:\MookVani\Backend\models\word_class_to_idx.json"
    csv_path = r"E:\channel\mp\data\isign_dataset\iSign_v1.1.csv"
    
    out_word_dir = r"D:\MookVani\Backend\data\tensors_word_level_163_train"
    out_sentence_dir = r"D:\MookVani\Backend\data\tensors_sentence_level_163_train"
    
    if not os.path.exists(isign_video_dir):
        print(f"Error: Could not find video directory {isign_video_dir}")
        return
        
    if not os.path.exists(csv_path):
        print(f"Error: Could not find CSV at {csv_path}")
        return

    valid_words_map = create_word_mapping(vocab_path)
    
    word_targets = {}     # uid -> class_key
    sentence_targets = {} # uid -> sentence_text
    
    print("Parsing CSV to identify target UIDs...")
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for row in reader:
            if len(row) < 2: continue
            uid, text = row[0], row[1]
            cleaned = clean_text(text)
            if not cleaned: continue
            
            if cleaned in valid_words_map:
                word_targets[uid] = valid_words_map[cleaned]
            else:
                words_in_sentence = cleaned.split()
                if len(words_in_sentence) > 1 and all(w in valid_words_map for w in words_in_sentence):
                    sentence_targets[uid] = cleaned

    target_uids = set(word_targets.keys()).union(set(sentence_targets.keys()))
    print(f"Found {len(word_targets)} word targets and {len(sentence_targets)} sentence targets.")
    
    # Locate files in the massive directory
    print(f"Scanning {isign_video_dir} for the {len(target_uids)} target videos...")
    # This recursively searches for the specific UIDs in case they are nested
    all_files = glob.glob(os.path.join(isign_video_dir, "**", "*.mp4"), recursive=True)
    
    # Map filename (without extension) to its full path
    file_map = {os.path.splitext(os.path.basename(f))[0]: f for f in all_files}
    
    found_count = 0
    
    # Process Word Targets
    print("\n--- Processing Word-Level Videos ---")
    for uid, class_key in tqdm(word_targets.items()):
        if uid in file_map:
            found_count += 1
            vid_path = file_map[uid]
            out_dir = os.path.join(out_word_dir, class_key)
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, f"isign_{uid}.pt")
            
            if not os.path.exists(out_file):
                tensor_data = extract_video_features(vid_path)
                if tensor_data is not None:
                    torch.save(tensor_data, out_file)
                    
    # Process Sentence Targets
    print("\n--- Processing Sentence-Level Videos ---")
    for uid, sentence in tqdm(sentence_targets.items()):
        if uid in file_map:
            found_count += 1
            vid_path = file_map[uid]
            # Convert sentence "what happened" to "what_happened" for folder name
            folder_name = sentence.replace(" ", "_")
            out_dir = os.path.join(out_sentence_dir, folder_name)
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, f"isign_{uid}.pt")
            
            if not os.path.exists(out_file):
                tensor_data = extract_video_features(vid_path)
                if tensor_data is not None:
                    torch.save(tensor_data, out_file)
                    
    print(f"\nProcessing complete! Successfully found and extracted {found_count} out of {len(target_uids)} videos.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_isign_videos.py <PATH_TO_EXTRACTED_ISIGN_VIDEOS_ON_E_DRIVE>")
        sys.exit(1)
        
    process_isign_videos(sys.argv[1])
