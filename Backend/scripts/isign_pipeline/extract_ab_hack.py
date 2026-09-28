import os
import zipfile
import json
import csv
import re
from tqdm import tqdm

class OffsetFile:
    def __init__(self, filename, offset):
        self.f = open(filename, 'rb')
        self.offset = offset
        self.f.seek(0, 2)
        self.real_length = self.f.tell()
        self.f.seek(0)
        
    def seek(self, pos, whence=0):
        if whence == 0:
            if pos < self.offset:
                raise ValueError(f"Trying to seek into missing part_aa! (pos={pos})")
            return self.f.seek(pos - self.offset, 0)
        elif whence == 1:
            return self.f.seek(pos, 1)
        elif whence == 2:
            return self.f.seek(pos, 2)
            
    def read(self, size=-1):
        return self.f.read(size)
        
    def tell(self):
        return self.f.tell() + self.offset
        
    def close(self):
        self.f.close()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def seekable(self):
        return True

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
    part_ab_path = r"D:\MookVani\iSign-videos_v1.1_part_ab"
    csv_path = r"E:\channel\mp\data\isign_dataset\iSign_v1.1.csv"
    vocab_path = r"D:\MookVani\Backend\models\word_class_to_idx.json"
    out_dir = r"D:\MookVani\Backend\data\extracted_videos"
    os.makedirs(out_dir, exist_ok=True)
    
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
                    
    # part_aa was exactly 32212254720 bytes
    PART_AA_SIZE = 32212254720
    
    print("Opening part_ab with virtual offset to bypass the missing part_aa...")
    with OffsetFile(part_ab_path, PART_AA_SIZE) as f:
        with zipfile.ZipFile(f) as zf:
            files = zf.namelist()
            files_to_extract = []
            for fname in files:
                uid_from_file = os.path.basename(fname).replace('.mp4', '')
                if uid_from_file in target_uids:
                    # Check if the file is fully contained in part_ab
                    info = zf.getinfo(fname)
                    # The file's data starts at header_offset + some header size
                    if info.header_offset >= PART_AA_SIZE:
                        files_to_extract.append(fname)
            
            print(f"Found {len(files_to_extract)} matching videos inside part_ab!")
            
            for fname in tqdm(files_to_extract):
                zf.extract(fname, path=out_dir)
                
    print(f"Extraction Complete! Saved to {out_dir}")

if __name__ == "__main__":
    main()
