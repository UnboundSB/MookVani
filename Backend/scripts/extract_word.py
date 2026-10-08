import os
import glob
import torch
import re
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import multiprocessing

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.extract_video import extract_video_features

def clean_word(w):
    import pandas as pd
    if pd.isna(w): return ""
    w = str(w).strip().lower()
    w = re.sub(r'^\d+\.\s*', '', w)
    w = re.sub(r'[^\w\s]', '', w)
    return w.strip()

def process_single_video(vid_path, output_dir):
    word_dir_name = os.path.basename(os.path.dirname(vid_path))
    class_name = clean_word(word_dir_name)
    if not class_name:
        return False
        
    img_name = os.path.splitext(os.path.basename(vid_path))[0]
    save_dir = os.path.join(output_dir, class_name)
    
    # exist_ok doesn't have race conditions in python 3.2+
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{img_name}.pt")
    
    if os.path.exists(save_path):
        return True # already extracted

    tensor_data = extract_video_features(vid_path)
    if tensor_data is not None:
        torch.save(tensor_data, save_path)
        return True
    return False

def extract_include50(source_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    all_videos = []
    for ext in ('*.mp4', '*.MP4', '*.mov', '*.MOV'):
        all_videos.extend(glob.glob(f"{source_dir}/*/*/{ext}"))
        
    print(f"Found {len(all_videos)} videos in INCLUDE50.")
    print("Starting multiprocessing extraction...")
    
    success_count = 0
    # User requested exactly 50% of the CPU cores (e.g. 5 out of 10)
    num_workers = max(1, multiprocessing.cpu_count() // 2)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_video, vid, output_dir): vid for vid in all_videos}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
            if future.result():
                success_count += 1
                
    print(f"Extracted {success_count}/{len(all_videos)} videos.")

if __name__ == "__main__":
    # Needed for safe multiprocessing on Windows
    multiprocessing.freeze_support()
    SOURCE_DIR = r"F:\dataset\archive_extracted"
    OUTPUT_DIR = "d:/MookVani/Backend/data/tensors_include50_word_level"
    extract_include50(SOURCE_DIR, OUTPUT_DIR)
