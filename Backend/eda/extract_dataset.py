import os
import zipfile
from pathlib import Path
import sys

def combine_files(part_files, output_file):
    if os.path.exists(output_file):
        print(f"{output_file} already exists, skipping combination.")
        return
    print(f"Combining {len(part_files)} files into {output_file}...")
    total_size = sum(os.path.getsize(f) for f in part_files)
    copied = 0
    with open(output_file, 'wb') as outfile:
        for part in part_files:
            print(f"Reading {part}...")
            with open(part, 'rb') as infile:
                while True:
                    chunk = infile.read(1024 * 1024 * 16) # 16MB chunks
                    if not chunk:
                        break
                    outfile.write(chunk)
                    copied += len(chunk)
                    # Print progress every ~1GB
                    if copied % (1024 * 1024 * 1024) < (1024 * 1024 * 16):
                        print(f"Combined {copied / (1024**3):.2f} GB / {total_size / (1024**3):.2f} GB")
    print("Combination complete.")

def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to}...")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.infolist()
        total_files = len(members)
        print(f"Found {total_files} files in {zip_path}.")
        for i, member in enumerate(members):
            zip_ref.extract(member, extract_to)
            if (i + 1) % 1000 == 0:
                print(f"Extracted {i + 1}/{total_files} files...")
    print("Extraction complete.")

if __name__ == "__main__":
    isign_dir = r"F:\dataset\isign"
    part_files = [
        os.path.join(isign_dir, "iSign-videos_v1.1_part_aa"),
        os.path.join(isign_dir, "iSign-videos_v1.1_part_ab")
    ]
    combined_zip = os.path.join(isign_dir, "iSign-videos_v1.1.zip")
    
    # 1. Combine
    combine_files(part_files, combined_zip)
    
    # 2. Extract iSign
    isign_extracted_dir = r"F:\dataset\isign_extracted"
    if not os.path.exists(isign_extracted_dir) or not os.listdir(isign_extracted_dir):
        extract_zip(combined_zip, isign_extracted_dir)
    else:
        print(f"{isign_extracted_dir} already has files, skipping isign extraction.")
        
    # 3. Extract archive.zip
    archive_zip = r"F:\dataset\archive.zip"
    archive_extracted_dir = r"F:\dataset\archive_extracted"
    if not os.path.exists(archive_extracted_dir) or not os.listdir(archive_extracted_dir):
        extract_zip(archive_zip, archive_extracted_dir)
    else:
        print(f"{archive_extracted_dir} already has files, skipping archive extraction.")
    
    print("All tasks finished successfully.")
