import os
import zipfile
import sys

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
    combined_zip = os.path.join(isign_dir, "iSign-videos_v1.1.zip")
    isign_extracted_dir = r"F:\dataset\isign_extracted"
    
    extract_zip(combined_zip, isign_extracted_dir)
    print("iSign extraction finished successfully.")
