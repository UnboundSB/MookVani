import os
import time
from tqdm import tqdm
import glob

output_dir = "d:/MookVani/Backend/data/tensors_include50_word_level"
total_videos = 4285

print("Monitoring background extraction progress...")
print("Press Ctrl+C to exit this monitor (it won't stop the background extraction).")

# Initialize tqdm
with tqdm(total=total_videos, desc="Extraction Progress", unit="vid") as pbar:
    last_count = 0
    while True:
        try:
            # Count current extracted files
            current_count = 0
            if os.path.exists(output_dir):
                current_count = sum(1 for _ in glob.iglob(f"{output_dir}/*/*.pt"))
                
            # Update progress bar if there are new files
            if current_count > last_count:
                pbar.update(current_count - last_count)
                last_count = current_count
                
            if current_count >= total_videos:
                print("\n✅ Extraction complete!")
                break
                
            time.sleep(2) # check every 2 seconds
        except KeyboardInterrupt:
            print("\nExiting monitor.")
            break
