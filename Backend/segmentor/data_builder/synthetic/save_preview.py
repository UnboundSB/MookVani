"""
Save preview frames of synthetic sentence skeleton to image files.
Run this to see what the skeleton looks like without interactive window.
"""
import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from synthetic.generator import generate_synthetic_sentence, extract_keypoints_2d, draw_skeleton, load_word_inventory

# Generate a sentence
words = ["hello", "good", "morning"]
sentence, boundaries = generate_synthetic_sentence(words)
print(f"Generated: {' '.join(words)}")
print(f"Frames: {len(sentence)}, Boundaries: {boundaries}")

# Create output directory
out_dir = Path("synthetic_preview")
out_dir.mkdir(exist_ok=True)

# Save key frames
key_frames = [0] + boundaries + [len(sentence) - 1]
for idx in key_frames:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    try:
        kpts = extract_keypoints_2d(sentence, idx)
        draw_skeleton(frame, kpts, (0, 255, 0))
    except Exception as e:
        cv2.putText(frame, f"Viz Error: {e}", (30, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Labels
    word_idx = 0
    for j, b in enumerate(boundaries):
        if idx >= b:
            word_idx = j
    current_word = words[word_idx] if word_idx < len(words) else ""
    
    cv2.putText(frame, f"Word: {current_word}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(frame, f"Sentence: {' '.join(words)}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Frame: {idx}/{len(sentence)}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    if idx in boundaries:
        cv2.rectangle(frame, (0, 0), (1280, 720), (0, 0, 255), 3)
        cv2.putText(frame, "BOUNDARY", (500, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    # Save
    fname = out_dir / f"frame_{idx:04d}_word_{current_word}.jpg"
    cv2.imwrite(str(fname), frame)
    print(f"Saved {fname}")

print(f"\nPreview images saved to {out_dir}/")