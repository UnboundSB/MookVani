"""
Quick preview of synthetic dataset structure (no extraction).
Shows the data shapes and format without running extraction.
"""

import numpy as np
from pathlib import Path

print("=" * 60)
print("SYNTHETIC SEGMENTOR DATASET STRUCTURE PREVIEW")
print("=" * 60)

print("""
The SyntheticSegmentorDataset class generates:

1. SINGLE SAMPLE:
   - words: List[str]  (e.g., ['hello', 'good', 'morning'])
   - x: np.ndarray     (max_seq_len, 163) float32
   - y: np.ndarray     (max_seq_len,) float32  -- boundary labels (1 at word boundaries)

2. BATCH:
   - batch_x: (batch_size, max_seq_len, 163) float32
   - batch_y: (batch_size, max_seq_len) float32
   - batch_words: List[List[str]]

3. FEATURE LAYOUT (163 dims):
   [0:63]      Left Hand  (21 landmarks x 3 = x,y,z)
   [63:126]    Right Hand (21 landmarks x 3)
   [126:138]   Arms       (4 landmarks x 3 = shoulders, elbows)
   [138:159]   Face       (7 landmarks x 3 = eyes, nose, cheeks, mouth)
   [159:163]   Flags      [lh_flag, rh_flag, arms_flag, face_flag]

4. BOUNDARY LABELS:
   - y[t] = 1.0 at word boundary frames
   - y[t] = 0.0 elsewhere
   - Perfect ground truth from stitch points

5. GENERATION PIPELINE:
   a) Pick 2-5 random words from 228 vocab (weighted by frequency)
   b) For each word: load .npy from conformer_features/ or extract from archive
   c) Find motion onset + 5 prep frames -> trim static prefix
   d) Gaussian time-warp (sigma=0.2) per word
   e) Stitch words -> synthetic sentence
   f) Boundary labels = stitch frame indices

6. OUTPUT SHAPES:
   Single:  x=(512, 163), y=(512,)
   Batch:   x=(32, 512, 163), y=(32, 512)

7. DATA SOURCES (in priority):
   1) conformer_features/word/*.npy  (from video_recorder)
   2) archive_extracted/ + MediaPipe  (fallback, slower first run)

8. AUGMENTATION:
   - Gaussian time-warp (sigma=0.2)
   - Prep frames (5 after motion onset)
   - Random word selection (inverse frequency weighting)
   - Unique words per sentence (no repeats)

USAGE:
    ds = SyntheticSegmentorDataset(max_seq_len=512, sigma=0.2)
    
    # Single sample
    x, y, words = ds._generate_one()
    
    # Training batches
    for batch_x, batch_y, batch_words in ds.batch_generator(batch_size=32):
        # batch_x: (32, 512, 163)
        # batch_y: (32, 512)  -- boundary targets
        model.train_on_batch(batch_x, batch_y)
""")

print("\n" + "=" * 60)
print("KEY CLASSES TO USE IN TRAINING PIPELINE")
print("=" * 60)

print("""
# In your training script:
from synthetic.dataset import SyntheticSegmentorDataset

# Create dataset
train_ds = SyntheticSegmentorDataset(
    max_seq_len=512,
    sigma=0.2,           # time-warp amount
    prep_frames=5,       # frames after motion onset
    use_archive_fallback=True
)

# Train loop
for epoch in range(epochs):
    for batch_x, batch_y, batch_words in train_ds.batch_generator(batch_size=32):
        # batch_x: (32, 512, 163) - input features
        # batch_y: (32, 512)       - boundary targets (1.0 at boundaries)
        
        loss = model.train_on_batch(batch_x, batch_y)
        # ...
""")

print("\n" + "=" * 60)
print("COMPLETE PIPELINE OVERVIEW")
print("=" * 60)

print("""
+------------------+    +------------------+    +------------------+
| video_recorder   |--> | conformer_       |    |                  |
| (webcam + MP)    |    | features/word/   |    |                  |
+------------------+    +--------+---------+    |                  |
                                |              |                  |
                                v              v                  |
                    +--------------------------+                |
                    | synthetic.generator      |                |
                    | - generate_synthetic     |                |
                    | - time_warp + prep       |                |
                    | - perfect boundaries     |                |
                    +------------+-------------+                |
                                 |                             |
                                 v                             |
                    +--------------------------+                |
                    | synthetic.dataset        |<-- TRAIN HERE  |
                    | SyntheticSegmentorDataset|                |
                    | - batch_generator()      |                |
                    | - (x, y, words)          |                |
                    +--------------------------+----------------+
""")

print("\nAll components ready in: backend/segmentor/data_builder/synthetic/")