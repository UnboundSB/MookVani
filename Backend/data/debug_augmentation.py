import cv2
import glob
import random
import numpy as np
import os

def get_random_geometric_matrix(w, h, max_shift_frac=0.06, max_rotation_deg=8, allow_flip=True):
    do_flip = allow_flip and random.random() < 0.5
    angle = random.uniform(-max_rotation_deg, max_rotation_deg)
    shift_x = random.uniform(-max_shift_frac, max_shift_frac) * w
    shift_y = random.uniform(-max_shift_frac, max_shift_frac) * h

    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rot_mat[0, 2] += shift_x
    rot_mat[1, 2] += shift_y

    return do_flip, rot_mat

def apply_geometric_matrix(img, do_flip, rot_mat):
    h, w = img.shape[:2]
    out = img.copy()
    if do_flip:
        out = cv2.flip(out, 1)
    out = cv2.warpAffine(out, rot_mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return out

raw_dir = "d:/MookVani/Backend/data/isl_csltr_dataset/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/Frames_Sentence_Level"
classes = [d for d in glob.glob(f"{raw_dir}/*") if os.path.isdir(d)]
# Pick a random class and sample
random_class = random.choice(classes)
samples = [d for d in glob.glob(f"{random_class}/*") if os.path.isdir(d)]
random_sample = random.choice(samples)

frames = sorted(glob.glob(f"{random_sample}/*.jpg"))
if not frames:
    print("No frames found!")
    exit()

# Load first 5 frames to visualize
visualize_frames = frames[:5]

test_img = cv2.imread(visualize_frames[0])
h, w = test_img.shape[:2]

# Generate ONE random matrix for the whole sequence
do_flip, rot_mat = get_random_geometric_matrix(w, h, allow_flip=True)

# Create a visualization grid
grid_rows = []
for f_path in visualize_frames:
    img = cv2.imread(f_path)
    aug_img = apply_geometric_matrix(img, do_flip, rot_mat)
    
    # Resize for visualization
    img = cv2.resize(img, (w//2, h//2))
    aug_img = cv2.resize(aug_img, (w//2, h//2))
    
    # Put text
    cv2.putText(img, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(aug_img, "Augmented", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Concat side by side
    row = np.hstack([img, aug_img])
    grid_rows.append(row)

# Concat top to bottom
final_grid = np.vstack(grid_rows)
save_path = "d:/MookVani/Backend/data/aug_visualization.jpg"
cv2.imwrite(save_path, final_grid)

print(f"Visualization saved to {save_path}")
print(f"Flip applied: {do_flip}")
print(f"Rotation Matrix:\n{rot_mat}")
