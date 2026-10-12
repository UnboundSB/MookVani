import os
import glob
import torch

data_dir = r"d:\MookVani\Backend\data\tensors_include50_word_level"
classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

counts = {}
min_frames_per_class = {}

for cls in classes:
    files = glob.glob(os.path.join(data_dir, cls, "*.pt"))
    counts[cls] = len(files)

min_count_cls = min(counts, key=counts.get)
min_count = counts[min_count_cls]
max_count_cls = max(counts, key=counts.get)
max_count = counts[max_count_cls]

print(f"Total classes: {len(classes)}")
print(f"Min sample count: {min_count} (Class: '{min_count_cls}')")
print(f"Max sample count: {max_count} (Class: '{max_count_cls}')")

# Check a sample of frame lengths
min_t = 999999
min_t_cls = None
for cls in classes[:30]:
    for f in glob.glob(os.path.join(data_dir, cls, "*.pt"))[:3]:
        x = torch.load(f, weights_only=True)
        if x.shape[0] < min_t:
            min_t = x.shape[0]
            min_t_cls = cls

print(f"Sample min frame length T: {min_t} (Class: '{min_t_cls}')")
