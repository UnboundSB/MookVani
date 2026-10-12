import os
import glob
import torch
import numpy as np

data_dir = r"d:\MookVani\Backend\data\tensors_include50_word_level"
classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

sample_counts = {}
frame_lengths = {}
class_avg_frames = {}

print("Scanning all 262 classes...")
for cls in classes:
    files = glob.glob(os.path.join(data_dir, cls, "*.pt"))
    sample_counts[cls] = len(files)
    lens = []
    for f in files:
        x = torch.load(f, weights_only=True)
        lens.append(x.shape[0])
    frame_lengths[cls] = lens
    class_avg_frames[cls] = np.mean(lens) if lens else 0

counts = np.array(list(sample_counts.values()))
all_frames = [l for lens in frame_lengths.values() for l in lens]

classes_ge_15 = [cls for cls, cnt in sample_counts.items() if cnt >= 15]
classes_lt_15 = [cls for cls, cnt in sample_counts.items() if cnt < 15]

print("\n" + "=" * 60)
print("SAMPLE COUNT DISTRIBUTION PER CLASS:")
print("=" * 60)
print(f"Total gesture classes: {len(classes)}")
print(f"Total video samples:   {sum(counts)}")
print(f"Min samples in a class: {min(counts)} (Class: '{min(sample_counts, key=sample_counts.get)}')")
print(f"Max samples in a class: {max(counts)} (Class: '{max(sample_counts, key=sample_counts.get)}')")
print(f"Median samples/class:   {np.median(counts):.1f}")
print(f"Mean samples/class:     {np.mean(counts):.1f}")
print(f"\nClasses with >= 15 samples: {len(classes_ge_15)} ({(len(classes_ge_15)/len(classes))*100:.1f}%)")
print(f"Classes with < 15 samples:  {len(classes_lt_15)} ({(len(classes_lt_15)/len(classes))*100:.1f}%)")

print("\n" + "=" * 60)
print("FRAME LENGTH (TEMPORAL DURATION) STATS:")
print("=" * 60)
print(f"Overall Min frames across all samples: {min(all_frames)}")
print(f"Overall Max frames across all samples: {max(all_frames)}")
print(f"Overall Mean frames:                   {np.mean(all_frames):.1f}")
print(f"Overall Median frames:                 {np.median(all_frames):.1f}")

# Look at classes with lowest and highest average frame lengths
sorted_by_len = sorted(class_avg_frames.items(), key=lambda x: x[1])
print(f"\nShortest classes by avg frames:")
for c, l in sorted_by_len[:5]:
    print(f"  • {c:<20}: avg {l:.1f} frames (sample count: {sample_counts[c]})")

print(f"\nLongest classes by avg frames:")
for c, l in sorted_by_len[-5:]:
    print(f"  • {c:<20}: avg {l:.1f} frames (sample count: {sample_counts[c]})")

# If we filter to classes >= 15:
print("\n" + "=" * 60)
print(f"IF FILTERING TO >= 15 SAMPLES ({len(classes_ge_15)} CLASSES):")
print("=" * 60)
ge_15_frames = [l for c in classes_ge_15 for l in frame_lengths[c]]
print(f"Total samples preserved: {len(ge_15_frames)} of {len(all_frames)} ({len(ge_15_frames)/len(all_frames)*100:.1f}%)")
print(f"Avg frames in >= 15 subset: {np.mean(ge_15_frames):.1f} (Min: {min(ge_15_frames)}, Max: {max(ge_15_frames)})")
print("\nSample list of classes with >= 15 samples:")
print(", ".join(classes_ge_15[:25]) + f", ... (+{len(classes_ge_15)-25} more)")
