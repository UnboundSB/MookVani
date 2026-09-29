import torch
import torch.nn.functional as F
import math
import glob
import os
from tqdm import tqdm

def augment_tensor(tensor, angle_deg=0.0, shift_x=0.0, shift_y=0.0, speed_factor=1.0):
    """
    Augments a (Sequence_Length, 163) MediaPipe tensor DIRECTLY.
    - angle_deg: Rotation angle in degrees (tilting)
    - shift_x, shift_y: Translation (shift)
    - speed_factor: < 1.0 (slows down), > 1.0 (speeds up)
    """
    seq_len, dim = tensor.shape
    assert dim == 163, "Must be the 163-dim MediaPipe format"
    
    # 1. Temporal Interpolation (Speed Up / Slow Down)
    if speed_factor != 1.0:
        new_len = max(1, int(seq_len / speed_factor))
        # Reshape to (Batch=1, Channels=163, SeqLen) for 1D interpolation
        t_seq = tensor.T.unsqueeze(0) 
        t_seq = F.interpolate(t_seq, size=new_len, mode='linear', align_corners=False)
        tensor = t_seq.squeeze(0).T # Back to (new_len, 163)
        
    # 2. Geometric Transformations (Rotation & Shift)
    # MediaPipe coordinates are normalized [0, 1]. Center of rotation is usually (0.5, 0.5)
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    out_tensor = tensor.clone()
    
    # Iterate over every (X, Y) pair. 
    # Indices 0-158 contain the (X, Y, Z) coordinates (159 is flags)
    # X is 0, 3, 6... Y is 1, 4, 7...
    for i in range(0, 159, 3):
        # Subtract center
        x = tensor[:, i] - 0.5
        y = tensor[:, i+1] - 0.5
        
        # Rotate
        new_x = x * cos_a - y * sin_a
        new_y = x * sin_a + y * cos_a
        
        # Add center back + shift
        out_tensor[:, i] = new_x + 0.5 + shift_x
        out_tensor[:, i+1] = new_y + 0.5 + shift_y
        
    return out_tensor

def process_dir(data_dir):
    files = glob.glob(os.path.join(data_dir, "**", "*.pt"), recursive=True)
    print(f"\nProcessing {data_dir}: Found {len(files)} tensors to augment.")
    
    # We don't want to re-augment already augmented files if run multiple times
    files = [f for f in files if "aug" not in os.path.basename(f)]
    
    for fpath in tqdm(files):
        fname = os.path.basename(fpath)
        tensor = torch.load(fpath)
        
        # Original directory
        file_dir = os.path.dirname(fpath)
        
        # Augment 1: Tilt right + Speed up
        aug1 = augment_tensor(tensor, angle_deg=5.0, shift_x=0.02, speed_factor=1.2)
        torch.save(aug1, os.path.join(file_dir, f"aug1_{fname}"))
        
        # Augment 2: Tilt left + Slow down
        aug2 = augment_tensor(tensor, angle_deg=-5.0, shift_y=-0.02, speed_factor=0.8)
        torch.save(aug2, os.path.join(file_dir, f"aug2_{fname}"))
        
        # Augment 3: Minor scale/shift
        aug3 = augment_tensor(tensor, angle_deg=0.0, shift_x=-0.03, shift_y=0.01, speed_factor=1.0)
        torch.save(aug3, os.path.join(file_dir, f"aug3_{fname}"))

def main():
    word_dir = r"D:\MookVani\Backend\data\tensors_word_level_163_train"
    sentence_dir = r"D:\MookVani\Backend\data\tensors_sentence_level_163_train"
    
    process_dir(word_dir)
    process_dir(sentence_dir)
    
    print("\nDataset successfully multiplied by 4x using direct tensor augmentation!")

if __name__ == "__main__":
    main()
