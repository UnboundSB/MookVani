import os
import torch
import numpy as np
from tqdm import tqdm

def generate_base_class():
    """
    Generates a unique base skeletal pose of 163 dimensions.
    Layout: [0:63] lh, [63:126] rh, [126:138] arms, [138:159] face, [159:163] flags
    """
    # 1. Face (7 pts x 3): tight cluster around origin
    face = np.random.normal(0, 0.05, size=(7, 3))
    
    # 2. Arms (4 pts x 3): L/R shoulders and elbows
    # Shoulders fixed around standard normalized width
    shoulders = np.array([[-0.5, 0.5, 0.0], [0.5, 0.5, 0.0]])
    # Elbows are placed randomly in the lower hemisphere
    elbows = np.array([
        [-0.5 + np.random.uniform(-0.5, 0.5), 1.0 + np.random.uniform(-0.5, 0.5), np.random.uniform(-0.3, 0.3)],
        [ 0.5 + np.random.uniform(-0.5, 0.5), 1.0 + np.random.uniform(-0.5, 0.5), np.random.uniform(-0.3, 0.3)]
    ])
    arms = np.vstack([shoulders, elbows])
    
    # 3. Hands (21 pts x 3 each)
    # Give each class a unique hand shape by picking a unique base posture and spreading the 21 points
    lh_center = elbows[0] + np.random.uniform(-0.4, 0.4, size=(3,))
    rh_center = elbows[1] + np.random.uniform(-0.4, 0.4, size=(3,))
    
    # Hand points are spread in a specific random orientation to make the "sign" unique
    lh_spread = np.random.normal(0, 0.15, size=(21, 3))
    rh_spread = np.random.normal(0, 0.15, size=(21, 3))
    
    lh = lh_center + lh_spread
    rh = rh_center + rh_spread
    
    # Flags: all elements present
    flags = np.array([1.0, 1.0, 1.0, 1.0])
    
    base_pose = np.concatenate([
        lh.flatten(), 
        rh.flatten(), 
        arms.flatten(), 
        face.flatten(), 
        flags
    ])
    return base_pose

def main():
    OUTPUT_DIR = "d:/MookVani/Backend/data/synthetic_poses_163"
    NUM_CLASSES = 500
    SAMPLES_PER_CLASS = 20
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Generating {NUM_CLASSES} unique classes with {SAMPLES_PER_CLASS} variations each...")
    
    for c in tqdm(range(NUM_CLASSES), desc="Synthesizing Classes"):
        class_name = f"synth_pose_{c:03d}"
        class_dir = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Each class gets a unique base skeleton
        base_pose = generate_base_class()
        
        for s in range(SAMPLES_PER_CLASS):
            # Create a variation by adding very small gaussian noise to the coordinates
            # This simulates natural human trembling/slight positional variance in the same pose
            noise = np.random.normal(0, 0.015, size=159) # Apply to the 159 coordinates
            
            variation = np.concatenate([
                base_pose[:159] + noise, 
                base_pose[159:] # Keep the 4 flags unchanged
            ])
            
            # Convert to expected tensor shape (1, 163) as float32
            tensor_data = torch.tensor(np.expand_dims(variation, axis=0), dtype=torch.float32)
            
            # Save the .pt file
            save_path = os.path.join(class_dir, f"sample_{s:02d}.pt")
            torch.save(tensor_data, save_path)
            
    print(f"\n✅ Synthetic dataset generation complete! Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
