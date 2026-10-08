import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys

# Ensure backend root is in python path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.isl_conformer import WordCategorizerModel

class SyntheticPoseDataset(Dataset):
    def __init__(self, file_paths, class_to_idx):
        self.file_paths = file_paths
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        # shape is (1, 163)
        tensor = torch.load(path)
        # We need shape (1, 163) for the model which expects (B, T, D)
        # So one sample is (1, 163) where T=1
        
        class_name = os.path.basename(os.path.dirname(path))
        label = self.class_to_idx[class_name]
        return tensor, label

def train_model():
    data_dir = "d:/MookVani/Backend/data/synthetic_poses_163"
    all_files = glob.glob(f"{data_dir}/*/*.pt")
    
    if len(all_files) == 0:
        print("No synthetic data found!")
        return

    # Create class mapping
    class_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_dirs)}
    num_classes = len(class_to_idx)
    
    print(f"Found {len(all_files)} samples across {num_classes} classes.")

    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)
    
    train_dataset = SyntheticPoseDataset(train_files, class_to_idx)
    val_dataset = SyntheticPoseDataset(val_files, class_to_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    model = WordCategorizerModel(num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    
    num_epochs = 10
    best_val_acc = 0.0
    
    os.makedirs("d:/MookVani/Backend/models", exist_ok=True)
    save_path = "d:/MookVani/Backend/models/best_synthetic_model.pth"
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            # inputs is (B, 1, 163)
            logits = model.forward_single_frame_logits(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            
        train_acc = train_correct / len(train_files)
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model.forward_single_frame_logits(inputs)
                loss = criterion(logits, labels)
                
                val_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                
        val_acc = val_correct / len(val_files)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_files):.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss/len(val_files):.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved new best model to {save_path}")
            
if __name__ == "__main__":
    train_model()
