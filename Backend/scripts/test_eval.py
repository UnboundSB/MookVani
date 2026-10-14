import os
import sys
import torch
import numpy as np

BACKEND_DIR = r"d:\MookVani\Backend"
sys.path.insert(0, BACKEND_DIR)
from models.isl_sentence_model import ISLSentenceReconformer
from scripts.dataset import get_balanced_word_dataloaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = os.path.join(BACKEND_DIR, "data", "tensors_include50_word_level")
train_loader, val_loader, class_to_idx, idx_to_class, qualifying_classes = get_balanced_word_dataloaders(
    data_dir=data_dir,
    batch_size=32,
    samples_per_class=14,
    val_samples_per_class=3,
    include_nothing_class=True,
    seed=42
)
num_classes = len(qualifying_classes)

model = ISLSentenceReconformer(
    input_dim=163,
    num_classes=num_classes,
    d_model=128,
    lstm_layers=1,
    trans_layers=4,
    num_heads=4,
    dropout=0.3,
    use_rnn=True,
    freeze_base=False
).to(device)

model_path = os.path.join(BACKEND_DIR, "models", "word_level", "reconformer_word_model.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Check accuracy on train set (without aug)
train_dataset_eval = train_loader.dataset
train_dataset_eval.is_train = False
eval_train_loader = torch.utils.data.DataLoader(train_dataset_eval, batch_size=32, shuffle=False, collate_fn=train_loader.collate_fn)

correct_train = 0
total_train = 0
with torch.no_grad():
    for x, lengths, labels in eval_train_loader:
        x, lengths = x.to(device), lengths.to(device)
        probs_pooled, _ = model.predict_word_probs(x, lengths)
        preds = probs_pooled.argmax(dim=-1).cpu()
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)

print(f"Train Acc (probs_pooled): {correct_train}/{total_train} = {correct_train/total_train*100:.2f}%")

# Check with logits pooling
correct_train_logits = 0
with torch.no_grad():
    for x, lengths, labels in eval_train_loader:
        x, lengths = x.to(device), lengths.to(device)
        logits_seq, _ = model(x, lengths)
        mask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        logits_seq = logits_seq * mask.unsqueeze(-1)
        logits_pooled = logits_seq.sum(dim=1) / lengths.unsqueeze(1).clamp(min=1)
        preds = logits_pooled.argmax(dim=-1).cpu()
        correct_train_logits += (preds == labels).sum().item()

print(f"Train Acc (logits_pooled): {correct_train_logits}/{total_train} = {correct_train_logits/total_train*100:.2f}%")
