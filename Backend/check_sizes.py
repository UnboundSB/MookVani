import torch

ckpt = torch.load('models/best_word_model.pth', map_location='cpu', weights_only=True)
if 'model_state_dict' in ckpt:
    ckpt = ckpt['model_state_dict']

for k, v in ckpt.items():
    if 'classifier' in k:
        print(f"{k}: {v.shape}")
