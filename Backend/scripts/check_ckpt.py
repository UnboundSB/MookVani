import torch

ckpt = torch.load(r"d:\MookVani\Backend\models\word_level\reconformer_word_model_final.pth", map_location="cpu", weights_only=True)
sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
for k, v in sd.items():
    if "cnn.8" in k or "input_dense" in k:
        print(k, v.shape)
