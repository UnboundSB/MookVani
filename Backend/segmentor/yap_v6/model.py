"""
Model setup for YAP Segmentor v6.
YAP backbone (2 hands = 42 joints) is trained; Conformer head is frozen.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from yap_conformer import YAPConfig, YAPpingConformer, ConformerConfig, hand_part

import torch
import torch.nn as nn
from typing import Optional

from .config import cfg


def create_model(pretrained_path: Optional[Path] = None) -> nn.Module:
    """
    Create YAP model for segmentor fine-tuning.

    Returns:
        model: YAPpingConformer with frozen Conformer blocks + head;
               only the YAP backbone requires grad.
    """
    model = YAPpingConformer(
        YAPConfig(parts=[hand_part("left_hand"), hand_part("right_hand")]),
        ConformerConfig(num_words=228),
    )

    # Load pretrained segmentor weights (backbone initialization)
    if pretrained_path is None:
        pretrained_path = Path(r"D:\MookVani\Backend\models\yap_segmentor\yap_segmentor_final.pth")
    if pretrained_path.exists():
        state = torch.load(pretrained_path, map_location="cpu")
        # Be tolerant of key prefixes ('model.' etc.)
        new_state = {}
        for k, v in state.items():
            nk = k[len("model."):] if k.startswith("model.") else k
            new_state[nk] = v
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        print(f"[INFO] Loaded pretrained segmentor weights from {pretrained_path}")
        if missing:
            print(f"[INFO] Missing keys: {missing[:10]}")
        if unexpected:
            print(f"[INFO] Unexpected keys: {unexpected[:10]}")
    else:
        print(f"[WARN] Pretrained weights not found at {pretrained_path}; training from scratch")

    # Freeze Conformer blocks + head; train only YAP backbone
    for param in model.blocks.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = False
    for param in model.yap.parameters():
        param.requires_grad = True

    print(f"[INFO] Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return model


class EMA:
    """Exponential Moving Average for teacher model (Mean Teacher)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


def count_parameters(model) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
