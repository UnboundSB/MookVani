"""
Model setup for YAP Segmentor
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from yap_conformer import YAPConfig, YAPpingConformer, ConformerConfig, hand_part

import torch
import torch.nn as nn
from typing import Optional

from .config import cfg


def create_model(pretrained_path: Optional[Path] = None) -> tuple:
    """
    Create YAP model for segmentor fine-tuning.
    
    Returns:
        model: YAPpingConformer with frozen Conformer head
        ema: EMA wrapper for teacher model
    """
    # Create YAP model (2 hands only = 42 joints)
    cfg = YAPConfig(parts=[hand_part("left_hand"), hand_part("right_hand")])
    conformer_cfg = ConformerConfig(num_words=228)
    model = YAPpingConformer(
        YAPConfig(parts=[hand_part("left_hand"), hand_part("right_hand")]),
        ConformerConfig(num_words=228)
    )
    
    # Load pretrained segmentor weights
    ckpt_path = Path(r"D:\MookVani\Backend\models\yap_segmentor\yap_segmentor_final.pth")
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
        print(f"[INFO] Loaded pretrained segmentor weights from {ckpt_path}")
    
    # Freeze Conformer blocks + head, train only YAP backbone
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


def count_parameters(model) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)