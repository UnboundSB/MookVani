"""
yapv3.utils.model — YAP model loading and wrapping for the phase-3 trainer.

Loads a YAP (presence/end segmentation) model from a checkpoint, supports the
checkpoint schema variants found across the codebase ({'yap': ...}, {'model': ...},
raw state_dict), and strips 'module.' / 'yap.' prefixes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from yap_conformer import YAP, YAPConfig


# ===========================================================================
# Checkpoint loading
# ===========================================================================
def _strip_prefix(state: dict, prefix: str) -> dict:
    if any(k.startswith(prefix) for k in state):
        return {k.removeprefix(prefix): v for k, v in state.items()
                if k.startswith(prefix)}
    return state


def load_state_from_checkpoint(path: Path | str) -> dict:
    """Load a YAP state_dict from a checkpoint file.

    Handles:
      * {'yap': state_dict}          (train_segmentor.py save format)
      * {'model': state_dict}        (segmentor_utils.py save format)
      * {'state_dict': state_dict}   (generic PyTorch format)
      * raw state_dict
    Strips 'module.' and 'yap.' prefixes.
    """
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict):
        raise ValueError("Checkpoint is not a state dictionary")
    for key in ("yap", "model", "state_dict"):
        if key in obj and isinstance(obj[key], dict):
            obj = obj[key]
            break
    obj = _strip_prefix(obj, "module.")
    obj = _strip_prefix(obj, "yap.")
    return obj


# ===========================================================================
# Model wrapper
# ===========================================================================
class YAPModelWrapper:
    """Thin wrapper around YAP that exposes forward_frames + segment_batch."""

    def __init__(self, yap: YAP, device: torch.device):
        self.yap = yap.to(device)
        self.device = device
        self.cfg = yap.cfg

    @classmethod
    def from_config(cls, device: torch.device | str = "cuda",
                    d_model: int = 128, n_mem_layers: int = 3,
                    summary_channels: int = 32, hippo_order: int = 16,
                    entity_dim: int = 256) -> "YAPModelWrapper":
        """Build a fresh YAP model with the corpus layout (J=53)."""
        dev = torch.device(device if isinstance(device, str) else device)
        cfg = YAPConfig.corpus(d_model=d_model, n_mem_layers=n_mem_layers,
                               summary_channels=summary_channels,
                               hippo_order=hippo_order, entity_dim=entity_dim)
        yap = YAP(cfg)
        return cls(yap, dev)

    @classmethod
    def from_checkpoint(cls, init_path: Optional[Path | str],
                        device: torch.device | str = "cuda",
                        **cfg_kw) -> "YAPModelWrapper":
        """Build YAP and optionally load pretrained weights from init_path."""
        dev = torch.device(device if isinstance(device, str) else device)
        wrapper = cls.from_config(device=dev, **cfg_kw)
        if init_path is not None:
            path = Path(init_path)
            if not path.exists():
                raise FileNotFoundError(f"Init checkpoint not found: {path}")
            state = load_state_from_checkpoint(path)
            own = wrapper.yap.state_dict()
            compatible = {k: v for k, v in state.items()
                          if k in own and own[k].shape == v.shape}
            if not compatible:
                raise RuntimeError(
                    f"ZERO compatible YAP tensors in init checkpoint {path}")
            missing, unexpected = wrapper.yap.load_state_dict(
                compatible, strict=False)
            print(f"[INIT] loaded {len(compatible)}/{len(own)} tensors from {path}; "
                  f"missing={len(missing)} unexpected={len(unexpected)}")
        return wrapper

    def forward_frames(self, pose, conf, valid, t):
        """Delegate to YAP.forward_frames."""
        return self.yap.forward_frames(pose, conf, valid, t)

    def segment_batch(self, output, times, lengths):
        """Delegate to YAP.segment_batch (production SegmentTracker)."""
        return self.yap.segment_batch(output, times, lengths)

    def state_dict(self):
        return self.yap.state_dict()

    def train(self):
        self.yap.train()

    def eval(self):
        self.yap.eval()

    def parameters(self):
        return self.yap.parameters()


def load_yap_model(init_path: Optional[Path | str] = None,
                   device: torch.device | str = "cuda",
                   **cfg_kw) -> YAPModelWrapper:
    """Convenience: build or load a YAP model."""
    return YAPModelWrapper.from_checkpoint(init_path, device=device, **cfg_kw)
