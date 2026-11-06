"""
yapv3.utils.losses — relevant losses for YAP gesture-segmentation pretraining.

All losses operate on the output of YAP.forward_frames() which returns a dict
with keys: presence, end, layer_conf, speed, quality, clean, feat.

The supervision signal is:
  * presence  — soft target (1 inside a word span, ramped to 0 outside)
  * end       — Gaussian bump at each word's end time (integrates to 1 per word)
  * layer_conf — deep supervision: every layer's confidence mirrors presence
  * count     — weak supervision: sum of sigmoid(end) == number of words

No CTC, no Conformer, no word-classifier head — this is pure gesture-boundary
pretraining (Stage 2 / Phase A style).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from yap_conformer import (presence_targets, end_targets, focal_presence_loss,
                           layer_conf_loss, soft_count_loss)


# ===========================================================================
# Presence loss — "is a gesture happening right now?"
# ===========================================================================
def presence_loss(output: dict, target_presence: torch.Tensor,
                  mask: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """Focal BCE with soft targets. Down-weights easy 'nothing happening' frames."""
    return focal_presence_loss(output["presence"], target_presence, mask, gamma=gamma)


# ===========================================================================
# End loss — "does a gesture end right here?"
# ===========================================================================
def end_bce_loss(output: dict, target_end: torch.Tensor,
                 mask: torch.Tensor, pos_weight: float = 10.0) -> torch.Tensor:
    """Masked weighted BCE on the end head.

    pos_weight > 1 because end-events are rare (only ~1 frame per word).
    """
    end_raw = F.binary_cross_entropy_with_logits(
        output["end"], target_end, reduction="none",
        pos_weight=torch.tensor(pos_weight, device=output["end"].device))
    return (end_raw * mask.float()).sum() / mask.float().sum().clamp_min(1)


# ===========================================================================
# Layer confidence loss — deep supervision
# ===========================================================================
def layer_conf_loss_fn(output: dict, target_presence: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:
    """Every layer's confidence should also predict presence (deep supervision)."""
    return layer_conf_loss(output["layer_conf"], target_presence, mask)


# ===========================================================================
# Soft count loss — weak supervision on gesture count
# ===========================================================================
def soft_count_loss_fn(output: dict, n_words: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:
    """Differentiable count: sum of sigmoid(end) over valid frames == n_words."""
    return soft_count_loss(output["end"], n_words, mask)


# ===========================================================================
# Combined loss
# ===========================================================================
def total_segmentation_loss(output: dict, presence_tgt: torch.Tensor,
                            end_tgt: torch.Tensor, counts: torch.Tensor,
                            mask: torch.Tensor,
                            w_presence: float = 1.0, w_end: float = 1.0,
                            w_layer: float = 0.5, w_count: float = 0.5,
                            end_pos_weight: float = 10.0) -> tuple[torch.Tensor, dict]:
    """Combined segmentation loss with all components logged.

    Returns (total, parts) where parts is a dict of detached floats — safe
    for history logging (NaN components raise before being written).
    """
    pres = presence_loss(output, presence_tgt, mask)
    end = end_bce_loss(output, end_tgt, mask, pos_weight=end_pos_weight)
    layer = layer_conf_loss_fn(output, presence_tgt, mask)
    cnt = soft_count_loss_fn(output, counts, mask)

    total = w_presence * pres + w_end * end + w_layer * layer + w_count * cnt

    # --- NaN safeguard (P0-2): never log non-finite components silently ---
    for name, val in (("presence", pres), ("end", end), ("layer", layer),
                      ("count", cnt), ("total", total)):
        if not torch.isfinite(val):
            raise FloatingPointError(
                f'Nonfinite loss component "{name}" = {float(val)}. '
                'Stopping before NaN gets written to history.')

    parts = {
        "presence": pres.detach().item(),
        "end": end.detach().item(),
        "layer": layer.detach().item(),
        "count": cnt.detach().item(),
    }
    return total, parts


# ===========================================================================
# Target builders
# ===========================================================================
def build_presence_targets(times: torch.Tensor,
                           spans: list[tuple[float, float]],
                           ramp_s: float = 0.1) -> torch.Tensor:
    """Soft per-pose presence target (T,) for one sequence."""
    return presence_targets(times, spans, ramp_s=ramp_s)


def build_end_targets(times: torch.Tensor,
                      spans: list[tuple[float, float]],
                      sigma_s: float = 0.05) -> torch.Tensor:
    """Gaussian-bump end target (T,) for one sequence."""
    return end_targets(times, spans, sigma_s=sigma_s, normalize=True)
