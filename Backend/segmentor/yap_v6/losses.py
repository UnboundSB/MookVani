"""
Loss Functions for YAP Segmentor v6 - Anti-Collapse Pipeline.

Losses:
  - Focal-ish BCE presence (pos_weight=2.0)
  - Weighted BCE end (pos_weight=10.0)
  - Temporal smoothing (lambda=0.02)
  - Boundary NMS penalty (weight=0.1, min_dist=5)
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple

from .config import cfg


def temporal_smooth_loss(end_logit: torch.Tensor, lambda_smooth: float = 0.02) -> torch.Tensor:
    """Temporal smoothing loss - penalizes rapid changes in end probability."""
    end_prob = torch.sigmoid(end_logit)
    diff = end_prob[:, 1:] - end_prob[:, :-1]
    return lambda_smooth * (diff ** 2).mean()


def boundary_nms_penalty(end_logit: torch.Tensor, min_distance: int = 5,
                         penalty_weight: float = 0.1) -> torch.Tensor:
    """Penalty for boundaries that are too close together (closer than min_distance)."""
    end_prob = torch.sigmoid(end_logit)
    B, T = end_logit.shape

    # Pad and find local peaks
    end_prob_padded = F.pad(end_prob, (1, 1), mode='constant', value=0.0)
    left = end_prob_padded[:, :-2]
    center = end_prob_padded[:, 1:-1]
    right = end_prob_padded[:, 2:]
    is_peak = (center > left) & (center > right)

    peak_indices = is_peak.nonzero(as_tuple=True)
    if len(peak_indices[0]) <= 1:
        return torch.tensor(0.0, device=end_logit.device)

    batch_indices, peak_positions = peak_indices
    penalty_loss = torch.tensor(0.0, device=end_logit.device)
    any_too_close = False

    for b in range(B):
        batch_peaks = peak_positions[batch_indices == b]
        if len(batch_peaks) > 1:
            sorted_peaks = batch_peaks.sort().values
            distances = sorted_peaks[1:] - sorted_peaks[:-1]
            too_close = distances < min_distance
            if too_close.any():
                penalty_loss = penalty_loss + (min_distance - distances[too_close].float()).mean()
                any_too_close = True

    if not any_too_close:
        return torch.tensor(0.0, device=end_logit.device)
    return penalty_weight * penalty_loss / max(B, 1)


def compute_losses(model_out: Dict, targets: Dict, epoch: int = 0,
                   max_epochs: int = 50) -> Tuple[torch.Tensor, Dict]:
    """
    Compute all losses for YAP segmentor v6.

    Args:
        model_out: Dict with 'presence_logit' (B, T) and 'end_logit' (B, T)
        targets: Dict with 'presence' (B, T), 'end_target' (B, T), 'lengths' (B,)
        epoch: Current epoch (0-indexed)
        max_epochs: Total epochs

    Returns:
        total_loss, loss_dict
    """
    presence_logit = model_out['presence_logit']   # (B, T)
    end_logit = model_out['end_logit']             # (B, T)

    presence_target = targets['presence']          # (B, T)
    end_target = targets['end_target']             # (B, T)
    lengths = targets['lengths']                   # (B,)

    B, T = presence_logit.shape

    # Guard against NaN in logits
    if torch.isnan(presence_logit).any() or torch.isnan(end_logit).any():
        print("[WARNING] NaN detected in logits!")
        presence_logit = torch.nan_to_num(presence_logit, nan=0.0, posinf=10.0, neginf=-10.0)
        end_logit = torch.nan_to_num(end_logit, nan=0.0, posinf=10.0, neginf=-10.0)

    # Mask for valid frames (ignore padding)
    mask = torch.arange(T, device=presence_logit.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.float()

    # Dynamic weights (ramp up over epochs)
    smooth_weight = min(0.02 * (epoch / 10.0), 0.02)
    seg_weight = min(0.1 * (epoch / 20.0), 0.1)

    # 1. Presence loss - BCE with logits, pos_weight=2.0
    presence_loss = F.binary_cross_entropy_with_logits(
        presence_logit, presence_target,
        pos_weight=torch.tensor([cfg.PRESENCE_POS_WEIGHT], device=presence_logit.device),
        reduction='none',
    )
    presence_loss = (presence_loss * mask).sum() / mask.sum().clamp(min=1)

    # 2. End boundary loss - weighted BCE, pos_weight=10.0
    end_loss = F.binary_cross_entropy_with_logits(
        end_logit, end_target,
        pos_weight=torch.tensor([cfg.END_POS_WEIGHT], device=end_logit.device),
        reduction='none',
    )
    end_loss = (end_loss * mask).sum() / mask.sum().clamp(min=1)

    # Temporal smoothing
    smooth_loss = temporal_smooth_loss(end_logit, lambda_smooth=smooth_weight)

    # Boundary NMS penalty
    seg_penalty = boundary_nms_penalty(end_logit, min_distance=cfg.BOUNDARY_MIN_DIST,
                                       penalty_weight=seg_weight)

    # NaN guards
    if not torch.isfinite(presence_loss):
        presence_loss = torch.tensor(0.0, device=presence_logit.device)
    if not torch.isfinite(end_loss):
        end_loss = torch.tensor(0.0, device=end_logit.device)
    if not torch.isfinite(smooth_loss):
        smooth_loss = torch.tensor(0.0, device=end_logit.device)
    if not torch.isfinite(seg_penalty):
        seg_penalty = torch.tensor(0.0, device=end_logit.device)

    total_loss = presence_loss + end_loss + smooth_weight * smooth_loss + seg_penalty
    total_loss = torch.clamp(total_loss, min=0.0, max=100.0)

    return total_loss, {
        'total': total_loss.item() if torch.isfinite(total_loss) else 0.0,
        'presence': presence_loss.item() if torch.isfinite(presence_loss) else 0.0,
        'end': end_loss.item() if torch.isfinite(end_loss) else 0.0,
        'smooth': smooth_loss.item() if torch.isfinite(smooth_loss) else 0.0,
        'seg_penalty': seg_penalty.item() if torch.isfinite(seg_penalty) else 0.0,
    }
