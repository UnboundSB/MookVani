"""
Loss Functions for YAP Segmentor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


def temporal_smooth_loss(end_logit: torch.Tensor, lambda_smooth: float = 0.01) -> torch.Tensor:
    """Temporal smoothing loss - penalizes rapid changes in end probability."""
    end_prob = torch.sigmoid(end_logit)
    diff = end_prob[:, 1:] - end_prob[:, :-1]
    return lambda_smooth * (diff ** 2).mean()


def boundary_nms_penalty(end_logit: torch.Tensor, min_distance: int = 5, penalty: float = 0.01) -> torch.Tensor:
    """Gentle penalty for boundaries that are too close - only activates when boundaries are too close."""
    end_prob = torch.sigmoid(end_logit)
    T = end_logit.shape[1]
    
    # Pad and find peaks
    end_prob_padded = F.pad(end_prob, (1, 1), mode='constant', value=0)
    left = end_prob_padded[:, :-2]
    center = end_prob_padded[:, 1:-1]
    right = end_prob_padded[:, 2:]
    is_peak = (center > left) & (center > right)
    
    # Find peaks
    peak_indices = is_peak.nonzero(as_tuple=True)
    if len(peak_indices[0]) <= 1:
        return torch.tensor(0.0, device=end_logit.device)
    
    batch_indices, peak_positions = peak_indices
    penalty_loss = torch.tensor(0.0, device=end_logit.device)
    
    for b in range(end_logit.shape[0]):
        batch_peaks = peak_positions[batch_indices == b]
        if len(batch_peaks) > 1:
            sorted_peaks = batch_peaks.sort().values
            distances = sorted_peaks[1:] - sorted_peaks[:-1]
            too_close = distances < 5
            if too_close.any():
                penalty_loss = (5 - distances[too_close].float()).mean()
                return penalty_loss * 0.01
    
    return torch.tensor(0.0, device=end_logit.device)


def temporal_smooth_loss(end_logit: torch.Tensor, lambda_smooth: float = 0.01) -> torch.Tensor:
    """Temporal smoothing loss - penalizes rapid changes in end probability."""
    end_prob = torch.sigmoid(end_logit)
    diff = end_prob[:, 1:] - end_prob[:, :-1]
    return lambda_smooth * (diff ** 2).mean()


def compute_losses(
    model_out: Dict,
    targets: Dict,
    epoch: int = 0,
    max_epochs: int = 50
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute all losses for YAP segmentor.
    
    Args:
        model_out: Dict with 'presence_logit' (B, T) and 'end_logit' (B, T)
        targets: Dict with 'presence' (B, T), 'end_target' (B, T), 'lengths' (B,)
        epoch: Current epoch (0-indexed)
        max_epochs: Total epochs
    
    Returns:
        total_loss, loss_dict
    """
    from .config import cfg
    
    presence_logit = model_out['presence_logit']  # (B, T)
    end_logit = model_out['end_logit']            # (B, T)
    
    presence_target = targets['presence']      # (B, T)
    end_target = targets['end_target']         # (B, T)
    
    lengths = targets['lengths']               # (B,)
    B, T = presence_logit.shape
    
    # Check for NaN in logits
    if torch.isnan(presence_logit).any() or torch.isnan(end_logit).any():
        print("[WARNING] NaN detected in logits!")
        presence_logit = torch.nan_to_num(presence_logit, nan=0.0, posinf=10.0, neginf=-10.0)
        end_logit = torch.nan_to_num(end_logit, nan=0.0, posinf=10.0, neginf=-10.0)
    
    # Create mask for valid frames
    mask = torch.arange(T, device=presence_logit.device).unsqueeze(0) < lengths.unsqueeze(1)
    
    # Dynamic weights (ramp up over epochs)
    seg_weight = min(0.05 * (epoch / 20), 0.05)
    smooth_weight = min(0.02 * (epoch / 10), 0.02)
    
    # 1. Presence loss (blank class: 0=idle, 1=signing) - BCE with logits
    presence_loss = F.binary_cross_entropy_with_logits(
        presence_logit, targets['presence'],
        pos_weight=torch.tensor([2.0], device=presence_logit.device),
        reduction='none'
    )
    presence_loss = (presence_loss * mask).sum() / mask.sum().clamp(min=1)
    
    # 2. End boundary loss (weighted BCE)
    end_loss = F.binary_cross_entropy_with_logits(
        end_logit, targets['end_target'],
        pos_weight=torch.tensor([10.0], device=end_logit.device),
        reduction='none'
    )
    end_loss = (end_loss * mask).sum() / mask.sum().clamp(min=1)
    
    # Temporal smoothing
    smooth_loss = temporal_smooth_loss(end_logit, lambda_smooth=smooth_weight)
    
    # Boundary NMS penalty
    seg_penalty = boundary_nms_penalty(end_logit, min_distance=5, penalty=seg_weight)
    
    # Check for NaN in individual losses
    if torch.isnan(presence_loss) or torch.isnan(end_loss):
        print("[WARNING] NaN detected in losses!")
        presence_loss = torch.clamp(presence_loss, 0.0, 100.0)
        end_loss = torch.clamp(end_loss, 0.0, 100.0)
    
    if torch.isnan(smooth_loss):
        smooth_loss = torch.tensor(0.0, device=smooth_loss.device)
    
    if torch.isnan(seg_penalty):
        seg_penalty = torch.tensor(0.0, device=seg_penalty.device)
    
    total_loss = presence_loss + end_loss + smooth_weight * smooth_loss + seg_penalty
    
    # Clamp total loss to prevent NaN
    total_loss = torch.clamp(total_loss, min=0.0, max=100.0)
    
    return total_loss, {
        'total': total_loss.item() if torch.isfinite(total_loss) else 0.0,
        'presence': presence_loss.item() if torch.isfinite(presence_loss) else 0.0,
        'end': end_loss.item() if torch.isfinite(end_loss) else 0.0,
        'smooth': smooth_loss.item() if isinstance(smooth_loss, torch.Tensor) and torch.isfinite(smooth_loss) else 0.0,
        'seg_penalty': seg_penalty.item() if isinstance(seg_penalty, torch.Tensor) and torch.isfinite(seg_penalty) else 0.0,
    }