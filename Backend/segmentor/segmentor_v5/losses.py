"""
Loss functions for YAP segmentor training.

Head targets:
    presence_logit / presence_target  -> is a sign happening? (blank vs signing)
    end_logit       / end_target      -> does a gesture end at this frame?
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


# ------------------------------------------------------------------ presence
def focal_bce_loss(logits: torch.Tensor, targets: torch.Tensor,
                   alpha: float = 0.25, gamma: float = 2.0,
                   pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    pt = prob * targets + (1 - prob) * (1 - targets)
    focal_weight = alpha * targets + (1 - alpha) * (1 - targets)
    loss = focal_weight * (1 - pt) ** gamma * ce
    return loss.mean()


def bce_with_logits_loss(logits: torch.Tensor, targets: torch.Tensor,
                         pos_weight: Optional[torch.Tensor] = None,
                         label_smoothing: float = 0.0) -> torch.Tensor:
    if label_smoothing > 0:
        targets = targets * (1 - label_smoothing) + 0.5 * label_smoothing
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


# ------------------------------------------------------------------ temporal / boundary
def temporal_smooth_loss(end_logit: torch.Tensor,
                         lambda_smooth: float = 0.01) -> torch.Tensor:
    end_prob = torch.sigmoid(end_logit)
    diff = end_prob[:, 1:] - end_prob[:, :-1]
    return lambda_smooth * (diff ** 2).mean()


def boundary_nms_penalty(end_logit: torch.Tensor,
                         min_distance: int = 5,
                         penalty: float = 0.01) -> torch.Tensor:
    """Gentle penalty for predicted boundaries spaced closer than min_distance."""
    end_prob = torch.sigmoid(end_logit)
    T = end_logit.shape[1]

    end_prob_padded = F.pad(end_prob, (1, 1), mode="constant", value=0)
    left = end_prob_padded[:, :-2]
    center = end_prob_padded[:, 1:-1]
    right = end_prob_padded[:, 2:]
    is_peak = (center > left) & (center > right)

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
            too_close = distances < min_distance
            if too_close.any():
                penalty_loss = (min_distance - distances[too_close].float()).mean()
                return penalty_loss * penalty

    return torch.tensor(0.0, device=end_logit.device)


# ------------------------------------------------------------------ combined
def compute_losses(model_out: Dict, targets: Dict,
                   epoch: int = 0, max_epochs: int = 50,
                   DEVICE: "torch.device | None" = None,
                   PRESENCE_POS_WEIGHT: float = 2.0,
                   END_POS_WEIGHT: float = 10.0) -> Tuple[torch.Tensor, Dict]:
    """Combined presence + end + smoothing + boundary-NMS loss.

    Returns (total_loss, loss_dict).
    """
    presence_logit = model_out["presence_logit"]      # (B, T)
    end_logit = model_out["end_logit"]                # (B, T)

    presence_target = targets["presence"]
    end_target = targets["end_target"]

    lengths = targets["lengths"]
    B, T = presence_logit.shape
    mask = torch.arange(T, device=presence_logit.device).unsqueeze(0) < lengths.unsqueeze(1)

    # Dynamic weights ramped over epochs.
    smooth_weight = min(0.02 * (epoch / 10), 0.02)
    seg_weight = min(0.05 * (epoch / 20), 0.05)

    # Presence loss (masked mean over valid frames).
    presence_loss = F.binary_cross_entropy_with_logits(
        presence_logit[mask], presence_target[mask],
        pos_weight=torch.tensor([PRESENCE_POS_WEIGHT], device=presence_logit.device),
    )

    # End boundary loss.
    end_loss = F.binary_cross_entropy_with_logits(
        end_logit[mask], end_target[mask],
        pos_weight=torch.tensor([END_POS_WEIGHT], device=end_logit.device),
    )

    # Temporal smoothing on end logits.
    smooth_loss = temporal_smooth_loss(end_logit,
                                       lambda_smooth=min(0.02 * (epoch / 10), 0.02))

    # Boundary NMS penalty (keeps predicted ends separated).
    seg_penalty = boundary_nms_penalty(
        end_logit, min_distance=5,
        penalty=min(0.05 * (epoch / 20), 0.05),
    )

    total_loss = presence_loss + end_loss + smooth_weight * smooth_loss + seg_penalty

    return total_loss, {
        "total": total_loss.item(),
        "presence": presence_loss.item(),
        "end": end_loss.item(),
        "smooth": smooth_loss.item() if isinstance(smooth_loss, torch.Tensor) else smooth_loss,
        "seg_penalty": seg_penalty.item() if isinstance(seg_penalty, torch.Tensor) else seg_penalty,
    }
