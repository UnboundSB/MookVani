"""
Anti-collapse CTC loss for word classification.

Prevents the model from learning to emit nothing (all blanks):
    - blank floor: minimum average probability mass on non-blank classes
    - emission barrier: penalty if total non-blank emission is too low
    - blank cap: maximum average blank probability
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AntiCollapseCTCLoss(nn.Module):
    def __init__(self, blank: int = 0, blank_floor: float = 0.50,
                 emission_barrier: float = 0.20, blank_cap: float = 0.80,
                 weight_nonblank: float = 50.0):
        super().__init__()
        self.blank = blank
        self.blank_floor = blank_floor
        self.emission_barrier = emission_barrier
        self.blank_cap = blank_cap
        self.weight_nonblank = weight_nonblank
        self.ctc = nn.CTCLoss(blank=blank, zero_infinity=True, reduction="mean")

    def forward(self, log_probs: torch.Tensor, targets: torch.Tensor,
                input_lengths: torch.Tensor, target_lengths: torch.Tensor):
        """
        log_probs: (B, T, num_words+1) log_softmax output
        targets: (B, L) ints in 1..num_words (blank=0 reserved)
        input_lengths: (B,)
        target_lengths: (B,)
        """
        # ---- standard CTC ----
        # CTCLoss expects (T, B, C) log_probs and (B, L) targets
        lp = log_probs.transpose(0, 1)  # (T, B, C)
        ctc_loss = self.ctc(lp, targets, input_lengths, target_lengths)

        # ---- collapse diagnostics ----
        probs = torch.exp(log_probs)                      # (B, T, C)
        blank_prob = probs[:, :, self.blank]              # (B, T)
        nonblank_prob = 1.0 - blank_prob                  # (B, T)

        # mean blank probability over valid frames
        mask = torch.arange(log_probs.shape[1], device=log_probs.device).unsqueeze(0)
        valid = mask < input_lengths.unsqueeze(1)         # (B, T)
        valid_frames = valid.sum().clamp(min=1)

        mean_blank = (blank_prob * valid).sum() / valid_frames
        mean_nonblank = (nonblank_prob * valid).sum() / valid_frames

        # ---- anti-collapse penalties ----
        # blank floor: encourage enough non-blank emission
        floor_penalty = F.relu(self.blank_floor - mean_nonblank) ** 2

        # blank cap: prevent blank from dominating
        cap_penalty = F.relu(mean_blank - self.blank_cap) ** 2

        # emission barrier: strong push if total emission is near zero
        barrier_penalty = F.relu(self.emission_barrier - mean_nonblank)

        total = ctc_loss + self.weight_nonblank * (
            floor_penalty + cap_penalty + barrier_penalty
        )
        return total, {
            "ctc": ctc_loss.item(),
            "mean_blank": mean_blank.item(),
            "mean_nonblank": mean_nonblank.item(),
            "floor_penalty": floor_penalty.item(),
            "cap_penalty": cap_penalty.item(),
            "barrier_penalty": barrier_penalty.item(),
        }
