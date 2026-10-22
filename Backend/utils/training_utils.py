"""
Training helpers shared by the word-level and the sentence-level trainers.
==========================================================================

Three things live here:

1. :func:`unfreeze_and_extend` — releases a frozen part of the model WITHOUT
   throwing away the optimiser state.  The phase trainers used to *recreate*
   the optimiser at the unfreeze point, which reset the AdamW moments of the
   head (and of the RNN/CNN) and restarted the LR schedule from scratch: the
   loss spiked for an epoch and the accumulated momentum was lost.

2. :func:`escape_from_plateau` — the escape used by the word-level phases.
   The CTC trainers already had one (``escape_local_minimum`` in
   ``train_conformer_ctc.py``); the word phases had nothing, so a phase that
   plateaued simply annealed its LR to zero and stayed there.

3. :func:`build_warm_restart_scheduler` — the cosine-with-warm-restart
   schedule used by the phase trainers, so a phase can recover from a plateau
   instead of dying with the LR at its floor.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim


# --------------------------------------------------------------------------- #
# 1. Unfreezing without losing the optimiser state
# --------------------------------------------------------------------------- #
def unfreeze_and_extend(
    optimizer: optim.Optimizer,
    params: Iterable[nn.Parameter],
    lr: float,
    name: Optional[str] = None,
    weight_decay: Optional[float] = None,
) -> List[dict]:
    """Add the parameters of a frozen part of the model to the optimiser.

    The parameters are added as a NEW parameter group, so the groups that are
    already training keep their AdamW moments and their current LR.
    """
    params = [p for p in params if p.requires_grad]
    for p in params:
        p.requires_grad = True
    if not params:
        return []
    group = {"params": params, "lr": lr}
    if name:
        group["name"] = name
    if weight_decay is not None:
        group["weight_decay"] = weight_decay
    optimizer.add_param_group(group)
    return [group]


# --------------------------------------------------------------------------- #
# 2. Escaping a plateau
# --------------------------------------------------------------------------- #
def escape_from_plateau(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scale: float = 0.02,
    skip_prefixes: Iterable[str] = ("backbone.",),
    blank_bias: Optional[float] = None,
    reset_lr: bool = True,
    rng: Optional[random.Random] = None,
) -> str:
    """Kick the optimiser out of a flat local minimum.

    Three mechanisms, applied together (any single one is usually not enough
    once a head has settled on a dominant token):

      1. a small Gaussian perturbation of the trainable weights (SGD-noise
         style).  The pretrained backbone is left untouched so the isolated
         sign representation survives;
      2. the reset of the BatchNorm running statistics of the layers that are
         actually in training mode;
      3. the reset of the learning rate to its base value, so the next step
         is a big one again.

    Returns a short human-readable description of what was done.
    """
    rng = rng or random
    skipped = tuple(skip_prefixes)

    touched = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not p.requires_grad or p.dim() < 1:
                continue
            if any(name.startswith(prefix) for prefix in skipped):
                continue
            sigma = max(float(p.std()), 1e-3)
            p.add_(torch.randn_like(p) * scale * sigma)
            touched += 1

    bn_reset = 0
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.training:
            m.reset_running_stats()
            bn_reset += 1

    if blank_bias is not None and hasattr(model, "rebalance_ctc_head"):
        model.rebalance_ctc_head(blank_bias=blank_bias)

    if reset_lr:
        for g in optimizer.param_groups:
            if "base_lr" in g:
                g["lr"] = g["base_lr"]
            else:
                g.setdefault("base_lr", g["lr"])
                g["lr"] = g["base_lr"]

    return (f"perturbed {touched} tensors (sigma={scale}), reset {bn_reset} trainable "
            f"BatchNorm stats, LR reset to base")


# --------------------------------------------------------------------------- #
# 3. Schedulers
# --------------------------------------------------------------------------- #
def build_warm_restart_scheduler(
    optimizer: optim.Optimizer,
    total_epochs: int,
    restart_every: int = 20,
    eta_min: float = 1e-6,
    warmup_epochs: int = 2,
):
    """Cosine annealing with warm restarts.

    A plain cosine anneals the LR to ``eta_min`` and stays there: a phase that
    plateaus half-way through can never recover.  With warm restarts the LR
    jumps back to its maximum every ``restart_every`` epochs, which is the
    cheapest reliable way out of a sharp minimum for a classification head.

    Returns ``(scheduler, restart_every)``.
    """
    if restart_every <= 0:
        return (optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_epochs), eta_min=eta_min), 0)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, restart_every), T_mult=1, eta_min=eta_min)
    return scheduler, restart_every


def cosine_warmup_scheduler(
    optimizer: optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int = 2,
    eta_min: float = 1e-6,
):
    """Linear warmup followed by a cosine decay (the schedule of the CTC trainers)."""
    warmup = max(1, min(warmup_epochs, max(1, total_epochs // 4)))
    warmup_sched = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, total_iters=warmup)
    cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_epochs - warmup), eta_min=eta_min)
    return optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup])


def plateau_escape_scheduler(
    scheduler,
    optimizer: optim.Optimizer,
    epochs_without_improvement: int,
    patience: int,
    restart_every: int = 0,
):
    """Rebuild the schedule after a plateau.

    When the phase has not improved for ``patience`` epochs, the LR is put back
    to its base value and the cosine cycle is restarted from there (a warm
    restart), instead of letting the schedule anneal to zero.
    """
    if patience <= 0 or epochs_without_improvement < patience:
        return scheduler, False

    for g in optimizer.param_groups:
        g["lr"] = g.get("base_lr", g["lr"])

    if restart_every > 0:
        new_sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(1, restart_every), T_mult=1, eta_min=1e-6)
    else:
        new_sched = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, 20), eta_min=1e-6)
    return new_sched, True
