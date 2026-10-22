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
from typing import Callable, Dict, Iterable, List, Optional

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


# --------------------------------------------------------------------------- #
# Per-layer learning rates and schedules
# --------------------------------------------------------------------------- #
#: The layers of the CTC stack, in the order in which they are trained.
LAYER_ORDER = ["front", "expander", "gru", "transformer", "tcnn", "conformer",
                "head", "aux"]

#: The default learning rate of every layer.  The rule:
#:   - a NEW module (front-end, expander, conformer blocks, heads) learns FAST,
#:     it has no pretrained weights to protect;
#:   - a PRETRAINED module (GRU, Transformer, TCNN) learns SLOW, a high LR
#     would destroy the representation it was trained for.
DEFAULT_LAYER_LRS = {
    "front":      3.0e-4,
    "expander":   3.0e-4,
    "gru":        1.0e-5,
    "transformer": 1.0e-5,
    "tcnn":       3.0e-5,
    "conformer":  1.5e-4,
    "head":       1.5e-4,
    "aux":        1.5e-4,
}


def _layer_of(name: str) -> str:
    """Which group does a parameter belong to?"""
    if name.startswith("front."):
        return "front"
    if name.startswith("expander.") or name.startswith("backbone.expander."):
        return "expander"
    if name.startswith("backbone.gru.") or name.startswith("gru."):
        return "gru"
    if name.startswith("backbone.trans.") or name.startswith("trans."):
        return "transformer"
    if name.startswith("backbone.t1.") or name.startswith("backbone.t2.") \
            or name.startswith("t1.") or name.startswith("t2."):
        return "tcnn"
    if name.startswith("conformer."):
        return "conformer"
    if name.startswith("aux_proj."):
        return "aux"
    return "head"


def build_layer_param_groups(model, layer_lrs: Optional[dict] = None,
                             weight_decay: float = 1e-4,
                             no_decay: bool = True) -> List[dict]:
    """One parameter group per LAYER, each with its own learning rate.

    The groups follow the structure of the stack: the new modules (front-end,
    expander, conformer blocks, CTC head) get a high LR because they start
    from scratch, and the pretrained modules (GRU, Transformer, TCNN) get a
    low LR because a high one would destroy the representation they were
    pretrained on.

    ``no_decay`` keeps the biases and the normalisation weights out of the
    weight decay, which is the standard recipe for a transformer.
    """
    lrs = dict(DEFAULT_LAYER_LRS)
    if layer_lrs:
        lrs.update(layer_lrs)

    buckets: Dict[str, List[nn.Parameter]] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        layer = _layer_of(name)
        buckets.setdefault(layer, []).append(p)

    groups = []
    for layer in LAYER_ORDER:
        params = buckets.pop(layer, [])
        if not params:
            continue
        if no_decay:
            decay_params = [p for n, p in model.named_parameters()
                            if _layer_of(n) == layer and p.requires_grad and p.ndim > 1]
            no_decay_params = [p for n, p in model.named_parameters()
                               if _layer_of(n) == layer and p.requires_grad and p.ndim <= 1]
            if decay_params:
                groups.append({"params": decay_params, "lr": lrs[layer],
                               "weight_decay": weight_decay, "name": layer})
            if no_decay_params:
                groups.append({"params": no_decay_params, "lr": lrs[layer],
                               "weight_decay": 0.0, "name": f"{layer}_nd"})
        else:
            groups.append({"params": params, "lr": lrs[layer],
                           "weight_decay": weight_decay, "name": layer})
    # anything left over (a custom module) goes to the head group
    for layer, params in buckets.items():
        groups.append({"params": params, "lr": lrs.get("head", 1.5e-4),
                       "weight_decay": weight_decay, "name": layer})
    return groups


class PerGroupCosineAnnealing:
    """A cosine schedule where every parameter group has its own floor.

    ``torch.optim.lr_scheduler.CosineAnnealingLR`` anneals EVERY group to the
    same ``eta_min``, which is wrong for a stack with a 30x learning-rate
    spread: the pretrained layers (LR 1e-5) would be pushed to 1e-6 while the
    new ones (LR 3e-4) would need to keep annealing much further down.

    This scheduler anneals every group to ``base_lr * min_lr_ratio``, so the
    RELATIVE distance travelled is the same for all of them.
    """

    def __init__(self, optimizer, total_epochs: int, min_lr_ratio: float = 0.01,
                 warmup_epochs: int = 0, last_epoch: int = -1):
        self.optimizer = optimizer
        self.total_epochs = max(1, total_epochs)
        self.min_lr_ratio = min_lr_ratio
        self.warmup_epochs = max(0, warmup_epochs)
        self.last_epoch = last_epoch
        self.base_lrs = [g.get("base_lr", g["lr"]) for g in optimizer.param_groups]
        self.step()

    def _factor(self, epoch: int) -> float:
        if self.warmup_epochs and epoch < self.warmup_epochs:
            return 0.1 + 0.9 * (epoch + 1) / self.warmup_epochs
        progress = (epoch - self.warmup_epochs) / max(
            1, self.total_epochs - self.warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        return self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (
            1 + math.cos(math.pi * progress))

    def step(self, epoch: int = None):
        if epoch is None:
            self.last_epoch += 1
            epoch = self.last_epoch
        factor = self._factor(epoch)
        for g, base in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = base * factor
        return [g["lr"] for g in self.optimizer.param_groups]

    def get_last_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]

    def state_dict(self):
        return {"last_epoch": self.last_epoch}

    def load_state_dict(self, state):
        self.last_epoch = state.get("last_epoch", -1)


def freeze_layers(model, layers: Sequence[str], freeze: bool = True) -> List[str]:
    """Freeze (or release) whole LAYERS of the stack.

    This is the staged schedule of the pipeline: the pretrained Transformer is
    frozen for the first epochs (so the new modules can settle around it) and
    released afterwards.
    """
    touched = []
    for name, p in model.named_parameters():
        layer = _layer_of(name)
        if layer in layers:
            p.requires_grad = not freeze
            touched.append(layer)
    return sorted(set(touched))
