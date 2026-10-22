"""
Checkpoint helpers that survive a change of the vocabulary.
=============================================================

Every model in this project has a head whose size depends on the number of
classes:

    word models   head.4.{weight,bias}          -> (num_classes + 1) logits
    CTC models    ctc_proj.1.{weight,bias}      -> (num_classes + 1) logits
    pretrainer    classifier.4.{weight,bias}    -> (num_classes + 1) logits

``nn.Module.load_state_dict(..., strict=False)`` tolerates MISSING and
UNEXPECTED keys, but it still raises a ``RuntimeError`` when a key exists on
both sides with a DIFFERENT SHAPE.  So the moment the vocabulary changes (the
word loader now trains on the 228 frequency-filtered classes instead of the
old 262), every warm start from an older checkpoint dies with

    RuntimeError: size mismatch for head.4.weight: copying a param with shape
    torch.Size([263, 256]) from checkpoint, the shape in current model is
    torch.Size([229, 256]).

:func:`load_flexible` drops the shape-mismatched entries (the heads) before
loading, reports what it dropped and lets the training continue from the
backbone - which is the only part of the checkpoint that is meaningful across
a vocabulary change.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn


def filter_state_dict_for_model(
    state_dict: Dict[str, torch.Tensor],
    model: nn.Module,
    drop_prefixes: Iterable[str] = (),
    verbose: bool = True,
    label: str = "",
) -> Tuple[Dict[str, torch.Tensor], List[Tuple[str, tuple, tuple]]]:
    """Keep only the entries whose shape matches ``model``.

    Returns the filtered state dict and the list of dropped
    ``(key, checkpoint_shape, model_shape)`` triples.
    """
    target = model.state_dict()
    prefixes = tuple(drop_prefixes)

    kept: Dict[str, torch.Tensor] = {}
    dropped: List[Tuple[str, tuple, tuple]] = []

    for key, value in state_dict.items():
        if prefixes and key.startswith(prefixes):
            continue
        if key not in target:
            continue                       # unexpected key -> ignored
        if hasattr(value, "shape") and tuple(value.shape) != tuple(target[key].shape):
            dropped.append((key, tuple(value.shape), tuple(target[key].shape)))
            continue
        kept[key] = value

    if dropped and verbose:
        tag = f"{label} " if label else ""
        print(f"[{tag}CHECKPOINT] Dropped {len(dropped)} shape-mismatched entries "
              f"(the vocabulary changed, the head has to be re-learned):")
        for key, old, new in dropped[:8]:
            print(f"    {key}: checkpoint {old} -> model {new}")
        if len(dropped) > 8:
            print(f"    ... and {len(dropped) - 8} more")

    return kept, dropped


def load_flexible(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    drop_prefixes: Iterable[str] = (),
    verbose: bool = True,
    label: str = "",
    strict: bool = False,
) -> Tuple[List[str], List[str]]:
    """``load_state_dict`` that ignores the entries whose shape changed.

    Returns ``(missing_keys, unexpected_keys)``.
    """
    kept, _ = filter_state_dict_for_model(
        state_dict, model, drop_prefixes=drop_prefixes, verbose=verbose, label=label
    )
    missing, unexpected = model.load_state_dict(kept, strict=strict)
    if verbose:
        tag = f"{label} " if label else ""
        print(f"[{tag}CHECKPOINT] Loaded {len(kept)} tensors "
              f"({len(missing)} missing, {len(unexpected)} unexpected).")
    return list(missing), list(unexpected)


def extract_state_dict(ckpt) -> Dict[str, torch.Tensor]:
    """Accept a raw state dict or a training checkpoint and return the weights."""
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "ema_state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt
