"""
Shared utilities for the YAP segmentor training pipeline.
=============================================================

Every segmentor trainer (Stage 1 noise filter, Stage 2A word-entity matching,
Stage 2B synthetic segmentation) imports from here so that:

  * clip discovery, dataset, collation and augmentation are written ONCE
  * model creation is a single factory function
  * checkpoint save/load + resume lives in one place
  * loss computation is shared

The trainers themselves contain ONLY stage-specific logic (which loss terms are
active, how many epochs, what the batch looks like).
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Project imports (Backend on sys.path)
# ---------------------------------------------------------------------------
try:
    from yap_conformer import (
        YAPConfig, YAPpingConformer, ConformerConfig, YAP,
        presence_targets, end_targets, focal_presence_loss, layer_conf_loss,
        soft_count_loss, min_gap_penalty, dip_prior_loss,
        sentence_ctc_loss, make_segments, Segment,
    )
    from utils.checkpoint_utils import load_flexible, extract_state_dict
    from utils.pose_augmentation import (
        AugmentConfig, DEFAULT_AUGMENT, augment_pose_sequence,
        augment_hands_batch, harmonize_features, reclamp_missing,
    )
except ImportError:  # direct-execution fallback
    import sys
    _SEGMENTOR_DIR = Path(__file__).resolve().parent.parent / "segmentor"
    _BACKEND_DIR = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_SEGMENTOR_DIR))
    sys.path.insert(0, str(_BACKEND_DIR))
    from yap_conformer import (  # type: ignore[no-redef]
        YAPConfig, YAPpingConformer, ConformerConfig, YAP,
        presence_targets, end_targets, focal_presence_loss, layer_conf_loss,
        soft_count_loss, min_gap_penalty, dip_prior_loss,
        sentence_ctc_loss, make_segments, Segment,
    )
    from utils.checkpoint_utils import (  # type: ignore[no-redef]
        load_flexible, extract_state_dict,
    )
    from utils.pose_augmentation import (  # type: ignore[no-redef]
        AugmentConfig, DEFAULT_AUGMENT, augment_pose_sequence,
        augment_hands_batch, harmonize_features, reclamp_missing,
    )

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent.parent
SEGMENTOR_DIR = BACKEND_DIR / "segmentor"

DEFAULT_CLASSES_PATH = BACKEND_DIR / "models" / "word_vocabulary" / "classes.json"
DEFAULT_WORD_CORPUS = BACKEND_DIR / "data" / "tensors_include50_word_level"
DEFAULT_SEGMENTOR_DATA = SEGMENTOR_DIR / "data"
DEFAULT_STAGE1_OUT = SEGMENTOR_DIR / "models" / "yap_stage1"
DEFAULT_STAGE2_OUT = SEGMENTOR_DIR / "models" / "yap_stage2"

# ---------------------------------------------------------------------------
# Config containers
# ---------------------------------------------------------------------------
@dataclass
class SegmentorLayout:
    name: str
    joints: int
    preset: Callable[..., YAPConfig]

    def build(self, **kw) -> YAPConfig:
        return self.preset(**kw)

LAYOUTS: Dict[str, SegmentorLayout] = {
    "corpus":   SegmentorLayout("corpus",   53, YAPConfig.corpus),
    "holistic": SegmentorLayout("holistic", 79, YAPConfig.holistic),
    "hands":    SegmentorLayout("hands",    42, YAPConfig),
}


@dataclass
class SegmentorModelCfg:
    layout: str = "corpus"
    d_model: int = 128
    n_mem_layers: int = 3
    summary_channels: int = 32
    hippo_order: int = 16
    entity_dim: int = 256
    conformer_d_model: int = 256
    conformer_heads: int = 4
    conformer_layers: int = 4
    conformer_conv_kernel: int = 7
    conformer_dropout: float = 0.1
    num_words: int = 0


# ---------------------------------------------------------------------------
# Clip discovery for segmentor/data/<word>/<take>/
# ---------------------------------------------------------------------------
def discover_segmentor_words(out_root: Path = DEFAULT_SEGMENTOR_DATA) -> Dict[str, List[Path]]:
    """
    Scan ``segmentor/data/`` and return ``{word: [take_dir, ...]}`.

    Expected layout::

        segmentor/data/
            <word>/
                1/
                    word.pt
                    processed.pt   (phase-2 target, optional)
                2/
                    ...
    """
    out_root = Path(out_root)
    words: Dict[str, List[Path]] = {}
    if not out_root.is_dir():
        return words
    for word_dir in sorted(out_root.iterdir()):
        if not word_dir.is_dir():
            continue
        takes = sorted(p for p in word_dir.iterdir() if p.is_dir())
        if takes:
            words[word_dir.name] = takes
    return words


def segmentor_take_items(
    words: Dict[str, List[Path]],
    classes: List[str],
    max_per_class: Optional[int] = None,
    seed: int = 42,
) -> List[Tuple[str, str, int, Path, Path]]:
    """
    Build the work list for Phase 2 (build_word_dataset) and Phase A/B training.

    Returns list of ``(word, take_idx, class_idx, word_path, proc_path)``.
    """
    class_to_idx = {c: i for i, c in enumerate(classes)}
    rng = random.Random(seed)
    items: List[Tuple[str, str, int, Path, Path]] = []
    for word, takes in words.items():
        cls_idx = class_to_idx.get(word)
        if cls_idx is None:
            continue
        take_pool = list(takes)
        rng.shuffle(take_pool)
        if max_per_class is not None:
            take_pool = take_pool[:max_per_class]
        for take_dir in take_pool:
            word_path = take_dir / "word.pt"
            proc_path = take_dir / "processed.pt"
            if not word_path.exists():
                continue
            items.append((word, take_dir.name, cls_idx, word_path, proc_path))
    rng.shuffle(items)
    return items


# ---------------------------------------------------------------------------
# Segmentor datasets
# ---------------------------------------------------------------------------
class SegmentorWordClip(Dataset):
    """
    Single word-level clip for Stage 1 (noise filter) and Phase A (entity match).

    Returns ``(pose, conf, valid, label, t, lengths)`` where pose is
    ``(T, J, 3)`` in the segmentor layout (J = 53 for corpus).
    """

    def __init__(
        self,
        items: List[Tuple[str, str, int, Path, Path]],
        joints: int,
        max_len: int = 400,
        augment: bool = True,
        augment_cfg: Optional[AugmentConfig] = None,
        seed: int = 42,
    ):
        self.items = items
        self.J = joints
        self.max_len = max_len
        self.augment = augment
        self.augment_cfg = augment_cfg or DEFAULT_AUGMENT
        self.seed = seed
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        word, take_idx, cls_idx, word_path, proc_path = self.items[idx]
        x163 = torch.load(word_path, map_location="cpu", weights_only=True).float()
        T = min(len(x163), self.max_len)
        clip = x163[:T]

        # 163 -> (T, 53, 3)
        pose = clip[:, : self.J * 3].contiguous().view(T, self.J, 3)
        conf = torch.ones(T, self.J)
        valid = torch.ones(T, self.J, dtype=torch.bool)
        t = torch.arange(T, dtype=torch.float32) / 30.0

        if self.augment and self.rng.random() < 0.9:
            # augment on canonical (T,163), project back to (T,J,3)
            canon = torch.zeros(T, 163, dtype=pose.dtype)
            canon[:, : self.J * 3] = clip[:, : self.J * 3]
            canon[:, 159:163] = 1.0
            canon = augment_pose_sequence(canon, self.augment_cfg, rng=self.rng)
            pose = canon[:, : self.J * 3].view(T, self.J, 3)

        return pose, conf, valid, cls_idx, t, torch.tensor(T)


def segmentor_collate(batch):
    """Pad variable-length clips to (B, T_max, J, 3)."""
    poses, confs, valids, labels, ts, lengths = zip(*batch)
    B = len(poses)
    T = max(int(l) for l in lengths)
    J = poses[0].shape[1]
    pose_b = torch.zeros(B, T, J, 3)
    conf_b = torch.zeros(B, T, J)
    valid_b = torch.zeros(B, T, J, dtype=torch.bool)
    t_b = torch.zeros(B, T)
    for i, (p, c, v, _, t_, l) in enumerate(zip(poses, confs, valids, labels, ts, lengths)):
        n = int(l)
        pose_b[i, :n] = p
        conf_b[i, :n] = c
        valid_b[i, :n] = v
        t_b[i, :n] = t_
        t_b[i, n:] = t_[-1]
    return pose_b, conf_b, valid_b, torch.tensor(labels, dtype=torch.long), t_b, torch.tensor(lengths, dtype=torch.long)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def create_yap_model(cfg: SegmentorModelCfg, device: torch.device) -> YAPpingConformer:
    """Create a YAPpingConformer with the segmentor layout preset."""
    layout = LAYOUTS[cfg.layout]
    ycfg = layout.build(
        d_model=cfg.d_model,
        n_mem_layers=cfg.n_mem_layers,
        summary_channels=cfg.summary_channels,
        hippo_order=cfg.hippo_order,
        entity_dim=cfg.entity_dim,
    )
    ccfg = ConformerConfig(
        num_words=cfg.num_words,
        d_model=cfg.conformer_d_model,
        heads=cfg.conformer_heads,
        layers=cfg.conformer_layers,
        conv_kernel=cfg.conformer_conv_kernel,
        dropout=cfg.conformer_dropout,
    )
    model = YAPpingConformer(ycfg, ccfg).to(device)
    return model


def load_yap_weights(model: YAPpingConformer, init_path: Optional[Path | str], label: str = "INIT") -> None:
    """Load weights into a YAPpingConformer with flexible (shape-mismatch-tolerant) loading."""
    if init_path is None or not os.path.exists(init_path):
        print(f"[{label}] no init checkpoint at {init_path}; random init")
        return
    sd = torch.load(init_path, map_location="cpu", weights_only=True)
    sd = extract_state_dict(sd)
    load_flexible(model, sd, label=label, verbose=True, strict=False)


# ---------------------------------------------------------------------------
# Checkpoint / resume
# ---------------------------------------------------------------------------
@dataclass
class SegmentorCheckpoint:
    epoch: int
    model_state: Dict[str, torch.Tensor]
    optimizer_state: Optional[Dict[str, torch.Tensor]] = None
    scheduler_state: Optional[Dict[str, torch.Tensor]] = None
    best_loss: float = float("inf")
    history: List[dict] = field(default_factory=list)
    rng_state: Optional[dict] = None


def save_segmentor_ckpt(
    out_dir: Path | str,
    epoch: int,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    best_loss: float = float("inf"),
    history: Optional[List[dict]] = None,
    filename: str = "yap_ckpt.pth",
    tag: str = "",
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "best_loss": best_loss,
        "history": history or [],
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    payload["rng"] = {
        "python": random.getstate(),
        "numpy": np.random.get_state()[1].tolist() if hasattr(np.random.get_state()[1], "tolist") else None,
        "torch": torch.get_rng_state(),
    }
    path = out_dir / filename
    torch.save(payload, path)
    print(f"[{tag}] checkpoint saved -> {path} (epoch {epoch})")
    return path


def load_segmentor_ckpt(
    path: Path | str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: torch.device = torch.device("cpu"),
    tag: str = "",
) -> SegmentorCheckpoint:
    path = Path(path)
    payload = torch.load(path, map_location=device, weights_only=False)
    sd = extract_state_dict(payload.get("model", payload))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[{tag}] loaded model from {path} ({len(sd)} tensors; "
          f"{len(missing)} missing, {len(unexpected)} unexpected)")

    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    if "rng" in payload and payload["rng"] is not None:
        rng = payload["rng"]
        if rng.get("python"):
            random.setstate(rng["python"])
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"])

    return SegmentorCheckpoint(
        epoch=int(payload.get("epoch", 0)),
        model_state=sd,
        optimizer_state=payload.get("optimizer"),
        scheduler_state=payload.get("scheduler"),
        best_loss=float(payload.get("best_loss", float("inf"))),
        history=list(payload.get("history", [])),
    )


# ---------------------------------------------------------------------------
# Loss computation (shared across Stage 1 / Phase A / Phase B)
# ---------------------------------------------------------------------------
def yap_losses(
    yap_output: Dict[str, torch.Tensor],
    spans_t: List[List[Tuple[float, float]]],
    times: torch.Tensor,
    mask: torch.Tensor,
    n_words: Optional[torch.Tensor] = None,
    speed: Optional[torch.Tensor] = None,
    min_gap_k: int = 6,
    presence_weight: float = 1.0,
    end_weight: float = 1.0,
    layer_weight: float = 0.5,
    count_weight: float = 1.0,
    gap_weight: float = 1.0,
    dip_weight: float = 1.0,
    sigma_s: float = 0.05,
    ramp_s: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the full YAP loss bundle.

    Returns ``(total_loss, parts_dict)`` where parts contains detached floats
    for logging.
    """
    presence = yap_output["presence"]
    end = yap_output["end"]
    layer_conf = yap_output["layer_conf"]
    B, T = mask.shape
    device = presence.device

    presence_t = torch.zeros(B, T, device=device)
    end_t = torch.zeros(B, T, device=device)
    for b, ss in enumerate(spans_t):
        n = int(mask[b].sum().item())
        presence_t[b, :n] = presence_targets(times[b, :n], ss, ramp_s=ramp_s)
        end_t[b, :n] = end_targets(times[b, :n], ss, sigma_s=sigma_s, normalize=True)

    pmask = mask.float()

    pres = focal_presence_loss(presence, presence_t, pmask)
    end_raw = F.binary_cross_entropy_with_logits(end, end_t, reduction="none",
                                                 pos_weight=presence.new_tensor(10.0))
    end = (end_raw * pmask).sum() / pmask.sum().clamp_min(1)
    layer = layer_conf_loss(layer_conf, presence_t, pmask)

    total = presence_weight * pres + end_weight * end + layer_weight * layer

    parts = {
        "presence": float(pres.detach()),
        "end": float(end.detach()),
        "layer": float(layer.detach()),
    }

    # Weak-supervision regularisers (Phase B / Stage 3 only)
    if n_words is not None:
        total = total + count_weight * soft_count_loss(end, n_words, pmask)
        parts["count"] = float(soft_count_loss(end, n_words, pmask).detach())

    if gap_weight > 0:
        total = total + gap_weight * min_gap_penalty(end, min_gap_k, pmask)
        parts["min_gap"] = float(min_gap_penalty(end, min_gap_k, pmask).detach())

    if dip_weight > 0 and speed is not None:
        total = total + dip_weight * dip_prior_loss(end, speed, pmask)
        parts["dip_prior"] = float(dip_prior_loss(end, speed, pmask).detach())

    parts["total"] = float(total.detach())
    return total, parts


def conformer_ce_loss(
    word_logits: torch.Tensor,
    token_mask: torch.Tensor,
    word_targets: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy over real entity slots.

    ``word_logits``: (B, K, V+1), class 0 = blank
    ``token_mask``:  (B, K) bool
    ``word_targets``: (B, K) with 0 for padding slots
    """
    B, K, V1 = word_logits.shape
    flat_logits = word_logits.reshape(-1, V1)
    flat_targets = word_targets.reshape(-1)
    flat_mask = token_mask.reshape(-1)
    ce = F.cross_entropy(flat_logits, flat_targets, reduction="none").view(B, K)
    return (ce * token_mask).sum() / token_mask.sum().clamp_min(1)


# ---------------------------------------------------------------------------
# History / reporting
# ---------------------------------------------------------------------------
def write_segmentor_report(
    history: List[dict],
    out_dir: Path | str,
    title: str = "Segmentor training",
    prefix: str = "",
) -> List[str]:
    from utils.plotting import write_report
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Normalise history keys to what plotting.py expects
    norm = {}
    for h in history:
        for k, v in h.items():
            norm.setdefault(k, []).append(v)
    return write_report(norm, str(out_dir), title=title, prefix=prefix)
