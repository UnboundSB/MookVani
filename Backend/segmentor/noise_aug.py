"""
noise_aug.py
============
Realistic MediaPipe noise augmentation for YAP Stage-1 noise-filtration training.

Applies all known MediaPipe failure modes to a clean word-level clip:
  1. Gaussian jitter on coordinates
  2. Per-joint random dropout (valid=False)
  3. Whole-part random dropout (hand disappears for N frames)
  4. Left/right hand swap (labels the wrong hand)
  5. Depth-axis noise (z coordinate sign flip / scale)
  6. Speed warp (non-uniform temporal resampling)
  7. Idle padding (random leading/trailing dead frames)

Usage
-----
    from noise_aug import add_noise
    pose_n, conf_n, valid_n = add_noise(pose, conf, valid, cfg, rng=rng)

The clean and noisy versions of the SAME clip are fed to two model copies
to compute the consistency loss in train_noise_filter.py.
"""
from __future__ import annotations
import random
import torch
from yap_conformer import YAPConfig


def add_noise(
    pose: torch.Tensor,
    conf: torch.Tensor,
    valid: torch.Tensor,
    cfg: YAPConfig,
    *,
    jitter_std: float = 0.015,
    joint_drop_p: float = 0.06,
    part_drop_p: float = 0.20,
    part_drop_max_frames: int = 12,
    swap_hands_p: float = 0.10,
    depth_flip_p: float = 0.12,
    depth_scale_range: tuple = (0.5, 2.0),
    speed_warp_range: tuple = (0.75, 1.35),
    idle_pad_max: int = 15,
    scale: float = 1.0,
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (pose_noisy, conf_noisy, valid_noisy) with all augmentations applied.
    Operates on a single clip: pose (T,J,3), conf (T,J), valid (T,J) bool.
    The returned tensors have the SAME shape (after speed warp + padding the T
    dimension is resampled; the caller should record the new lengths).

    ``scale`` is the GLOBAL intensity of the corruption: 0 = a pristine clip,
    1 = the nominal noise, 2 = twice as bad.  The magnitudes that are a
    probability or a ratio (the dropout rates, the speed-warp range) are
    scaled towards their nominal value, while the magnitudes that are a size
    (the jitter, the idle padding) are multiplied.  Drawing ``scale`` at
    RANDOM for every batch is what makes the student robust across the whole
    range of noise levels instead of only the nominal one.
    """
    if rng is None:
        rng = random.Random()
    T, J, _ = pose.shape
    dev = pose.device
    scale = max(0.0, float(scale))

    pose = pose.clone()
    conf = conf.clone()
    valid = valid.clone()

    # 1. Gaussian jitter
    if jitter_std > 0 and scale > 0:
        pose += (jitter_std * scale) * torch.randn_like(pose)

    # 2. Per-joint dropout
    if scale > 0:
        drop_mask = torch.rand(T, J, device=dev) < min(1.0, joint_drop_p * scale)
        valid = valid & ~drop_mask

    # 3. Whole-part dropout
    offs = cfg.part_offsets()
    for spec in cfg.parts:
        if scale > 0 and rng.random() < min(1.0, part_drop_p * scale):
            n_frames = max(1, int(round(rng.randint(1, part_drop_max_frames) * scale)))
            start = rng.randint(0, max(0, T - n_frames))
            o = offs[spec.name]
            valid[start:start + n_frames, o:o + spec.num_joints] = False

    # 4. Left/right hand swap (only when both hands are present in the layout)
    hand_parts = [s for s in cfg.parts if s.kind == "hand"]
    if len(hand_parts) >= 2 and scale > 0 and rng.random() < min(1.0, swap_hands_p * scale):
        o0, o1 = offs[hand_parts[0].name], offs[hand_parts[1].name]
        n = hand_parts[0].num_joints
        pose[:, o0:o0 + n], pose[:, o1:o1 + n] = pose[:, o1:o1 + n].clone(), pose[:, o0:o0 + n].clone()
        conf[:, o0:o0 + n], conf[:, o1:o1 + n] = conf[:, o1:o1 + n].clone(), conf[:, o0:o0 + n].clone()
        valid[:, o0:o0 + n], valid[:, o1:o1 + n] = valid[:, o1:o1 + n].clone(), valid[:, o0:o0 + n].clone()

    # 5. Depth-axis corruption (z)
    if scale > 0:
        if rng.random() < min(1.0, depth_flip_p * scale):
            pose[:, :, 2] = -pose[:, :, 2]
        z_lo, z_hi = depth_scale_range
        z_scale = rng.uniform(z_lo, z_hi) if scale >= 1.0 else rng.uniform(1.0 - (1.0 - z_lo) * scale,
                                                                              1.0 + (z_hi - 1.0) * scale)
        pose[:, :, 2] = pose[:, :, 2] * z_scale

    # 6. Speed warp (non-uniform temporal resampling via linear interpolation)
    if scale > 0:
        lo, hi = speed_warp_range
        centre = 1.0
        amp = (hi - centre) * scale
        warp = rng.uniform(max(0.2, centre - amp), centre + amp)
    else:
        warp = 1.0
    new_T = max(4, int(round(T * warp)))
    idx_f = torch.linspace(0, T - 1, new_T, device=dev)
    idx_lo = idx_f.long().clamp(0, T - 1)
    idx_hi = (idx_lo + 1).clamp(0, T - 1)
    alpha = (idx_f - idx_lo.float()).view(new_T, 1, 1)
    pose = pose[idx_lo] * (1 - alpha) + pose[idx_hi] * alpha
    conf = conf[idx_lo] * (1 - alpha.squeeze(-1)) + conf[idx_hi] * alpha.squeeze(-1)
    valid = valid[idx_lo]  # nearest-neighbour for bool
    T = new_T

    # 7. Idle padding (dead frames at random ends)
    pad_lead = rng.randint(0, int(round(idle_pad_max * scale)))
    pad_tail = rng.randint(0, int(round(idle_pad_max * scale)))
    if pad_lead or pad_tail:
        z3 = pose.new_zeros(pad_lead + T + pad_tail, J, 3)
        zc = conf.new_zeros(pad_lead + T + pad_tail, J)
        zv = torch.zeros(pad_lead + T + pad_tail, J, dtype=torch.bool, device=dev)
        z3[pad_lead:pad_lead + T] = pose
        zc[pad_lead:pad_lead + T] = conf
        zv[pad_lead:pad_lead + T] = valid
        pose, conf, valid = z3, zc, zv

    return pose, conf, valid
