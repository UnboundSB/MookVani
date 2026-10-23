"""
Canonical MediaPipe pose augmentation for the 163-dim ISL landmark tensors.
==============================================================================

This module is the SINGLE SOURCE OF TRUTH for every geometric / temporal
augmentation applied to an ISL landmark clip.  Before it existed the same
augmentation was copy-pasted into nine different files with nine different
index conventions, and two of those copies were buggy:

  * ``train_synthetic_sentences.py`` and ``train_synthetic_sentences_phase2.py``
    swapped the 5-landmark block ``42:47`` against ``47:52``.  In the 53-landmark
    view, ``42:45`` are the arms and ``46:52`` are the face, so that swap mixed
    arm landmarks with eye landmarks.
  * ``pretrainer.py`` / ``phase2..phase6`` never swapped the mouth corners
    (landmarks 51/52), so a mirrored clip had an asymmetric mouth.
  * ``scripts/tensor_augmentation.py`` rotated around (0.5, 0.5) because it
    assumed raw ``[0, 1]`` image coordinates, while the corpus is
    anchor-relative and scale-normalised.

Tensor layout (canonical, produced by ``scripts/extraction.py``)::

    [0:63]    left hand    21 landmarks x 3
    [63:126]  right hand   21 landmarks x 3
    [126:138] arms         4 landmarks x 3  (ls, rs, le, re)
    [138:159] face         7 landmarks x 3  (l_eye, r_eye, nose, l_cheek, r_cheek, l_mouth, r_mouth)
    [159:163] flags        [lh, rh, arms, face]

The 159 landmark channels are 53 landmarks x 3, so the 53-landmark view is::

    0:21  left hand      21:42 right hand     42:46 arms     46:53 face

Geometry conventions of the corpus
-----------------------------------
* Coordinates are *anchor relative* and *scale normalised* (1.0 == one shoulder
  width), so the body does NOT sit at the origin: the neck (shoulder midpoint)
  sits at roughly ``(3.59, 2.72, -1.20)`` in the word corpus and at
  ``(0.01, 0.61, -0.52)`` in the sentence corpus.  Every rotation therefore
  pivots on the NECK, never on the origin - an origin-centred rotation would
  swing the skeleton several shoulder widths out of frame.
* An undetected modality is encoded as an ALL-ZERO block (``raw_zero``) in the
  word corpus, and MIXED (zeros / resting placeholder) in the sentence corpus.
  Every augmentation restores the zero convention afterwards, so the presence
  flags and the geometry can never disagree.
* The FACE block is dead in the word corpus (present in 0.45% of the frames),
  so it is excluded from the geometric transforms entirely.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Canonical layout
# --------------------------------------------------------------------------- #
LH_LO, LH_HI = 0, 63          # left hand block
RH_LO, RH_HI = 63, 126        # right hand block
ARM_LO, ARM_HI = 126, 138     # arms block
FACE_LO, FACE_HI = 138, 159   # face block
FLAG_LO, FLAG_HI = 159, 163   # presence flags [lh, rh, arms, face]

N_LANDMARKS = 53              # 21 + 21 + 4 + 7
BODY_DIMS = ARM_HI            # 138 -> lh + rh + arms (the face block is dead)

# 53-landmark sub-ranges
LH_PTS = (0, 21)
RH_PTS = (21, 42)
ARM_PTS = (42, 46)
FACE_PTS = (46, 53)

# Landmark indices inside the 53-landmark view
LS, RS, LE, RE = 42, 43, 44, 45
L_EYE, R_EYE, NOSE, L_CHEEK, R_CHEEK, L_MOUTH, R_MOUTH = 46, 47, 48, 49, 50, 51, 52

# Presence flag indices
F_LH, F_RH, F_ARMS, F_FACE = 159, 160, 161, 162

# --------------------------------------------------------------------------- #
# Geometric frame of each corpus (measured on the finished tensors)
# --------------------------------------------------------------------------- #
NECK_ANCHOR_WORD = (3.592, 2.769, -1.233)   # data/tensors_include50_word_level
NECK_ANCHOR_SENT = (0.011, 0.605, -0.523)   # data/tensors_sentence_level
NECK_OFFSET = tuple(round(a - b, 3) for a, b in zip(NECK_ANCHOR_WORD, NECK_ANCHOR_SENT))
NECK_OFFSET_WORD = (0.0, 0.0, 0.0)

# The canonical resting pose of an undetected hand (21 landmarks x 3).
# NOTE: the corpora themselves store an undetected hand as an ALL-ZERO block
# (see :func:`reclamp_missing`); these constants describe the *physical*
# resting pose and are only used to synthesise the idle / rest segments.
MISSING_LH = torch.tensor([-1.5, 3.0, 0.0]).repeat(21)   # (63,)
MISSING_RH = torch.tensor([+1.5, 3.0, 0.0]).repeat(21)   # (63,)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def as_sequence(seq: torch.Tensor) -> torch.Tensor:
    """Normalise any clip to a contiguous float32 ``(T, 163)`` tensor.

    Accepts ``(163,)``, ``(1, 163)`` and ``(T, 163)`` so that the loaders can
    consume the single-frame word tensors and the multi-frame sentence tensors
    through the same code path.
    """
    if not torch.is_tensor(seq):
        seq = torch.as_tensor(seq)
    if seq.dim() == 1:
        seq = seq.unsqueeze(0)
    elif seq.dim() > 2:
        seq = seq.squeeze(0)
    return seq.contiguous().float()


def reclamp_missing(seq: torch.Tensor) -> torch.Tensor:
    """Restore the corpus convention: a modality whose presence flag is 0 owns an
    all-zero geometry block.

    This is what keeps the presence flags and the geometry consistent after a
    geometric augmentation, and it removes the mixed encoding of the sentence
    corpus (zeros *and* resting placeholders for the same physical state).
    """
    seq = seq.clone()
    for lo, hi, flag in (
        (LH_LO, LH_HI, F_LH),
        (RH_LO, RH_HI, F_RH),
        (ARM_LO, ARM_HI, F_ARMS),
        (FACE_LO, FACE_HI, F_FACE),
    ):
        absent = seq[:, flag] < 0.5
        if bool(absent.any()):
            seq[absent, lo:hi] = 0.0
    return seq


def harmonize_features(seq: torch.Tensor, zero_face: bool = True,
                        offset: Optional[tuple] = None) -> torch.Tensor:
    """Move a landmark clip into the geometric frame the backbone was pretrained in.

    The three tensor corpora in ``Backend/data`` were extracted with three
    different anchors, so the same physical point sits in three different places
    (see the module docstring).  The correction is a single constant
    translation per corpus, plus clearing the dead face block.

    Call this BEFORE :func:`reclamp_missing`, so the zero convention is written
    last.
    """
    if seq.shape[-1] != 163:
        return seq
    if offset is None:
        offset = NECK_OFFSET
    seq = seq.contiguous().clone()
    if any(abs(float(v)) > 1e-9 for v in offset):
        lm = seq[:, :159].reshape(-1, N_LANDMARKS, 3)
        off = seq.new_tensor(offset).view(1, 1, 3)
        seq[:, :159] = (lm + off).reshape(-1, 159)
    if zero_face:
        seq[:, FACE_LO:FACE_HI] = 0.0
        seq[:, F_FACE] = 0.0
    return reclamp_missing(seq)


def harmonize_frame(feat) -> "numpy.ndarray":
    """Align a SINGLE 163-dim frame to the reference frame of the backbone.

    Same operation as :func:`harmonize_features`, but for the live inference
    path, which produces the features frame by frame.
    """
    import numpy as np

    f = np.asarray(feat, dtype=np.float32).copy()
    if f.shape[-1] != 163:
        return f
    f[:159] += np.tile(np.asarray(NECK_OFFSET, dtype=np.float32), N_LANDMARKS)
    f[FACE_LO:FACE_HI] = 0.0
    f[F_FACE] = 0.0
    if f[F_LH] < 0.5:
        f[LH_LO:LH_HI] = 0.0
    if f[F_RH] < 0.5:
        f[RH_LO:RH_HI] = 0.0
    return f


def _neck_pivot(coords: torch.Tensor) -> torch.Tensor:
    """Per-frame rotation pivot: the shoulder midpoint when the arms are visible,
    the centroid of the visible landmarks otherwise.  Returns ``(T, 2)``."""
    T = coords.shape[0]
    # the arms block (landmarks 42..45) is zero when the pose was not detected
    arms_present = coords[:, ARM_PTS[0]:ARM_PTS[1], :].abs().sum(dim=(1, 2)) > 0
    ls = coords[:, LS, :2]
    rs = coords[:, RS, :2]
    pivot = (ls + rs) / 2.0
    # frames without arms -> centroid of the non-zero landmarks
    mask = coords[:, :, :2].abs().sum(dim=2) > 0     # (T, 53)
    n = mask.sum(dim=1).clamp(min=1).unsqueeze(1)    # (T, 1)
    centroid = (coords[:, :, :2] * mask.unsqueeze(-1)).sum(dim=1) / n
    use_centroid = ~arms_present
    pivot[use_centroid] = centroid[use_centroid]
    return pivot


# --------------------------------------------------------------------------- #
# Geometric / temporal primitives
# --------------------------------------------------------------------------- #
def mirror_pose(seq: torch.Tensor, about: str = "neck") -> torch.Tensor:
    """Mirror the pose so the clip reads as if the same sign were performed by a
    LEFT-handed signer.

    The X axis is inverted and every left/right landmark PAIR is swapped
    (hands, shoulders, elbows, eyes, cheeks AND the mouth corners).  Y (up/down)
    and Z (depth) are left untouched, so the signer never turns upside down or
    backwards - only the handedness changes.  The presence flags of the two
    hands are swapped with the hands themselves.

    ``about``
        ``"neck"``   - reflect the X axis around the SIGNER'S OWN vertical axis
                       (the shoulder midpoint).  This is the default and it is
                       the only correct choice for this corpus: the landmarks
                       are anchored at the image origin, so the neck sits at
                       x ~ +3.5 shoulder widths.  Reflecting around the origin
                       would teleport the whole body to x ~ -3.5, far outside
                       the region the backbone was pretrained on.
        ``"origin"`` - reflect around x = 0 (only correct for a corpus that is
                       centred on the body, which this one is not).
    """
    seq = as_sequence(seq).clone()
    T = seq.shape[0]
    c = seq[:, :159].reshape(T, N_LANDMARKS, 3).clone()

    if about == "neck":
        pivot = _neck_pivot(c)
        c[:, :, 0] = 2.0 * pivot[:, 0:1] - c[:, :, 0]
    elif about == "origin":
        c[:, :, 0] = -c[:, :, 0]
    else:
        raise ValueError(f"Unknown mirror pivot '{about}' - use 'neck' or 'origin'")

    c[:, LH_PTS[0]:LH_PTS[1]], c[:, RH_PTS[0]:RH_PTS[1]] = (   # swap hands
        c[:, RH_PTS[0]:RH_PTS[1]].clone(), c[:, LH_PTS[0]:LH_PTS[1]].clone()
    )
    c[:, LS], c[:, RS] = c[:, RS].clone(), c[:, LS].clone()    # shoulders
    c[:, LE], c[:, RE] = c[:, RE].clone(), c[:, LE].clone()    # elbows
    c[:, L_EYE], c[:, R_EYE] = c[:, R_EYE].clone(), c[:, L_EYE].clone()
    c[:, L_CHEEK], c[:, R_CHEEK] = c[:, R_CHEEK].clone(), c[:, L_CHEEK].clone()
    c[:, L_MOUTH], c[:, R_MOUTH] = c[:, R_MOUTH].clone(), c[:, L_MOUTH].clone()

    seq[:, :159] = c.reshape(T, 159)
    seq[:, F_LH], seq[:, F_RH] = seq[:, F_RH].clone(), seq[:, F_LH].clone()
    return reclamp_missing(seq)


def tilt_pose(seq: torch.Tensor, max_deg: float = 12.0,
              rng: Optional[random.Random] = None) -> torch.Tensor:
    """Rotate the skeleton in the image plane (the signer leans left / right).

    The rotation pivots on the NECK, not on the origin: the corpus is anchored
    at the image origin, so the neck sits ~3.6 shoulder widths away from it and
    an origin-centred rotation would throw the whole body out of frame.
    """
    seq = as_sequence(seq).clone()
    T = seq.shape[0]
    rng = rng or random
    ang = math.radians(rng.uniform(-max_deg, max_deg))
    ca, sa = math.cos(ang), math.sin(ang)

    c = seq[:, :BODY_DIMS].reshape(T, -1, 3).clone()
    pivot = _neck_pivot(c)
    dx = c[:, :, 0] - pivot[:, 0:1]
    dy = c[:, :, 1] - pivot[:, 1:2]
    c[:, :, 0] = dx * ca - dy * sa + pivot[:, 0:1]
    c[:, :, 1] = dx * sa + dy * ca + pivot[:, 1:2]
    seq[:, :BODY_DIMS] = c.reshape(T, BODY_DIMS)
    return reclamp_missing(seq)


def shift_pose(seq: torch.Tensor, max_dx: float = 0.06, max_dy: float = 0.03,
               rng: Optional[random.Random] = None) -> torch.Tensor:
    """Translate the skeleton LEFT / RIGHT (and slightly up / down).

    The magnitude is bounded and expressed in shoulder widths (1.0 == one
    shoulder width), so the coordinates can never be pushed out of the frame:
    the whole point of the shift is a small camera / body displacement, not a
    translation that makes the landmarks vanish away.  Only the body block is
    moved; the (dead) face block is left alone.
    """
    seq = as_sequence(seq).clone()
    T = seq.shape[0]
    rng = rng or random
    dx = rng.uniform(-max_dx, max_dx)
    dy = rng.uniform(-max_dy, max_dy)

    c = seq[:, :BODY_DIMS].reshape(T, -1, 3).clone()
    c[:, :, 0] += dx
    c[:, :, 1] += dy
    seq[:, :BODY_DIMS] = c.reshape(T, BODY_DIMS)
    return reclamp_missing(seq)


def scale_pose(seq: torch.Tensor, lo: float = 0.90, hi: float = 1.10,
               rng: Optional[random.Random] = None) -> torch.Tensor:
    """Scale the skeleton about the NECK (a different camera distance).

    The scale is applied around the pivot, not around the origin, so the body
    stays where it is and only the apparent size changes.
    """
    seq = as_sequence(seq).clone()
    T = seq.shape[0]
    rng = rng or random
    s = rng.uniform(lo, hi)

    c = seq[:, :BODY_DIMS].reshape(T, -1, 3).clone()
    pivot = _neck_pivot(c)
    c[:, :, 0] = (c[:, :, 0] - pivot[:, 0:1]) * s + pivot[:, 0:1]
    c[:, :, 1] = (c[:, :, 1] - pivot[:, 1:2]) * s + pivot[:, 1:2]
    seq[:, :BODY_DIMS] = c.reshape(T, BODY_DIMS)
    return reclamp_missing(seq)


def time_stretch(seq: torch.Tensor, lo: float = 0.80, hi: float = 1.25,
                 rng: Optional[random.Random] = None) -> torch.Tensor:
    """Speed the sign up or slow it down.

    ``factor < 1`` produces MORE frames (the sign is slower), ``factor > 1``
    produces fewer frames (the sign is faster).  Only the 159 landmark channels
    are interpolated; the 4 presence flags are resampled with nearest
    neighbour and re-thresholded, because interpolating them would create
    fractional flags and silently corrupt the missing-modality bookkeeping.
    """
    seq = as_sequence(seq).clone()
    T = seq.shape[0]
    rng = rng or random
    factor = rng.uniform(lo, hi)
    new_T = max(8, int(round(T * factor)))
    if new_T == T:
        return seq
    lm = F.interpolate(seq[:, :159].T.unsqueeze(0), size=new_T,
                       mode="linear", align_corners=True).squeeze(0).T
    fl = F.interpolate(seq[:, FLAG_LO:FLAG_HI].T.unsqueeze(0), size=new_T,
                       mode="nearest").squeeze(0).T
    return torch.cat([lm, (fl > 0.5).to(seq.dtype)], dim=1)


def temporal_shift(seq: torch.Tensor, max_frames: int = 3,
                   rng: Optional[random.Random] = None) -> torch.Tensor:
    """Slide the clip in TIME (the sign starts earlier or later).

    The frames that leave the window are replaced by the edge frame, so the
    clip keeps its length and the CTC alignment is not disturbed.
    """
    seq = as_sequence(seq).clone()
    T = seq.shape[0]
    if T < 4 or max_frames <= 0:
        return seq
    rng = rng or random
    k = rng.randint(-max_frames, max_frames)
    if k == 0:
        return seq
    idx = torch.arange(T, device=seq.device) + k
    idx = idx.clamp(0, T - 1)
    return seq[idx]


def temporal_mask(seq: torch.Tensor, max_mask: int = 3, n_masks: int = 2,
                  rng: Optional[random.Random] = None) -> torch.Tensor:
    """SpecAugment-style temporal cutout.

    The masked frames are zeroed AND their presence flags are cleared, so the
    model never sees the inconsistent state "modality present but geometry
    exactly zero" (a state that does not occur in the extracted data).
    """
    seq = as_sequence(seq).clone()
    T = seq.shape[0]
    rng = rng or random
    for _ in range(rng.randint(1, max(1, n_masks))):
        ml = rng.randint(1, max(1, max_mask))
        if ml >= T:
            continue
        st = rng.randint(0, T - ml)
        seq[st:st + ml, :159] = 0.0
        seq[st:st + ml, FLAG_LO:FLAG_HI] = 0.0
    return seq


def add_noise(seq: torch.Tensor, std: float = 0.005,
              rng: Optional[random.Random] = None) -> torch.Tensor:
    """Add a small Gaussian jitter on the BODY block only (never on the flags)."""
    if std <= 0:
        return seq
    seq = as_sequence(seq).clone()
    rng = rng or random
    noise = torch.randn(seq.shape[0], BODY_DIMS // 3, 3, generator=None) * std
    seq[:, :BODY_DIMS] += noise.reshape(seq.shape[0], BODY_DIMS).to(seq.dtype)
    return reclamp_missing(seq)


# --------------------------------------------------------------------------- #
# Temporal hole filling ("carry the last valid value forward")
# --------------------------------------------------------------------------- #
FILLABLE_BLOCKS = {
    "lh":   (LH_LO, LH_HI, F_LH),
    "rh":   (RH_LO, RH_HI, F_RH),
    "arms": (ARM_LO, ARM_HI, F_ARMS),
    "face": (FACE_LO, FACE_HI, F_FACE),
}


def fill_missing_temporal(
    seq: torch.Tensor,
    mode: str = "hold",
    modalities: Sequence[str] = ("lh", "rh"),
    min_presence: float = 0.2,
    set_flags: bool = True,
) -> torch.Tensor:
    """Fill the holes of a clip with the LAST VALID value of that landmark.

    A landmark that MediaPipe fails to detect in a single frame is currently
    encoded as an all-zero block.  The model then sees the whole skeleton
    "teleport" to the origin for that frame, which reads as a real (but
    impossible) movement and confuses both the temporal encoder and the
    geometric augmentations.

    Instead the missing block is filled with the value the landmark had in the
    last frame where it WAS detected ("hold" = carry forward), or with the
    linear interpolation between the previous and the next detection
    ("interpolate").  The result is a continuous trajectory, which is what the
    sign actually looked like.

    ``mode``
        ``"hold"``       - carry the last valid value forward (default)
        ``"interpolate"``- linear interpolation between the surrounding frames
        ``"none"``       - keep the raw corpus encoding (zero blocks)

    ``modalities``
        Which blocks may be filled.  The default is the two hands: they are the
        ones that drop out in the middle of a sign.  The FACE is excluded on
        purpose - it is dead in the pretraining corpus (present in 0.45% of the
        frames), so filling it would put values into 21 input dimensions the
        backbone has never seen.  The ARMS are present in 100% of the word
        frames, so there is nothing to fill.

    ``min_presence``
        A modality is only filled when it is present in at least this fraction
        of the clip - otherwise the "filled" value is a wild extrapolation.

    ``set_flags``
        When True the presence flag of a filled frame is set to 1, so the
        "flag present <=> geometry non-zero" invariant of the corpus is kept.
    """
    if mode in (None, "none", ""):
        return seq
    seq = as_sequence(seq).clone()
    T = seq.shape[0]
    if T == 0:
        return seq

    arange = torch.arange(T)
    for name in modalities:
        if name not in FILLABLE_BLOCKS:
            raise KeyError(f"Unknown modality '{name}' - use one of {list(FILLABLE_BLOCKS)}")
        lo, hi, flag = FILLABLE_BLOCKS[name]

        present = seq[:, flag] >= 0.5
        n_present = int(present.sum())
        if n_present == 0 or n_present < min_presence * T:
            continue

        block = seq[:, lo:hi]
        missing = ~present

        if mode == "hold":
            last = torch.cummax(torch.where(present, arange, torch.full_like(arange, -1)),
                                dim=0)[0]
            nxt = torch.flip(
                torch.cummin(
                    torch.flip(torch.where(present, arange, torch.full_like(arange, T)), [0]),
                    dim=0)[0],
                [0],
            )
            fwd_ok = last >= 0
            idx = torch.where(fwd_ok, last, nxt.clamp(min=0, max=T - 1))
            filled = block[idx]
        elif mode == "interpolate":
            import numpy as np

            present_np = present.numpy()
            idx_np = np.where(present_np)[0]
            x = np.arange(T)
            values = block.detach().cpu().numpy()
            interp = np.empty_like(values)
            for c in range(values.shape[1]):
                interp[:, c] = np.interp(x, idx_np, values[idx_np, c])
            filled = torch.from_numpy(interp).to(seq.dtype)
            fwd_ok = torch.ones(T, dtype=torch.bool)
        else:
            raise ValueError(f"Unknown fill mode '{mode}'")

        need = missing & fwd_ok
        if not bool(need.any()):
            continue

        out = block.clone()
        out[need] = filled[need]
        seq[:, lo:hi] = out
        if set_flags:
            seq[need, flag] = 1.0

    return seq


class TemporalPoseTracker:
    """Online version of :func:`fill_missing_temporal` for the live stream.

    The webcam pipeline produces the frames one by one, so a hole cannot be
    filled with a FUTURE detection.  The tracker keeps the last valid value of
    every modality and re-uses it while the modality is missing, which is
    exactly the "hold" behaviour of the offline filler.
    """

    def __init__(self, modalities: Sequence[str] = ("lh", "rh"), max_age: int = 30):
        self.modalities = tuple(modalities)
        self.max_age = max_age
        self._last: Dict[str, Optional[torch.Tensor]] = {m: None for m in self.modalities}
        self._age: Dict[str, int] = {m: 0 for m in self.modalities}

    def reset(self) -> None:
        for m in self.modalities:
            self._last[m] = None
            self._age[m] = 0

    def update(self, frame: torch.Tensor) -> torch.Tensor:
        """Fill the missing modalities of a single ``(163,)`` frame."""
        frame = as_sequence(frame)[0]
        out = frame.clone()
        for name in self.modalities:
            lo, hi, flag = FILLABLE_BLOCKS[name]
            if float(frame[flag]) >= 0.5:
                self._last[name] = frame[lo:hi].clone()
                self._age[name] = 0
            else:
                last = self._last[name]
                self._age[name] += 1
                if last is not None and self._age[name] <= self.max_age:
                    out[lo:hi] = last
                    out[flag] = 1.0
        return out



# --------------------------------------------------------------------------- #
# The hands-only layout used by the YAP segmentor
# --------------------------------------------------------------------------- #
# The segmentor works on the two MediaPipe hands alone: (T, 42, 3) = 21
# landmarks per hand.  The augmentations below reuse the SAME primitives as
# the 163-dim pipeline (so there is only ONE implementation of "mirror",
# "tilt", "shift" and "speed"), by lifting the hands into the canonical
# layout, augmenting there and dropping them back.

def hands_to_canonical(pose: torch.Tensor) -> torch.Tensor:
    """(T, 42, 3) hands -> (T, 163) with the hands in place and the flags set."""
    x = pose.new_zeros(pose.shape[0], 163)
    x[:, :63] = pose[:, :21].reshape(pose.shape[0], 63)
    x[:, 63:126] = pose[:, 21:42].reshape(pose.shape[0], 63)
    x[:, 159] = 1.0     # the left hand is present
    x[:, 160] = 1.0     # the right hand is present
    return x


def canonical_to_hands(x: torch.Tensor) -> torch.Tensor:
    """(T, 163) -> (T, 42, 3): the two hands, in MediaPipe order."""
    T = x.shape[0]
    return torch.cat([x[:, :63].reshape(T, 21, 3), x[:, 63:126].reshape(T, 21, 3)], dim=1)


# --------------------------------------------------------------------------- #
# The corpus layout used by the segmentor (J = 53)
# --------------------------------------------------------------------------- #
# landmarks  0..20  the left hand  | 21..41 the right hand
#            42..45  the shoulders + the elbows | 46..52 the face
CORPUS_JOINT_SLICES = {
    "left_hand": (0, 21),
    "right_hand": (21, 42),
    "upper_body": (42, 46),
    "face": (46, 53),
}


def pose_to_canonical(pose: torch.Tensor) -> torch.Tensor:
    """(T, J, 3) -> (T, 163) for J = 42 (the hands) or J = 53 (the corpus).

    The corpus layout of J = 53 is EXACTLY the part order of
    ``YAPConfig.corpus()``, so the landmarks are copied verbatim.
    """
    T, J = pose.shape[0], pose.shape[1]
    x = pose.new_zeros(T, 163)
    if J == 53:
        x[:, :63] = pose[:, 0:21].reshape(T, 63)
        x[:, 63:126] = pose[:, 21:42].reshape(T, 63)
        x[:, 126:138] = pose[:, 42:46].reshape(T, 12)
        x[:, 138:159] = pose[:, 46:53].reshape(T, 21)
        x[:, 159:163] = 1.0          # all four modalities are present
    else:                            # the hands only
        x[:, :63] = pose[:, 0:21].reshape(T, 63)
        x[:, 63:126] = pose[:, 21:42].reshape(T, 63)
        x[:, 159] = 1.0
        x[:, 160] = 1.0
    return x


def canonical_to_pose(x: torch.Tensor, n_joints: int = 53) -> torch.Tensor:
    """(T, 163) -> (T, n_joints, 3): the corpus landmarks, in the corpus order."""
    T = x.shape[0]
    if n_joints == 53:
        return torch.cat([
            x[:, :63].reshape(T, 21, 3),
            x[:, 63:126].reshape(T, 21, 3),
            x[:, 126:138].reshape(T, 4, 3),
            x[:, 138:159].reshape(T, 7, 3),
        ], dim=1)
    return torch.cat([x[:, :63].reshape(T, 21, 3), x[:, 63:126].reshape(T, 21, 3)], dim=1)


def augment_hands_pose(
    pose: torch.Tensor,
    cfg: Optional[AugmentConfig] = None,
    rng: Optional[random.Random] = None,
) -> torch.Tensor:
    """The canonical augmentation set applied to a hands-only (or corpus) pose.

    ``pose``: (T, 42, 3) — the two hands — or (T, 53, 3) — the corpus layout.
    Returns the same shape.  The rotation pivots on the centroid of the visible
    landmarks (there are no shoulder points in the hands-only layout), and the
    mirror swaps the two hands and reflects the X axis around that centroid.
    """
    cfg = cfg or DEFAULT_AUGMENT
    rng = rng or random
    n_joints = pose.shape[1]
    x = pose_to_canonical(pose)
    x = augment_pose_sequence(x, cfg, rng=rng)
    return canonical_to_pose(x, n_joints)


def augment_hands_batch(
    pose: torch.Tensor,
    lengths: torch.Tensor,
    cfg: Optional[AugmentConfig] = None,
    rng: Optional[random.Random] = None,
) -> torch.Tensor:
    """The canonical augmentation applied to a padded batch of hands-only poses.

    ``pose``: (B, T, J, 3), ``lengths``: (B,) the valid length of every clip.
    Every clip is augmented on its own (the padded frames are dropped and put
    back) and the length is RESTORED afterwards: the temporal augmentations
    (speed, time shift) change the number of frames, and a padded batch needs
    every clip to keep its slot.
    """
    cfg = cfg or DEFAULT_AUGMENT
    rng = rng or random
    B, T = pose.shape[0], pose.shape[1]
    out = pose.clone()
    for b in range(B):
        n = int(lengths[b].item()) if lengths is not None else T
        n = max(1, min(n, T))
        aug = augment_hands_pose(pose[b, :n], cfg, rng=rng)
        if aug.shape[0] != n:
            aug = _resample_time(aug, n)
        out[b, :n] = aug
    return out


def _resample_time(x: torch.Tensor, n: int) -> torch.Tensor:
    """Resample a (T, ...) clip to n frames (linear on the coordinates)."""
    T = x.shape[0]
    if T == n:
        return x
    flat = x.reshape(T, -1).T.unsqueeze(0)                 # (1, C, T)
    out = F.interpolate(flat, size=n, mode="linear", align_corners=True)
    return out.squeeze(0).T.reshape(n, *x.shape[1:])


# --------------------------------------------------------------------------- #
# Augmentation configuration + composite pipeline
# --------------------------------------------------------------------------- #
@dataclass
class AugmentConfig:
    """Probabilities and magnitudes of the augmentation set.

    The four operations the pipeline is built around are the ones the project
    needs everywhere:

      * ``flip``      - mirror the pose (left-handed signer), all L/R pairs swap
      * ``tilt``      - rotate around the neck
      * ``shift``     - translate left / right, bounded so nothing vanishes
      * ``speed``     - speed the clip up or slow it down
    """

    strength: float = 1.0

    flip_prob: float = 0.50
    flip_swap_prob: float = 1.0

    tilt_prob: float = 0.50
    tilt_deg: float = 12.0

    shift_prob: float = 0.50
    shift_dx: float = 0.05
    shift_dy: float = 0.02

    scale_prob: float = 0.00
    scale_lo: float = 0.90
    scale_hi: float = 1.10

    speed_prob: float = 0.50
    speed_lo: float = 0.80
    speed_hi: float = 1.25

    time_shift_prob: float = 0.20
    time_shift_max: int = 3

    mask_prob: float = 0.20
    mask_max: int = 3
    mask_count: int = 2

    noise_std: float = 0.0

    def scaled(self, strength: float) -> "AugmentConfig":
        """Return a copy whose magnitudes are scaled by ``strength`` (0 = off)."""
        return replace(
            self,
            strength=strength,
            flip_prob=self.flip_prob * strength,
            tilt_prob=self.tilt_prob * strength,
            shift_prob=self.shift_prob * strength,
            speed_prob=self.speed_prob * strength,
            time_shift_prob=self.time_shift_prob * strength,
            mask_prob=self.mask_prob * strength,
            tilt_deg=self.tilt_deg * strength,
            shift_dx=self.shift_dx * strength,
            shift_dy=self.shift_dy * strength,
            noise_std=self.noise_std * strength,
        )


DEFAULT_AUGMENT = AugmentConfig()


def augment_pose_sequence(seq: torch.Tensor, cfg: Optional[AugmentConfig] = None,
                          rng: Optional[random.Random] = None) -> torch.Tensor:
    """Apply the full augmentation set to a ``(T, 163)`` clip.

    Order: mirror -> tilt -> shift -> speed -> time shift -> temporal mask ->
    jitter.  The geometric steps run before the temporal ones so the temporal
    resampling never mixes coordinates from two different geometries.
    """
    cfg = cfg or DEFAULT_AUGMENT
    rng = rng or random
    seq = as_sequence(seq)

    if cfg.flip_prob > 0 and rng.random() < cfg.flip_prob:
        seq = mirror_pose(seq)
    if cfg.tilt_prob > 0 and rng.random() < cfg.tilt_prob:
        seq = tilt_pose(seq, cfg.tilt_deg, rng)
    if cfg.shift_prob > 0 and rng.random() < cfg.shift_prob:
        seq = shift_pose(seq, cfg.shift_dx, cfg.shift_dy, rng)
    if cfg.speed_prob > 0 and rng.random() < cfg.speed_prob:
        seq = time_stretch(seq, cfg.speed_lo, cfg.speed_hi, rng)
    if cfg.time_shift_prob > 0 and rng.random() < cfg.time_shift_prob:
        seq = temporal_shift(seq, cfg.time_shift_max, rng)
    if cfg.mask_prob > 0 and rng.random() < cfg.mask_prob:
        seq = temporal_mask(seq, cfg.mask_max, cfg.mask_count, rng)
    if cfg.noise_std > 0 and rng.random() < 0.5:
        seq = add_noise(seq, cfg.noise_std, rng)

    return seq.contiguous()


# --------------------------------------------------------------------------- #
# Backwards-compatible aliases (the historic names used across the trainers)
# --------------------------------------------------------------------------- #
aug_flip = mirror_pose
aug_tilt = tilt_pose
aug_shift = shift_pose
aug_speed = time_stretch
aug_temporal_mask = temporal_mask


def augment(seq: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """Historic composite used by the CTC sentence trainers.

    Kept as a thin wrapper over :func:`augment_pose_sequence` so that all the
    trainers share ONE implementation.
    """
    cfg = DEFAULT_AUGMENT.scaled(strength)
    return augment_pose_sequence(seq, cfg)
