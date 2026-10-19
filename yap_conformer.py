"""
==============================================================================
YAP  ("You Adapt Progressively")  +  YAPping conFORMER
==============================================================================
ONE FILE, written so that another AI agent (or a human) can pick it up cold.
Read this docstring first; every class/function below also explains itself.

------------------------------------------------------------------------------
0. WHAT THIS IS, IN PLAIN WORDS
------------------------------------------------------------------------------
Input : a LIVE STREAM of "poses": the coordinates of ANY set of tracked points
        (hands, arms/body, face subset, or your own custom points; declared in
        YAPConfig.parts) plus a timestamp. There is NO video/image in this model.
Job 1 : YAP watches the stream one pose at a time, cleans it, expands it into
        richer features, decides WHERE one gesture starts and ends
        ("segmenting"), and squashes each segment into ONE fixed-size vector
        (an "entity"). The variable time length T is gone after this step.
Job 2 : the YAPping conFORMER (a Conformer = transformer blocks + convolution
        blocks) reads the sequence of entities (= a "sentence") and says which
        word each entity is.

YAP itself does NOT recognise words. It only finds boundaries and packages
each segment. The transformer decides what each package is.

GLOSSARY
  pose      : one snapshot of all landmark coordinates at one moment
              (J joints x 3 coords). "Frame" and "pose" are used for the same thing.
  t / dt    : real timestamp in SECONDS / seconds since the previous pose.
              The maths always uses the real dt, never an assumed frame rate.
  segment   : [start_idx, end_idx) range of poses believed to be one gesture.
  entity    : the single vector YAP produces for one segment.
  presence  : per-pose probability that "a gesture is happening right now".
  end prob  : per-pose probability that "a gesture ends right here" (a cut).
  pre-roll  : a few poses BEFORE the segment start that are also fed to the
              pooling step, flagged as "context" (this is our 50% overlap idea).
  trust     : per-joint weight = tracker confidence x (joint present ? 1 : 0).

------------------------------------------------------------------------------
1. DATA FLOW (what happens to every incoming pose)
------------------------------------------------------------------------------
 raw pose (J,3) + conf (J) + valid (J) + dt
   |
   v  PoseCleaner        (no learnable params)
        - missing joints are marked, bridged briefly by constant-velocity
          prediction, never silently invented for long
        - One-Euro-style adaptive smoothing (slow motion smoothed hard, fast
          motion passes through)
        - bone-length projection (bones keep their length RELATIVE to hand
          size, so moving toward the camera is not "corrected away")
   v  FeatureExpander    (no learnable params)
        - palm-frame canonicalisation (hand SHAPE independent of position /
          rotation / distance) kept SEPARATE from ROOT motion (position and
          orientation in camera frame; needed for direction-dependent signs)
        - bone directions, joint-angle cosines, fingertip distances,
          velocities, residual (raw - cleaned), trust, gap age
        - "core" block always full strength; "rich" block scaled by the
          PREVIOUS step's presence ("adaptive expansion")
   v  lift (LayerNorm + Linear)     -> d_model vector
   v  CausalConvStem  = the "CNN head": depthwise-separable 1-D conv over the
        last k poses. Causal (never looks at the future) so it works live.
   v  MemoryLayer x n   each layer keeps a bank of leaky integrators
        (memory over many time-scales) AND emits a confidence ("assurity of
        presence") that is passed to the next layer.
   v  heads: presence logit, end logit, per-layer confidences, summary feature
   v  SegmentTracker     (plain python state machine, hysteresis)
        decides start / end from presence & end probabilities
   v  HiPPO-LegS pooling (encode_spans)
        folds all poses of the segment (+pre-roll) into a fixed-size state
        with real timestamps, then a readout makes the ENTITY vector.
   v  entities -> YAPpingConformer -> word logits

------------------------------------------------------------------------------
2. KEY DESIGN DECISIONS (and why) -- do not silently change these
------------------------------------------------------------------------------
 a) Real timestamps everywhere. MediaPipe frames can drop / be uneven.
 b) Missing landmark = explicit "missing" flag (valid=False). We never feed
    fake data without telling the model. Short gaps are tolerated by the
    segmenter (gap_tolerance_s) so a tracker dropout does not split one sign.
 c) Presence and "end" are SEPARATE heads (YOLO idea: objectness vs class).
 d) Hysteresis (on_thr > off_thr + hold times) so scores that bounce around a
    threshold do not flicker segments on/off.
 e) Hard safety nets: max_seg_s (force-cut) and still_flush_s (3-4 s of
    stillness = ultimate clear moment: end segment, RESET ALL MEMORY).
 f) Overlap: the pre-roll length = overlap_fraction * previous segment length,
    clamped to [preroll_min_s, preroll_max_s]. Pre-roll frames get a learned
    "context" flag so the entity can tell lead-in from the gesture itself.
    The 0.5 / 0.1 / 0.5 numbers are GUESSES, not validated.
 g) Pooling is HiPPO-LegS (Gu et al. 2020): an online, fixed-size
    approximation of the whole history by Legendre polynomials. Order is
    preserved (unlike plain averaging). Backward-Euler step handles uneven dt.
    A step with dt = 0 is an exact no-op, which is how we batch segments of
    different length (pad with dt = 0).
 h) The segment is pooled AFTER its end is known (not incrementally), because
    the end index is back-dated (e.g. presence dropped 0.2 s ago). This also
    guarantees live == offline results (see YAPStream).
 i) Training and live use the SAME step function and SAME tracker. This was a
    deliberate fix for the "train/live mismatch" risk.
 j) Frame-level memory (MemoryLayer) is a simple multi-time-scale leaky
    integrator, NOT an RNN/SSM kitchen sink. Easy to analyse, easy to swap.
 l) CONFIGURABLE BODY LAYOUT (the "one class + one config" idea): everything about
    WHICH points are tracked lives in YAPConfig.parts (list of PartSpec: name,
    num_joints, kind, parents, anchor, scale_pair, motion) and YAPConfig.body_ref.
    No other code assumes a layout. kind="hand" adds palm-frame features (needs the
    21-point MediaPipe hand); kind="generic" works for any points. Missing parts are
    handled per part (valid=False), so a vanished hand never silences the face.
    CHANGING the layout changes the input size => weights must be RETRAINED.
 k) "CNN head" interpretation: we put the small causal conv at the FRONT of
    the model (head = start). The Conformer blocks additionally contain their
    own depthwise conv modules. If you meant a conv classifier on top, add a
    Conv1d over the token dimension after the Conformer blocks.

------------------------------------------------------------------------------
3. TENSOR CONVENTIONS
------------------------------------------------------------------------------
 B batch, T poses, J = cfg.total_joints = sum of num_joints over cfg.parts.
 The J axis is the parts CONCATENATED in cfg.parts order (cfg.part_offsets() gives
 where each part starts; assemble_parts() builds the tensors from per-part data).
 Default config = 2 MediaPipe hands (J=42). YAPConfig.holistic() = 2 hands + 9-point
 upper body + 28-point face subset + shoulder body frame (J=79).
 pose  (B,T,J,3) float  coordinates (x,y normalised image units, z relative)
 conf  (B,T,J)   float  tracker confidence in [0,1] (use 1 if unavailable)
 valid (B,T,J)   bool   False = landmark missing this frame (coords ignored)
 t     (B,T)     float  seconds, monotonic non-decreasing
 lengths (B,)    int    true number of poses when padded
 A part that is absent (e.g. a hand out of view): valid=False for all its joints.
 All parts must share ONE clock: if your models deliver at different moments,
 resample/align them to one timestamp per pose BEFORE assemble_parts().

------------------------------------------------------------------------------
4. TRAINING CURRICULUM (agreed in discussion; implemented only as helpers)
------------------------------------------------------------------------------
 Stage 1  single-word clips: exactly 1 segment. Random idle padding, random
          start moment, speed jitter. Use presence_targets()/end_targets() with
          focal_presence_loss() + end_bce, plus layer_conf_loss() (deep supervision).
 Stage 2  stitched sentences (several word clips glued, joins blended): known
          boundaries and counts. Same losses + word cross-entropy on entities.
 Stage 3  real sentences: only the word COUNT (and word list) is known.
          soft_count_loss() + min_gap_penalty() + dip_prior_loss() on YAP,
          sentence_ctc_loss() on the Conformer.
 Later    clean-vs-noisy consistency training (teacher = clean copy, student =
          noisy copy with REALISTIC MediaPipe noise: dropouts, jitter, jumps,
          left/right swaps, wrong depth, speed changes, idle padding). To stop
          collapse, also require both to predict the word label.

------------------------------------------------------------------------------
5. KNOWN WEAKNESSES / OPEN ITEMS (honest list)
------------------------------------------------------------------------------
 1. Firing decision is the single point of failure; a wrong early/late cut
    cascades. Mitigations here: hysteresis, hold times, pre-roll context,
    CTC at sentence level. Not solved.
 2. Hard segmentation hides ambiguity from the transformer (no confidence is
    passed besides meta features). Two simultaneous gestures are not
    representable by one entity stream.
 3. Continuous signing with no pause: stillness cue absent; we rely on the
    end head and the dip prior. Merged signs are possible.
 4. Trust = tracker confidence, but MediaPipe can be confidently WRONG (hand
    swaps, jumps). Not handled yet (needs the noise augmenter + checks).
 5. Palm frame degenerates when the palm is edge-on; we down-weight (palm_w)
    but do not repair it.
 6. Entity depends on the exact start/end; train with boundary jitter.
 7. All numeric defaults (thresholds, tau range, orders) are unvalidated.
 8. Latency adds up: off_hold_s + (1 frame for cut peak) + any look-ahead.
 9. The per-step Python loop is slow for long training sequences (see
    cfg.tbptt). Memory layers are linear recurrences, so a parallel scan could
    replace the loop later.
 10. Word count != sign count for some words (compounds, fingerspelling).
 11. PoseCleaner is NOT a proven denoiser. Measured on synthetic fake hands
     (RMS error vs truth, normalised units): with jitter 0.01 the cleaned pose is
     about break-even on slow motion (0.0094-0.0121 vs raw 0.0100) and WORSE on
     fast motion (0.035-0.10 vs raw 0.0100) because any causal smoother lags a
     flick. With jitter 0.03 it gains ~20% on slow motion (0.0244 vs 0.0301) but
     is still worse on fast motion (0.044 vs 0.030). Its solid benefits are
     dropout bridging, snapping back after gaps, and bone-length projection.
     This is why FeatureExpander ALSO passes the residual (raw - clean) and the
     velocities: the network can recover fast motion. Tune beta/min_cutoff_hz on
     REAL MediaPipe data (or replace with a learned or zero-phase filter offline).
     A 6-pose landmark dropout was bridged with ~2x the jitter error (0.021).
 12. LIMITS of the "one for all" config. Works for: any FIXED set of tracked points
     (2D or 3D; for 2D set z=0), any bone tree, any number of points per part, one
     shared clock. Does NOT cover: a VARIABLE number of tracked entities per frame
     (e.g. several people), non-coordinate signals (audio, IMU), unsynchronised
     sources, or reusing weights across layouts. Generic parts get weaker features
     than hands (no rotation canonicalisation). Segmenter thresholds, stillness
     speed and cleaner settings were designed around human gestures; retune for
     other subjects. FACE_LANDMARKS defaults are UNVERIFIED: check them against your
     MediaPipe version.

------------------------------------------------------------------------------
6. QUICK USAGE
------------------------------------------------------------------------------
  cfg   = YAPConfig()                      # default: 2 hands
  cfg   = YAPConfig.holistic()             # 2 hands + upper body + face subset + body frame
  cfg   = YAPConfig(parts=[PartSpec("obj", 5), hand_part("right_hand")])   # anything you track
  pose, conf, valid = assemble_parts(cfg, {"right_hand": (hand_xyz, hand_conf, None), ...})
  model = YAPpingConformer(cfg, ConformerConfig(num_words=100))
  out   = model(pose, conf, valid, t, lengths)          # YAP finds segments
  out   = model(pose, conf, valid, t, lengths, segments=gt_segments)  # teacher forcing
  stream = YAPStream(model.yap)                          # live, one pose at a time
  for each incoming pose: entities = stream.push(pose_JJ3, conf_J, valid_J, t_seconds)
  at end of recording:    entities += stream.flush()

FILE MAP (search for these banners)
  BODY PARTS + CONFIG -> PoseCleaner -> FeatureExpander -> CausalConvStem ->
  MemoryLayer -> HiPPOLegS -> SEGMENTATION (Segment, SegmentTracker) ->
  YAPCore -> YAP -> YAPStream -> LOSSES / TARGET HELPERS -> CONFORMER -> __main__ tests
"""
from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# BODY PARTS + CONFIG
# ==============================================================================
# The pose vector fed to YAP is a CONCATENATION of "parts" (left hand, right hand,
# upper body, face subset, ...). Which parts exist, how many points each has and
# how the points connect is described by a list of PartSpec in YAPConfig.parts.
# Nothing else in the code assumes a particular body layout.

# Parent joint of each MediaPipe hand landmark (index -> parent). 0 = wrist.
# Every parent index is smaller than its child index, so "process in index order"
# is also "process root-outward".
MEDIAPIPE_HAND_PARENTS = [-1, 0, 1, 2, 3,      # thumb  1-4
                          0, 5, 6, 7,          # index  5-8
                          0, 9, 10, 11,        # middle 9-12
                          0, 13, 14, 15,       # ring   13-16
                          0, 17, 18, 19]       # pinky  17-20
# NOTE: list above has 21 entries: 1 wrist + 4 thumb + 4*4 fingers.
FINGERTIPS = [4, 8, 12, 16, 20]
WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 0, 5, 9, 17


@dataclass
class PartSpec:
    """
    Describes ONE body part inside the concatenated pose vector.

      name        unique name, used by body_ref and assemble_parts().
      num_joints  how many points this part contributes (any number you like).
      kind        "hand"    -> 21-point MediaPipe hand; gets palm-frame features
                               (rotation-invariant shape, fingertip distances).
                  "generic" -> any other set of points (arms/torso, face subset, ...);
                               gets translation+scale-invariant features only.
      parents     parents[j] = index (INSIDE this part) of joint j's parent, -1 for a root.
                  A parent must have a smaller index than its child. None = no bones
                  (all roots), e.g. a face subset. Bones are used for bone-length
                  projection, bone-direction and joint-angle features.
      anchor      joint used as the part's origin for its local coordinates and for the
                  body-relative position feature.
      scale_pair  two joints (inside this part) whose distance is the part's SIZE UNIT
                  (hand: wrist->middle base, torso: shoulder width, face: outer eye corners).
                  All local features are divided by it so camera distance does not matter.
      motion      True if this part's movement counts for the "is the person still?"
                  check and for "is something tracked?" (hands/arms yes, face no).
      landmark_indices  OPTIONAL, informational only: which indices of the source model
                  (e.g. MediaPipe Pose/FaceMesh) your adapter should copy into this part,
                  in order. The model itself never reads it.
    """
    name: str
    num_joints: int
    kind: str = "generic"
    parents: Optional[List[int]] = None
    anchor: int = 0
    scale_pair: Tuple[int, int] = (0, 1)
    motion: bool = True
    landmark_indices: Optional[List[int]] = None

    def __post_init__(self):
        n = self.num_joints
        if self.kind not in ("hand", "generic"):
            raise ValueError(f"part {self.name}: kind must be 'hand' or 'generic'")
        if self.parents is None:
            self.parents = [-1] * n
        if len(self.parents) != n:
            raise ValueError(f"part {self.name}: parents has {len(self.parents)} entries, expected {n}")
        for j, p in enumerate(self.parents):
            if p >= j:
                raise ValueError(f"part {self.name}: parent of joint {j} must have a smaller index (got {p})")
        a, b = self.scale_pair
        if not (0 <= a < n and 0 <= b < n and a != b and 0 <= self.anchor < n):
            raise ValueError(f"part {self.name}: bad anchor/scale_pair for {n} joints")
        if self.kind == "hand" and not (n == 21 and self.parents == MEDIAPIPE_HAND_PARENTS):
            raise ValueError(f"part {self.name}: kind='hand' needs the 21-point MediaPipe hand layout")


def hand_part(name: str) -> PartSpec:
    """Preset: one 21-point MediaPipe hand."""
    return PartSpec(name, 21, "hand", list(MEDIAPIPE_HAND_PARENTS), anchor=WRIST,
                    scale_pair=(WRIST, MIDDLE_MCP), motion=True)


# MediaPipe Pose indices: 11/12 shoulders, 13/14 elbows, 15/16 wrists, 23/24 hips, 0 nose.
UPPER_BODY_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24, 0]


def upper_body_part(name: str = "upper_body") -> PartSpec:
    """Preset: shoulders, elbows, wrists, hips, nose (9 points). Arms are chains from the shoulders."""
    return PartSpec(name, 9, "generic", parents=[-1, -1, 0, 1, 2, 3, -1, -1, -1], anchor=0,
                    scale_pair=(0, 1), motion=True, landmark_indices=list(UPPER_BODY_LANDMARKS))


# 28 commonly used MediaPipe FaceMesh indices: brows(10), eye corners/lids(8), mouth(8), nose tip, chin.
# UNVERIFIED DEFAULT: check these against your MediaPipe version's face-mesh map before relying on them.
FACE_LANDMARKS = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300,      # eyebrows
                  33, 133, 159, 145, 362, 263, 386, 374,              # eyes (corners, upper/lower lid)
                  61, 291, 0, 17, 13, 14, 78, 308,                    # mouth (corners, outer/inner lips)
                  1, 152]                                             # nose tip, chin


def face_part(name: str = "face") -> PartSpec:
    """Preset: 28-point face subset, no bones, scale = distance between the outer eye corners."""
    return PartSpec(name, len(FACE_LANDMARKS), "generic", parents=None, anchor=FACE_LANDMARKS.index(1),
                    scale_pair=(FACE_LANDMARKS.index(33), FACE_LANDMARKS.index(263)), motion=False,
                    landmark_indices=list(FACE_LANDMARKS))


def _default_parts() -> List[PartSpec]:
    return [hand_part("left_hand"), hand_part("right_hand")]


@dataclass
class YAPConfig:
    """All YAP knobs in one place. Units: seconds, Hz, normalised coordinates."""

    # --- body layout ----------------------------------------------------------
    parts: List[PartSpec] = field(default_factory=_default_parts)   # order = order in the pose vector
    # Two joints ((part, joint), (part, joint)) that define the BODY FRAME: origin = their midpoint,
    # unit = their distance (shoulders). With it every part's position is expressed relative to the
    # body, so features survive moving/leaning/standing nearer the camera and "hand near chin"
    # becomes visible. None = camera frame (old behaviour).
    body_ref: Optional[Tuple[Tuple[str, int], Tuple[str, int]]] = None

    # --- time -----------------------------------------------------------------
    nominal_fps: float = 30.0     # ONLY used for the very first dt of a stream

    # --- PoseCleaner ----------------------------------------------------------
    min_conf: float = 0.05        # conf below this counts as "missing"
    min_cutoff_hz: float = 1.0    # One-Euro: smoothing cutoff at rest
    beta: float = 4.0             # One-Euro: cutoff growth per unit speed (coords/s).
                                  # 4.0 suits normalised-image coords; retune if you rescale
    d_cutoff_hz: float = 1.0      # cutoff for the velocity estimate itself
    max_gap_predict_s: float = 0.25   # bridge a missing joint by prediction up to this long
    velocity_decay_s: float = 0.15    # predicted velocity decays with this time constant
    use_bone_projection: bool = True
    bone_ratio_ema: float = 0.98  # running average of bone_length / part_scale

    # --- FeatureExpander ------------------------------------------------------
    expand_floor: float = 0.25    # "rich" features always keep >= this fraction
    hand_ok_frac: float = 0.3     # a part counts as present if >30% of its joints are trusted
    vel_clip: float = 10.0        # clip velocity features (part-scales per second)

    # --- network widths -------------------------------------------------------
    d_model: int = 128
    cnn_kernel: int = 5           # causal conv "CNN head" kernel (in poses)
    n_mem_layers: int = 3
    ffn_mult: int = 2
    mem_tau_min_s: float = 0.1    # shortest memory time constant
    mem_tau_max_s: float = 2.0    # longest memory time constant

    # --- pooling into one entity ---------------------------------------------
    summary_channels: int = 32    # channels fed to HiPPO
    hippo_order: int = 16         # Legendre coefficients per channel
    hippo_t0_s: float = 0.05      # time offset so 1/t never blows up at segment start
    hold_quality: float = 0.2     # frames with quality below this reuse the last good feature
    entity_dim: int = 256

    # -
    # 
    # 
    # 
    # -- segmentation (SegmentTracker) ---------------------------------------
    on_thr: float = 0.6           # presence >= on_thr (and something tracked) to start
    off_thr: float = 0.4          # presence < off_thr to count as "low"
    on_hold_s: float = 0.06       # presence must stay high this long to start
    off_hold_s: float = 0.20      # presence must stay low this long to end
    gap_tolerance_s: float = 0.25 # tracker dropout shorter than this does NOT count as "low"
    cut_thr: float = 0.7          # end-prob peak above this cuts a segment
    min_seg_s: float = 0.20       # no cut before the segment is this old
    max_seg_s: float = 4.0        # force-cut (also bounds memory)
    still_speed_thr: float = 0.05 # part-scales/s below this = "still" (or nothing tracked)
    still_flush_s: float = 3.5    # still this long = flush + reset ALL memory

    # --- overlap / pre-roll ---------------------------------------------------
    overlap_fraction: float = 0.5 # pre-roll = this * previous segment duration ...
    preroll_min_s: float = 0.10   # ... clamped to this minimum
    preroll_max_s: float = 0.50   # ... and this maximum

    # --- training -------------------------------------------------------------
    tbptt: Optional[int] = None   # detach recurrent state every N steps (None = never)

    def __post_init__(self):
        names = [p.name for p in self.parts]
        if len(set(names)) != len(names):
            raise ValueError(f"part names must be unique, got {names}")
        if not self.parts:
            raise ValueError("need at least one part")
        if self.body_ref is not None:
            for pname, j in self.body_ref:
                if pname not in names or not (0 <= j < self.parts[names.index(pname)].num_joints):
                    raise ValueError(f"body_ref {pname}[{j}] does not exist")
        if not any(p.motion for p in self.parts):
            raise ValueError("at least one part needs motion=True (used for stillness / tracked checks)")

    # ---- layout helpers ----
    @property
    def total_joints(self) -> int:
        """J = number of points in the concatenated pose vector."""
        return sum(p.num_joints for p in self.parts)

    def part_offsets(self) -> Dict[str, int]:
        """name -> index of the part's first joint inside the concatenated vector."""
        offs, o = {}, 0
        for p in self.parts:
            offs[p.name] = o
            o += p.num_joints
        return offs

    def joint_index(self, part: str, j: int) -> int:
        """Global index of joint j of the named part."""
        return self.part_offsets()[part] + j

    @classmethod
    def holistic(cls, **kw) -> "YAPConfig":
        """Preset: both hands + upper body + face subset, shoulder-based body frame."""
        return cls(parts=[hand_part("left_hand"), hand_part("right_hand"), upper_body_part(), face_part()],
                   body_ref=(("upper_body", 0), ("upper_body", 1)), **kw)


def assemble_parts(cfg: YAPConfig, data: Dict[str, Tuple]):
    """
    Build the concatenated (pose, conf, valid) tensors from per-part data, so you can feed
    YAP from several different models.
        data[name] = (pose (..., n, 3), conf (..., n) or None, valid (..., n) or None)
    Leading dims (...) are whatever you use (e.g. (B,T) or none for a single live frame) but
    must be the same for all parts. A part you do not provide is marked valid=False.
    NOTE: different models usually deliver at different moments; resample/align them to ONE
    shared clock (one timestamp per pose) BEFORE calling this.
    """
    unknown = set(data) - {p.name for p in cfg.parts}
    if unknown:
        raise ValueError(f"unknown part names: {sorted(unknown)}")
    first = next(iter(data.values()))[0]
    lead, J = first.shape[:-2], cfg.total_joints
    pose = first.new_zeros(*lead, J, 3)
    conf = first.new_zeros(*lead, J)
    valid = torch.zeros(*lead, J, dtype=torch.bool, device=first.device)
    offs = cfg.part_offsets()
    for spec in cfg.parts:
        if spec.name not in data:
            continue
        p, c, v = data[spec.name]
        if p.shape[-2] != spec.num_joints:
            raise ValueError(f"part {spec.name}: got {p.shape[-2]} joints, expected {spec.num_joints}")
        o, n = offs[spec.name], spec.num_joints
        pose[..., o:o + n, :] = p
        conf[..., o:o + n] = 1.0 if c is None else c
        valid[..., o:o + n] = True if v is None else v
    return pose, conf, valid


def frame_dts(t: torch.Tensor, fps: float) -> torch.Tensor:
    """
    dt per pose from timestamps t (B,T) -> (B,T). First pose gets 1/fps.
    Clamped to [1 ms, 0.5 s] so a stalled stream cannot create absurd steps.
    THE SAME RULE is used by training (forward_frames) and live (YAPStream).
    """
    dt = torch.empty_like(t)
    dt[:, 0] = 1.0 / fps
    dt[:, 1:] = (t[:, 1:] - t[:, :-1]).clamp(1e-3, 0.5)
    return dt


# ---- helpers to treat nested recurrent state (dict / list / tensor) uniformly --
def _tree_select(rowmask: torch.Tensor, fresh, cur):
    """Per-batch-row choice: where rowmask is True take `fresh`, else keep `cur`."""
    if isinstance(cur, dict):
        return {k: _tree_select(rowmask, fresh[k], cur[k]) for k in cur}
    if isinstance(cur, list):
        return [_tree_select(rowmask, f, c) for f, c in zip(fresh, cur)]
    m = rowmask.view(-1, *([1] * (cur.dim() - 1)))
    return torch.where(m, fresh, cur)


def _tree_detach(x):
    """Cut the autograd graph through recurrent state (truncated BPTT)."""
    if isinstance(x, dict):
        return {k: _tree_detach(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_tree_detach(v) for v in x]
    return x.detach()


# ==============================================================================
# PoseCleaner  (stabilise: dropouts, jitter, bone lengths)
# ==============================================================================
class PoseCleaner(nn.Module):
    """
    Streaming, per-pose cleaner. No learnable parameters, so it runs under
    torch.no_grad(). State (a dict of tensors, batch on dim 0) is passed in and
    returned. Works for ANY layout of parts (cfg.parts).

    Per joint, per step:
      * ok = valid & conf >= min_conf.
      * If ok: One-Euro-style filter. Velocity is low-passed; the smoothing
        cutoff = min_cutoff + beta*|velocity|, so SLOW motion is smoothed
        strongly and FAST motion (a flick) passes through. The filter blends
        the measurement with a constant-velocity PREDICTION (not the previous
        position) to reduce lag.
      * If not ok: bridge by prediction for at most max_gap_predict_s with
        decaying velocity; after that hold position. The gap age is tracked
        and exposed as a feature so the network knows the data is stale.
      * A joint that was missing longer than max_gap_predict_s snaps straight
        to the new measurement when it reappears (otherwise the filter would
        crawl from a stale position).
    Then bone projection (optional): walking root -> outward, each bone is
    rescaled to (running average of bone_length/part_scale) * current part_scale.
    part_scale = distance between the part's scale_pair joints (hand: wrist ->
    middle-finger base; torso: shoulder width). A bone that IS the scale pair is
    not projected because it defines the scale. Using a RATIO means a body moving
    toward the camera (everything gets bigger) is not "corrected" back.
    Parts without bones (face subset) simply skip projection.
    """

    def __init__(self, cfg: YAPConfig):
        super().__init__()
        self.cfg = cfg
        self.J, self.P = cfg.total_joints, len(cfg.parts)
        parents: List[int] = []
        part_of: List[int] = []
        sa: List[int] = []
        sb: List[int] = []
        off = 0
        for pi, spec in enumerate(cfg.parts):
            parents += [(-1 if p < 0 else p + off) for p in spec.parents]     # global parent indices
            part_of += [pi] * spec.num_joints
            sa.append(off + spec.scale_pair[0])
            sb.append(off + spec.scale_pair[1])
            off += spec.num_joints
        depth = [0] * self.J
        for j, p in enumerate(parents):
            depth[j] = 0 if p < 0 else depth[p] + 1
        self.max_depth = max(depth)
        # one index tensor per depth level (1..max_depth) so we can vectorise per level
        for d in range(1, self.max_depth + 1):
            self.register_buffer(f"lvl{d}", torch.tensor([j for j in range(self.J) if depth[j] == d]))
        self.register_buffer("parent_idx", torch.tensor([max(p, 0) for p in parents]))
        self.register_buffer("part_of_joint", torch.tensor(part_of))
        self.register_buffer("scale_a", torch.tensor(sa))
        self.register_buffer("scale_b", torch.tensor(sb))
        # joints whose bone length we project: have a parent, and are not the scale-defining bone
        pm = torch.tensor([p >= 0 for p in parents])
        for a, b in zip(sa, sb):
            if parents[a] == b:
                pm[a] = False
            if parents[b] == a:
                pm[b] = False
        self.register_buffer("project_mask", pm)

    def init_state(self, B: int, device) -> Dict[str, torch.Tensor]:
        J = self.J
        z3 = torch.zeros(B, J, 3, device=device)
        return dict(
            x=z3.clone(), v=z3.clone(),
            inited=torch.zeros(B, J, dtype=torch.bool, device=device),
            gap=torch.zeros(B, J, device=device),
            ratio=torch.zeros(B, J, device=device),
            ratio_ok=torch.zeros(B, J, dtype=torch.bool, device=device),
        )

    @staticmethod
    def _alpha(dt: torch.Tensor, cutoff) -> torch.Tensor:
        """One-Euro smoothing factor for a given dt and cutoff (Hz)."""
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    @torch.no_grad()
    def step(self, st, pose, conf, valid, dt):
        """
        pose (B,J,3) conf (B,J) valid (B,J) bool dt (B,)
        returns (new_state, dict(x, v, trust, gap)) where
          x (B,J,3) cleaned pose, v (B,J,3) velocity (coords/s),
          trust (B,J) conf*ok, gap (B,J) seconds since last good measurement.
        """
        cfg = self.cfg
        dtc = dt.clamp_min(1e-3)
        dtv = dtc.view(-1, 1, 1)
        ok = valid & (conf >= cfg.min_conf)
        okv = ok.unsqueeze(-1)
        x_prev, v_prev, inited, gap = st["x"], st["v"], st["inited"], st["gap"]

        x_pred = x_prev + v_prev * dtv
        raw_v = (pose - x_prev) / dtv
        a_d = self._alpha(dtc, cfg.d_cutoff_hz).view(-1, 1, 1)
        v_meas = a_d * raw_v + (1 - a_d) * v_prev
        spd = v_meas.norm(dim=-1, keepdim=True)
        cutoff = cfg.min_cutoff_hz + cfg.beta * spd           # adaptive cutoff
        a = self._alpha(dtv, cutoff)
        x_meas = a * pose + (1 - a) * x_pred

        # first sighting, or reappearing after a long gap: snap to measurement
        snap = (ok & (~inited | (gap >= cfg.max_gap_predict_s))).unsqueeze(-1)
        x_ok = torch.where(snap, pose, x_meas)
        v_ok = torch.where(snap, torch.zeros_like(v_meas), v_meas)

        # missing: bridge briefly by prediction, afterwards hold still
        can_pred = (inited & (gap < cfg.max_gap_predict_s)).unsqueeze(-1)
        x_miss = torch.where(can_pred, x_pred, x_prev)
        v_miss = torch.where(can_pred, v_prev * torch.exp(-dtv / cfg.velocity_decay_s), torch.zeros_like(v_prev))

        x_new = torch.where(okv, x_ok, x_miss)
        v_new = torch.where(okv, v_ok, v_miss)
        gap_new = torch.where(ok, torch.zeros_like(gap), gap + dtc.view(-1, 1))
        inited_new = inited | ok

        ratio, rok = st["ratio"].clone(), st["ratio_ok"].clone()
        if cfg.use_bone_projection and self.max_depth > 0:
            x_new = x_new.clone()
            part_scale = (x_new[:, self.scale_a] - x_new[:, self.scale_b]).norm(dim=-1).clamp_min(1e-4)   # (B,P)
            scale_j = part_scale[:, self.part_of_joint]                                                    # (B,J)
            for d in range(1, self.max_depth + 1):
                idx = getattr(self, f"lvl{d}")
                par = self.parent_idx[idx]
                vec = x_new[:, idx] - x_new[:, par]
                length = vec.norm(dim=-1)
                sc = scale_j[:, idx]
                obs = length / sc
                pm = self.project_mask[idx].unsqueeze(0)
                upd = ok[:, idx] & ok[:, par] & pm & (length > 1e-6)   # learn ratio only from real data
                prev_r, prev_ok = ratio[:, idx], rok[:, idx]
                new_r = torch.where(prev_ok, cfg.bone_ratio_ema * prev_r + (1 - cfg.bone_ratio_ema) * obs, obs)
                ratio[:, idx] = torch.where(upd, new_r, prev_r)
                rok[:, idx] = prev_ok | upd
                proj = x_new[:, par] + (vec / length.clamp_min(1e-6).unsqueeze(-1)) * (ratio[:, idx] * sc).unsqueeze(-1)
                use = (rok[:, idx] & pm & (length > 1e-6)).unsqueeze(-1)
                x_new[:, idx] = torch.where(use, proj, x_new[:, idx])

        new_st = dict(x=x_new, v=v_new, inited=inited_new, gap=gap_new, ratio=ratio, ratio_ok=rok)
        return new_st, dict(x=x_new, v=v_new, trust=conf * ok.float(), gap=gap_new)


# ==============================================================================
# FeatureExpander  (canonicalise + expand)
# ==============================================================================
class FeatureExpander(nn.Module):
    """
    Turns the cleaned pose into a long feature vector. No learnable parameters.
    Works for ANY layout of parts (cfg.parts); each part is processed by its own block.

    BODY FRAME (only if cfg.body_ref is set): origin = midpoint of the two reference joints
    (shoulders), unit = their distance. Each part's anchor position is expressed in it
    (rel = (anchor - origin)/unit), so features do not depend on where the person stands,
    leans or how far they are from the camera, and "hand near chin" becomes visible.
    If a reference joint is untrusted this frame, the last good frame is reused and a
    body_ok flag (0/1) is part of the features. With body_ref=None, origin=0 and unit=1
    (camera frame, the old behaviour).

    Per part, local coordinates are divided by that part's own scale (its scale_pair).

    kind == "hand" (21 points). Palm-frame canonicalisation => shape independent of
    position / rotation / distance:
        core : shape 3n | root 16 (body-relative wrist 3, palm rotation 9, log scale 1,
               wrist velocity/scale 3) | palm_w 1 | part_frac 1
        rich : bones 3*nb (unit directions) | angles (cos between a bone and its parent bone)
               | tips 10 (pairwise fingertip distances: pinch!) | vel 3n
        Palm frame: origin wrist, e1 toward index base, e3 = e1 x (wrist->pinky base),
        e2 = e3 x e1. If the palm is edge-on e3 is ill-defined -> palm_w shrinks to 0 and
        frame-dependent features fade out (known weakness).
    kind == "generic" (arms/torso, face subset, anything else). Translation+scale
        invariant only (no rotation canonicalisation):
        core : local coords 3n (relative to anchor / scale) | root 6 (body-relative anchor 3,
               anchor velocity/scale 3) | part_frac 1
        rich : bones 3*nb | angles | vel 3n         (bones/angles only if the part has parents)
    Global: core += trust (J), tanh(gap age) (J), [body_ok]; rich += residual (raw-clean)/scale (3J).

    A part with too few trusted joints (part_frac <= hand_ok_frac) has its WHOLE block
    multiplied by 0 so stale/garbage geometry is not fed forward; other parts are unaffected
    (a vanished hand does not silence the face).

    aux outputs: quality (B,) mean trust over present parts, speed (B,) mean joint speed in
    part-scales/s over PRESENT parts with motion=True, hand_ok (B,) 1 if any motion part is
    present (the name is historical: "something that moves is tracked").
    """

    def __init__(self, cfg: YAPConfig):
        super().__init__()
        self.cfg = cfg
        self.parts = list(cfg.parts)
        self.J = cfg.total_joints
        self.offs: List[int] = []
        self.use_body = cfg.body_ref is not None
        if self.use_body:
            (n0, j0), (n1, j1) = cfg.body_ref
            self.body_r0, self.body_r1 = cfg.joint_index(n0, j0), cfg.joint_index(n1, j1)
        self.register_buffer("tips", torch.tensor(FINGERTIPS))
        tri = torch.triu_indices(len(FINGERTIPS), len(FINGERTIPS), offset=1)
        self.register_buffer("tri_i", tri[0])
        self.register_buffer("tri_j", tri[1])
        n_tip = tri.shape[1]
        core_dim = 2 * self.J + (1 if self.use_body else 0)
        rich_dim = 3 * self.J
        off = 0
        for pi, spec in enumerate(self.parts):
            n, par = spec.num_joints, spec.parents
            self.offs.append(off)
            off += n
            nonroot = [j for j in range(n) if par[j] >= 0]
            ang = [j for j in nonroot if par[par[j]] >= 0]         # joints whose parent also has a parent
            pos = {j: i for i, j in enumerate(nonroot)}            # position inside the nonroot list
            self.register_buffer(f"nr{pi}", torch.tensor(nonroot, dtype=torch.long))
            self.register_buffer(f"nrp{pi}", torch.tensor([par[j] for j in nonroot], dtype=torch.long))
            self.register_buffer(f"ang{pi}", torch.tensor([pos[j] for j in ang], dtype=torch.long))
            self.register_buffer(f"angp{pi}", torch.tensor([pos[par[j]] for j in ang], dtype=torch.long))
            if spec.kind == "hand":
                core_dim += 3 * n + 16 + 2
                rich_dim += 3 * len(nonroot) + len(ang) + n_tip + 3 * n
            else:
                core_dim += 3 * n + 6 + 1
                rich_dim += 3 * len(nonroot) + len(ang) + 3 * n
        self.core_dim, self.rich_dim = core_dim, rich_dim

    def init_state(self, B: int, device):
        return dict(prev_local=[torch.zeros(B, s.num_joints, 3, device=device) for s in self.parts],
                    prev_ok=[torch.zeros(B, dtype=torch.bool, device=device) for _ in self.parts],
                    origin=torch.zeros(B, 3, device=device), bscale=torch.ones(B, 1, device=device))

    @torch.no_grad()
    def step(self, st, raw, x, v, trust, gap, dt):
        """
        raw/x/v (B,J,3) trust/gap (B,J) dt (B,)
        returns new_state, core (B,core_dim), rich (B,rich_dim), aux dict(quality, speed, hand_ok)
        """
        cfg = self.cfg
        B, J = x.shape[0], self.J
        dtv = dt.clamp_min(1e-3)
        dev = x.device

        # ---- body frame ----
        if self.use_body:
            r0, r1 = self.body_r0, self.body_r1
            org_now = 0.5 * (x[:, r0] + x[:, r1])
            bs_now = (x[:, r0] - x[:, r1]).norm(dim=-1, keepdim=True).clamp_min(1e-3)
            body_ok = (trust[:, r0] > 0) & (trust[:, r1] > 0)
            origin = torch.where(body_ok.unsqueeze(-1), org_now, st["origin"])
            bscale = torch.where(body_ok.unsqueeze(-1), bs_now, st["bscale"])
        else:
            origin, bscale = torch.zeros(B, 3, device=dev), torch.ones(B, 1, device=dev)
            body_ok = None

        core_parts, rich_parts, new_prev_local, new_prev_ok = [], [], [], []
        scale_j = x.new_zeros(B, J)
        q_num = x.new_zeros(B)
        q_den = x.new_zeros(B)
        sp_num = x.new_zeros(B)
        sp_den = x.new_zeros(B)
        hand_ok = x.new_zeros(B)

        for pi, spec in enumerate(self.parts):
            o, n = self.offs[pi], spec.num_joints
            xp, tp, vp = x[:, o:o + n], trust[:, o:o + n], v[:, o:o + n]
            a_, b_ = spec.scale_pair
            scale = (xp[:, a_] - xp[:, b_]).norm(dim=-1).clamp_min(1e-4)                 # (B,)
            scale_j[:, o:o + n] = scale.unsqueeze(-1)
            frac = (tp > 0).float().mean(-1)                                              # (B,)
            gate = (frac > cfg.hand_ok_frac).float()                                      # (B,)
            g = gate.unsqueeze(-1)
            anchor = xp[:, spec.anchor]
            rel = (anchor - origin) / bscale                                              # body-relative position
            prev_local, prev_ok = st["prev_local"][pi], st["prev_ok"][pi]
            nr, nrp = getattr(self, f"nr{pi}"), getattr(self, f"nrp{pi}")
            ang, angp = getattr(self, f"ang{pi}"), getattr(self, f"angp{pi}")

            if spec.kind == "hand":
                wrist = xp[:, WRIST]
                e1 = F.normalize(xp[:, INDEX_MCP] - wrist, dim=-1)
                nvec = torch.linalg.cross(e1, xp[:, PINKY_MCP] - wrist, dim=-1)
                n_norm = nvec.norm(dim=-1)
                e3 = nvec / n_norm.clamp_min(1e-6).unsqueeze(-1)
                e2 = torch.linalg.cross(e3, e1, dim=-1)
                R = torch.stack([e1, e2, e3], dim=-1)                                     # (B,3,3) columns = axes
                frame_w = (n_norm / (0.3 * scale)).clamp(0, 1)                            # 1 normal, ->0 edge-on
                local = torch.einsum("bkc,bcd->bkd", xp - wrist.unsqueeze(1), R) / scale[:, None, None]
                wvel = (vp[:, WRIST] / scale.unsqueeze(-1)).clamp(-cfg.vel_clip, cfg.vel_clip)
                root = torch.cat([rel, R.flatten(1), scale.log().unsqueeze(-1), wvel], dim=-1)      # 16
            else:
                frame_w = torch.ones(B, device=dev)
                local = (xp - anchor.unsqueeze(1)) / scale[:, None, None]
                avel = (vp[:, spec.anchor] / scale.unsqueeze(-1)).clamp(-cfg.vel_clip, cfg.vel_clip)
                root = torch.cat([rel, avel], dim=-1)                                                # 6

            unit = F.normalize(local[:, nr] - local[:, nrp], dim=-1)                      # (B,nb,3)
            angles = (unit[:, ang] * unit[:, angp]).sum(-1)                               # (B,na)
            vel_ok = (prev_ok & (frame_w > 0.5)).float().view(B, 1, 1)
            vloc = ((local - prev_local) / dtv.view(B, 1, 1)).clamp(-cfg.vel_clip, cfg.vel_clip) * vel_ok
            pw = frame_w.unsqueeze(-1)
            shape = local.flatten(1) * pw
            if spec.kind == "hand":
                tpts = local[:, self.tips]
                tdist = (tpts[:, self.tri_i] - tpts[:, self.tri_j]).norm(dim=-1)          # (B,10)
                core_p = torch.cat([shape, root, frame_w.unsqueeze(-1), frac.unsqueeze(-1)], dim=-1) * g
                rich_p = torch.cat([unit.flatten(1) * pw, angles * pw, tdist * pw, vloc.flatten(1) * pw], dim=-1) * g
            else:
                core_p = torch.cat([shape, root, frac.unsqueeze(-1)], dim=-1) * g
                rich_p = torch.cat([unit.flatten(1), angles, vloc.flatten(1)], dim=-1) * g
            core_parts.append(core_p)
            rich_parts.append(rich_p)
            new_prev_local.append(local)
            new_prev_ok.append((gate > 0) & (frame_w > 0.5))

            # aux scalars
            present = (tp > 0).float()
            q_num = q_num + (tp * g).sum(-1)
            q_den = q_den + gate * n
            if spec.motion:
                sp = (vp.norm(dim=-1) * present).sum(-1) / present.sum(-1).clamp_min(1.0) / scale
                sp_num = sp_num + sp * gate
                sp_den = sp_den + gate
                hand_ok = torch.maximum(hand_ok, gate)

        resid = ((raw - x) / scale_j.unsqueeze(-1)) * (trust > 0).unsqueeze(-1).float()
        resid = resid.clamp(-1, 1).flatten(1)
        core_all = core_parts + [trust, torch.tanh(gap)]
        if self.use_body:
            core_all.append(body_ok.float().unsqueeze(-1))
        core = torch.cat(core_all, dim=-1)
        rich = torch.cat(rich_parts + [resid], dim=-1)
        new_st = dict(prev_local=new_prev_local, prev_ok=new_prev_ok,
                      origin=origin if self.use_body else st["origin"],
                      bscale=bscale if self.use_body else st["bscale"])
        aux = dict(quality=q_num / q_den.clamp_min(1.0), speed=sp_num / sp_den.clamp_min(1.0), hand_ok=hand_ok)
        return new_st, core, rich, aux


# ==============================================================================
# CausalConvStem  == the "CNN head"
# ==============================================================================
class CausalConvStem(nn.Module):
    """
    Depthwise-separable 1-D convolution over the last `k` poses (CAUSAL: only
    the present and the past). Streaming-safe: state = the previous k-1 inputs.
    out = LayerNorm(x + pointwise(silu(depthwise(x_{t-k+1..t}))))
    """

    def __init__(self, d: int, k: int):
        super().__init__()
        self.k, self.d = k, d
        w = torch.randn(k, d) * 0.05
        w[-1] += 1.0 / 2                      # start near "look mostly at the present pose"
        self.dw = nn.Parameter(w)
        self.pw = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def init_state(self, B, device):
        return torch.zeros(B, self.k - 1, self.d, device=device)

    def step(self, x, state):
        buf = torch.cat([state, x.unsqueeze(1)], dim=1)               # (B,k,d)
        y = (buf * self.dw.unsqueeze(0)).sum(1)
        out = self.norm(x + self.pw(F.silu(y)))
        return out, buf[:, 1:]


# ==============================================================================
# MemoryLayer  (memory + assurity of presence, in EVERY layer)
# ==============================================================================
class MemoryLayer(nn.Module):
    """
    One layer of the frame-level network. It owns:
      * a bank of leaky integrators with d different time constants
        (tau from mem_tau_min_s to mem_tau_max_s, log-spaced, learnable):
            s <- a*s + (1-a)*u,   a = exp(-dt/tau),   u = in_proj(x) * write_gate(x)
        so the layer "remembers what came previously" at many time-scales and
        uses the REAL dt (uneven frame spacing is handled exactly).
      * a confidence c in (0,1) ("assurity of presence") computed from the layer's
        input and memory. It is passed to the next layer as an extra input and
        returned for deep supervision (layer_conf_loss).
    Residual structure: x -> x + mix([x,c_prev]) -> + out(s) -> + FFN.
    """

    def __init__(self, d: int, tau_min: float, tau_max: float, ffn_mult: int):
        super().__init__()
        self.mix = nn.Linear(d + 1, d)
        self.norm1 = nn.LayerNorm(d)
        self.inp = nn.Linear(d, d)
        self.write = nn.Linear(d, d)
        taus = torch.exp(torch.linspace(math.log(tau_min), math.log(tau_max), d))
        self.log_rate = nn.Parameter(torch.log(1.0 / taus))          # rate = 1/tau, kept positive via exp
        self.conf_head = nn.Linear(2 * d, 1)
        self.out = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d * ffn_mult), nn.GELU(), nn.Linear(d * ffn_mult, d))

    def init_state(self, B, device):
        return torch.zeros(B, self.out.in_features, device=device)

    def step(self, x, conf_prev, dt, s):
        x = x + self.mix(torch.cat([x, conf_prev], dim=-1))
        h = self.norm1(x)
        a = torch.exp(-self.log_rate.exp().unsqueeze(0) * dt.unsqueeze(-1))      # (B,d)
        u = self.inp(h) * torch.sigmoid(self.write(h))
        s = a * s + (1 - a) * u
        conf = torch.sigmoid(self.conf_head(torch.cat([h, s], dim=-1)))          # (B,1)
        x = x + self.out(s)
        x = x + self.ffn(self.norm2(x))
        return x, conf, s


# ==============================================================================
# HiPPO-LegS pooling  (time axis -> fixed-size state)
# ==============================================================================
class HiPPOLegS(nn.Module):
    """
    Online projection of a signal's whole history onto N Legendre polynomials.
        dc/dt = -(1/t) A c + (1/t) B f(t)
    A,B are the fixed LegS matrices. We integrate with BACKWARD EULER using the
    real time step so uneven timestamps work and the step is unconditionally
    stable:
        (I + r A) c_new = c_old + r B f,   r = dt / t_new
    Properties we rely on:
      * dt == 0  =>  r == 0  =>  c_new == c_old (exact no-op): lets us batch
        segments of different length by padding with dt = 0.
      * state is fixed-size (C channels x N coefficients) however long the segment.
    c has shape (S, N, C): S segments, N Legendre orders, C channels.
    """

    def __init__(self, N: int):
        super().__init__()
        A = torch.zeros(N, N)
        for n in range(N):
            for k in range(N):
                if n > k:
                    A[n, k] = math.sqrt((2 * n + 1) * (2 * k + 1))
                elif n == k:
                    A[n, k] = n + 1
        self.register_buffer("A", A)
        self.register_buffer("Bv", torch.sqrt(2.0 * torch.arange(N) + 1.0))
        self.register_buffer("I", torch.eye(N))
        self.N = N

    def step(self, c, f, dt, tau_new):
        r = (dt / tau_new).view(-1, 1, 1)
        M = self.I.unsqueeze(0) + r * self.A.unsqueeze(0)
        rhs = c + r * self.Bv.view(1, -1, 1) * f.unsqueeze(1)
        return torch.linalg.solve(M, rhs)


# ==============================================================================
# SEGMENTATION
# ==============================================================================
@dataclass
class Segment:
    """One detected gesture. Frames [start_idx, end_idx) of the sequence."""
    start_idx: int
    end_idx: int          # exclusive
    start_t: float
    end_t: float          # time of the LAST frame inside the segment
    reason: str           # presence_drop | cut | max_len | still_flush | eos | given
    preroll_s: float      # how many seconds of lead-in context to feed along


def overlap_preroll_s(cfg: YAPConfig, prev_dur: Optional[float]) -> float:
    """The '50% overlap' rule: pre-roll = overlap_fraction * previous segment duration, clamped."""
    if prev_dur is None:
        return cfg.preroll_min_s
    return float(min(cfg.preroll_max_s, max(cfg.preroll_min_s, cfg.overlap_fraction * prev_dur)))


def pre_idx_from_time(t_list: List[float], start_idx: int, start_t: float, preroll_s: float) -> int:
    """First frame index whose time is >= start_t - preroll_s (never after start_idx)."""
    pre = start_idx
    while pre > 0 and t_list[pre - 1] >= start_t - preroll_s - 1e-9:
        pre -= 1
    return pre


class SegmentTracker:
    """
    Plain-python state machine that turns per-pose (presence p, end prob q,
    speed, hand tracked?) into segments. Used identically by offline code and
    by YAPStream. Call update() once per pose; call flush() at end of stream.

    Rules (all thresholds in YAPConfig):
      IDLE   -> ACTIVE : p >= on_thr AND hand tracked, for on_hold_s. The start is
                         BACK-DATED to the first pose of that run.
      ACTIVE -> end    : (1) presence_drop: p < off_thr for off_hold_s. Tracker
                         dropouts shorter than gap_tolerance_s do not count as low
                         (a vanished landmark must not split one sign).
                         The end index is the first pose of the low run.
                         (2) max_len : segment older than max_seg_s (hard cap).
                         (3) cut     : a local maximum of the end probability above
                         cut_thr, segment older than min_seg_s. The next segment
                         starts immediately (back-to-back signs).
      Stillness: speed < still_speed_thr (or no hand) for still_flush_s ->
                 end the active segment and set reset=True (caller must wipe ALL
                 memory). Fires once per still period.
    update() returns (list_of_Segment, reset_flag).
    Latency: a presence_drop segment is only reported off_hold_s after it truly
    ended; a cut is reported one pose late (needs the next pose to confirm the peak).
    """

    def __init__(self, cfg: YAPConfig):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.idx = 0
        self.t_of: Dict[int, float] = {}
        self.active = False
        self.seg_start_idx = 0
        self.seg_start_t = 0.0
        self.cur_preroll_s = self.cfg.preroll_min_s
        self.prev_seg_dur: Optional[float] = None
        self.on_time = 0.0
        self.on_run_start = 0
        self.low_time = 0.0
        self.low_start_idx = 0
        self.lost_time = 0.0
        self.still_time = 0.0
        self.still_start_idx = 0
        self.flushed = False
        self.q1 = 0.0   # end prob one pose ago
        self.q2 = 0.0   # end prob two poses ago

    def _begin(self, start_idx: int):
        self.active = True
        self.seg_start_idx = start_idx
        self.seg_start_t = self.t_of[start_idx]
        self.cur_preroll_s = overlap_preroll_s(self.cfg, self.prev_seg_dur)
        self.low_time = 0.0

    def _end(self, end_idx: int, reason: str) -> Optional[Segment]:
        self.active = False
        if end_idx <= self.seg_start_idx:
            return None
        seg = Segment(self.seg_start_idx, end_idx, self.seg_start_t, self.t_of[end_idx - 1], reason, self.cur_preroll_s)
        self.prev_seg_dur = seg.end_t - seg.start_t
        return seg

    def update(self, t: float, dt: float, p: float, q: float, speed: float, ok: bool):
        c, i = self.cfg, self.idx
        self.idx += 1
        self.t_of[i] = t
        self.t_of.pop(i - 4096, None)           # bounded memory
        out: List[Segment] = []
        reset = False

        # stillness bookkeeping (counts "no hand" as still)
        if speed < c.still_speed_thr:
            if self.still_time == 0.0:
                self.still_start_idx = i
            self.still_time += dt
        else:
            self.still_time = 0.0
            self.flushed = False
        # tracker dropout bookkeeping
        self.lost_time = 0.0 if ok else self.lost_time + dt
        high = ok and p >= c.on_thr
        low = (p < c.off_thr) and (ok or self.lost_time > c.gap_tolerance_s)

        if not self.active:
            if high:
                if self.on_time == 0.0:
                    self.on_run_start = i
                self.on_time += dt
                if self.on_time >= c.on_hold_s:
                    self._begin(self.on_run_start)
                    self.on_time = 0.0
            else:
                self.on_time = 0.0
        else:
            if low:
                if self.low_time == 0.0:
                    self.low_start_idx = i
                self.low_time += dt
            else:
                self.low_time = 0.0
            if self.low_time >= c.off_hold_s:
                s = self._end(self.low_start_idx, "presence_drop")
                if s:
                    out.append(s)
                self.low_time = 0.0
            elif t - self.seg_start_t >= c.max_seg_s:
                s = self._end(i, "max_len")
                if s:
                    out.append(s)
                if high:
                    self._begin(i)
            elif (self.q1 >= c.cut_thr and self.q1 >= self.q2 and self.q1 >= q
                  and i - 1 > self.seg_start_idx
                  and (self.t_of[i - 1] - self.seg_start_t) >= c.min_seg_s):
                s = self._end(i, "cut")            # pose i-1 (the peak) is the last pose inside
                if s:
                    out.append(s)
                self._begin(i)                     # back-to-back: next sign starts right away

        if self.still_time >= c.still_flush_s and not self.flushed:
            self.flushed = True
            reset = True
            if self.active:
                s = self._end(max(self.still_start_idx, self.seg_start_idx + 1), "still_flush")
                if s:
                    out.append(s)
        self.q2, self.q1 = self.q1, q
        return out, reset

    def flush(self) -> List[Segment]:
        """End of stream: close an open segment."""
        if self.active:
            s = self._end(self.idx, "eos")
            return [s] if s else []
        return []


def segment_sequence(cfg: YAPConfig, t, dt, p, q, speed, ok) -> List[Segment]:
    """Offline segmentation of one sequence (python lists). Same tracker as live."""
    tr = SegmentTracker(cfg)
    segs: List[Segment] = []
    for k in range(len(t)):
        s, _ = tr.update(t[k], dt[k], p[k], q[k], speed[k], ok[k])
        segs += s
    return segs + tr.flush()


def make_segments(t_list: List[float], spans_t: List[Tuple[float, float]], cfg: YAPConfig) -> List[Segment]:
    """
    Build ground-truth Segments from labelled (start_time, end_time) pairs using the
    SAME overlap/pre-roll rule as the tracker. Used for teacher forcing in training.
    """
    segs, prev_dur = [], None
    for s_t, e_t in spans_t:
        s_idx = bisect_left(t_list, s_t)
        e_idx = bisect_right(t_list, e_t)
        if e_idx <= s_idx:
            continue
        segs.append(Segment(s_idx, e_idx, t_list[s_idx], t_list[e_idx - 1], "given", overlap_preroll_s(cfg, prev_dur)))
        prev_dur = t_list[e_idx - 1] - t_list[s_idx]
    return segs


# ==============================================================================
# YAPCore  (per-pose network)
# ==============================================================================
class YAPCore(nn.Module):
    """
    Everything that happens to ONE incoming pose. `step()` is the only place the
    per-pose maths lives; `forward_frames()` (training / offline) just loops it and
    YAPStream (live) calls it once per arriving pose. Therefore train == live.
    The state dict is MUTATED IN PLACE and also returned.
    """

    def __init__(self, cfg: YAPConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.cleaner = PoseCleaner(cfg)
        self.expander = FeatureExpander(cfg)
        fdim = self.expander.core_dim + self.expander.rich_dim
        self.in_norm = nn.LayerNorm(fdim)
        self.lift = nn.Sequential(nn.Linear(fdim, d), nn.GELU())
        self.stem = CausalConvStem(d, cfg.cnn_kernel)
        self.layers = nn.ModuleList([MemoryLayer(d, cfg.mem_tau_min_s, cfg.mem_tau_max_s, cfg.ffn_mult)
                                     for _ in range(cfg.n_mem_layers)])
        self.presence_head = nn.Linear(d + 1, 1)
        self.end_head = nn.Linear(d + 1, 1)
        nn.init.constant_(self.end_head.bias, -3.0)       # "cuts are rare" prior
        self.summary_head = nn.Linear(d, cfg.summary_channels)

    def init_state(self, B: int, device):
        return dict(clean=self.cleaner.init_state(B, device), exp=self.expander.init_state(B, device),
                    stem=self.stem.init_state(B, device),
                    mem=[l.init_state(B, device) for l in self.layers],
                    prev_presence=torch.zeros(B, 1, device=device))

    def step(self, st, pose, conf, valid, dt):
        cfg = self.cfg
        with torch.no_grad():                            # filters have no parameters
            st["clean"], cl = self.cleaner.step(st["clean"], pose, conf, valid, dt)
            st["exp"], core, rich, aux = self.expander.step(st["exp"], pose, cl["x"], cl["v"], cl["trust"], cl["gap"], dt)
        # adaptive expansion: rich block strength follows the previous presence estimate
        scale = cfg.expand_floor + (1 - cfg.expand_floor) * st["prev_presence"]
        x = self.lift(self.in_norm(torch.cat([core, rich * scale], dim=-1)))
        x, st["stem"] = self.stem.step(x, st["stem"])
        conf_prev = aux["quality"].unsqueeze(-1)
        confs = []
        for i, layer in enumerate(self.layers):
            x, c, st["mem"][i] = layer.step(x, conf_prev, dt, st["mem"][i])
            confs.append(c)
            conf_prev = c
        hc = torch.cat([x, conf_prev], dim=-1)
        pres = self.presence_head(hc).squeeze(-1)
        end = self.end_head(hc).squeeze(-1)
        feat = torch.tanh(self.summary_head(x))
        st["prev_presence"] = torch.sigmoid(pres).detach().unsqueeze(-1)
        return st, dict(presence=pres, end=end, layer_conf=torch.cat(confs, dim=-1), feat=feat,
                        quality=aux["quality"], speed=aux["speed"], hand_ok=aux["hand_ok"], clean=cl["x"])

    def forward_frames(self, pose, conf, valid, t, reset_mask: Optional[torch.Tensor] = None):
        """
        Run all T poses of B sequences. Returns dict of stacked per-pose outputs:
          presence (B,T) logits, end (B,T) logits, layer_conf (B,T,L), feat (B,T,C),
          quality (B,T), speed (B,T), hand_ok (B,T), clean (B,T,J,3), dt (B,T).
        reset_mask (B,T) bool: wipe that row's memory BEFORE that pose (mirrors the
        still-flush reset that YAPStream performs live).
        """
        B, T = pose.shape[:2]
        dev = pose.device
        dt = frame_dts(t, self.cfg.nominal_fps)
        st = self.init_state(B, dev)
        fresh = self.init_state(B, dev)
        outs: List[Dict[str, torch.Tensor]] = []
        for k in range(T):
            if reset_mask is not None and bool(reset_mask[:, k].any()):
                st = _tree_select(reset_mask[:, k], fresh, st)
            if self.cfg.tbptt and k > 0 and k % self.cfg.tbptt == 0:
                st = _tree_detach(st)
            st, o = self.step(st, pose[:, k], conf[:, k], valid[:, k], dt[:, k])
            outs.append(o)
        res = {key: torch.stack([o[key] for o in outs], dim=1) for key in outs[0]}
        res["dt"] = dt
        return res


# ==============================================================================
# YAP  (core + segmentation + pooling into entities)
# ==============================================================================
class YAP(nn.Module):
    """
    Full YAP module. Typical use:
        out      = yap.forward_frames(...)                    # per-pose heads
        segments = yap.segment_batch(out, t, lengths)         # or ground truth
        ents, meta, owner = yap.encode_segments(out, t, segments)
    """

    def __init__(self, cfg: YAPConfig):
        super().__init__()
        self.cfg = cfg
        self.core = YAPCore(cfg)
        C, N = cfg.summary_channels, cfg.hippo_order
        self.hippo = HiPPOLegS(N)
        self.ctx_vec = nn.Parameter(torch.randn(C) * 0.02)     # "this pose is pre-roll context" flag
        self.readout_norm = nn.LayerNorm(C * N)
        self.readout = nn.Sequential(nn.Linear(C * N + 3, cfg.entity_dim), nn.GELU(),
                                     nn.Linear(cfg.entity_dim, cfg.entity_dim))

    # ---- per-pose --------------------------------------------------------------
    def forward_frames(self, pose, conf, valid, t, reset_mask=None):
        return self.core.forward_frames(pose, conf, valid, t, reset_mask)

    # ---- segmentation ----------------------------------------------------------
    @torch.no_grad()
    def segment_batch(self, out, t, lengths: Optional[torch.Tensor] = None) -> List[List[Segment]]:
        """Run the SegmentTracker over each sequence's presence/end probabilities."""
        p = torch.sigmoid(out["presence"]).cpu()
        q = torch.sigmoid(out["end"]).cpu()
        sp, dt, tt = out["speed"].cpu(), out["dt"].cpu(), t.cpu()
        ok = (out["hand_ok"] > 0.5).cpu()
        res = []
        for b in range(p.shape[0]):
            L = int(lengths[b]) if lengths is not None else p.shape[1]
            res.append(segment_sequence(self.cfg, tt[b, :L].tolist(), dt[b, :L].tolist(), p[b, :L].tolist(),
                                        q[b, :L].tolist(), sp[b, :L].tolist(), ok[b, :L].tolist()))
        return res

    # ---- pooling ---------------------------------------------------------------
    def encode_spans(self, feat, quality, dt, t, spans: List[Tuple[int, int, int, int]]):
        """
        Fold segments into entity vectors (DIFFERENTIABLE w.r.t. feat).
        spans: list of (b, pre_idx, start_idx, end_idx): pool poses pre_idx..end_idx-1 of
        sequence b; poses before start_idx are flagged as context.
        Returns entity (S, entity_dim) and extras (S,3) = [duration_s, mean_quality, preroll_s].
        Segments of different length are padded with dt = 0 (exact no-ops).
        """
        cfg = self.cfg
        S = len(spans)
        L = max(e - p for (_, p, _, e) in spans)
        C = feat.shape[-1]
        dev = feat.device
        F_ = feat.new_zeros(S, L, C)
        DT = feat.new_zeros(S, L)
        Q = feat.new_zeros(S, L)
        CTX = feat.new_zeros(S, L)
        extras = feat.new_zeros(S, 3)
        for s, (b, pre, st_, en) in enumerate(spans):
            n = en - pre
            F_[s, :n] = feat[b, pre:en]
            DT[s, :n] = dt[b, pre:en]
            Q[s, :n] = quality[b, pre:en]
            CTX[s, :n * 1] = (torch.arange(pre, en, device=dev) < st_).float()
            extras[s, 0] = (t[b, en - 1] - t[b, st_]).clamp_min(0)
            extras[s, 1] = quality[b, st_:en].mean()
            extras[s, 2] = t[b, st_] - t[b, pre]
        F_ = F_ + CTX.unsqueeze(-1) * self.ctx_vec            # tell the pooling which poses are lead-in
        c = feat.new_zeros(S, cfg.hippo_order, C)
        tau = feat.new_full((S,), cfg.hippo_t0_s)
        hold = F_[:, 0]
        for k in range(L):
            good = (Q[:, k] >= cfg.hold_quality).unsqueeze(-1)
            hold = torch.where(good, F_[:, k], hold)          # zero-order hold over unreliable poses
            tau = tau + DT[:, k]
            c = self.hippo.step(c, hold, DT[:, k], tau)
        flat = self.readout_norm(c.flatten(1))
        ent = self.readout(torch.cat([flat, torch.log1p(extras[:, 0:1]), extras[:, 1:2], torch.log1p(extras[:, 2:3])], dim=-1))
        return ent, extras

    def encode_segments(self, out, t, segments: List[List[Segment]]):
        """
        Segments (per batch row) -> entities.
        Returns entities (S,D), meta (S,3)=[log1p(duration), log1p(gap_before), mean_quality],
        owner list (len S) giving the batch row of each entity.
        """
        spans, owner, gaps = [], [], []
        for b, segs in enumerate(segments):
            tl = t[b].tolist()
            prev_end = None
            for s in segs:
                pre = pre_idx_from_time(tl, s.start_idx, s.start_t, s.preroll_s)
                spans.append((b, pre, s.start_idx, s.end_idx))
                owner.append(b)
                gaps.append(0.0 if prev_end is None else max(0.0, s.start_t - prev_end))
                prev_end = s.end_t
        dev = out["feat"].device
        if not spans:
            return out["feat"].new_zeros(0, self.cfg.entity_dim), out["feat"].new_zeros(0, 3), []
        ent, ex = self.encode_spans(out["feat"], out["quality"], out["dt"], t, spans)
        meta = torch.stack([torch.log1p(ex[:, 0]), torch.log1p(torch.tensor(gaps, device=dev)), ex[:, 1]], dim=-1)
        return ent, meta, owner

    def stream(self, device="cpu") -> "YAPStream":
        return YAPStream(self, device)


# ==============================================================================
# YAPStream  (LIVE, one pose at a time, batch size 1)
# ==============================================================================
@dataclass
class Entity:
    vector: torch.Tensor   # (entity_dim,)
    meta: torch.Tensor     # (3,) [log1p(duration), log1p(gap_before), mean_quality]
    start_t: float
    end_t: float
    reason: str


class YAPStream:
    """
    Live driver. Feed poses with push(); it returns the entities that became
    complete with that pose (usually none, sometimes one, rarely two).
    Internally: YAPCore.step -> per-pose buffer -> SegmentTracker -> when the
    tracker reports a segment, the buffered poses of that segment (+pre-roll) are
    pooled with the SAME encode_spans used offline, so live == offline.
    On a still-flush the whole state AND the buffer are wiped.
    The buffer is bounded (deque) so memory never grows.
    """

    def __init__(self, yap: YAP, device="cpu"):
        self.yap, self.cfg, self.device = yap.eval(), yap.cfg, device
        self.state = yap.core.init_state(1, device)
        self.tracker = SegmentTracker(self.cfg)
        self.buf: deque = deque(maxlen=2048)
        self.t_prev: Optional[float] = None
        self.last_end_t: Optional[float] = None
        self.last_out: Optional[Dict[str, torch.Tensor]] = None

    @torch.no_grad()
    def push(self, pose, conf, valid, t: float) -> List[Entity]:
        """pose (J,3) tensor, conf (J,), valid (J,) bool, t seconds."""
        dt = 1.0 / self.cfg.nominal_fps if self.t_prev is None else float(min(max(t - self.t_prev, 1e-3), 0.5))
        self.t_prev = t
        dev = self.device
        self.state, o = self.yap.core.step(self.state, pose[None].to(dev), conf[None].to(dev), valid[None].to(dev),
                                           torch.tensor([dt], device=dev))
        self.last_out = o
        idx = self.tracker.idx
        self.buf.append(dict(idx=idx, t=t, dt=dt, feat=o["feat"][0], q=o["quality"][0]))
        segs, reset = self.tracker.update(t, dt, float(torch.sigmoid(o["presence"])[0]), float(torch.sigmoid(o["end"])[0]),
                                          float(o["speed"][0]), bool(o["hand_ok"][0] > 0.5))
        ents = [self._encode(s) for s in segs]
        if reset:                                              # ultimate clear moment: wipe everything
            self.state = self.yap.core.init_state(1, dev)
            self.buf.clear()
        return ents

    @torch.no_grad()
    def flush(self) -> List[Entity]:
        """Call at end of stream to close an open segment."""
        return [self._encode(s) for s in self.tracker.flush()]

    def _encode(self, seg: Segment) -> Entity:
        items = list(self.buf)
        t_list = [it["t"] for it in items]
        first_idx = items[0]["idx"]
        start_l, end_l = max(seg.start_idx - first_idx, 0), seg.end_idx - first_idx
        pre_l = pre_idx_from_time(t_list, start_l, seg.start_t, seg.preroll_s)
        feat = torch.stack([it["feat"] for it in items]).unsqueeze(0)
        q = torch.stack([it["q"] for it in items]).unsqueeze(0)
        dt = torch.tensor([[it["dt"] for it in items]], dtype=feat.dtype, device=feat.device)
        tt = torch.tensor([t_list], dtype=feat.dtype, device=feat.device)
        ent, ex = self.yap.encode_spans(feat, q, dt, tt, [(0, pre_l, start_l, end_l)])
        gap = 0.0 if self.last_end_t is None else max(0.0, seg.start_t - self.last_end_t)
        self.last_end_t = seg.end_t
        meta = torch.stack([torch.log1p(ex[0, 0]), torch.log1p(torch.tensor(gap)), ex[0, 1]])
        return Entity(ent[0], meta, seg.start_t, seg.end_t, seg.reason)


# ==============================================================================
# LOSSES / TARGET HELPERS  (used by the training stages, see docstring section 4)
# ==============================================================================
def presence_targets(t: torch.Tensor, spans_t: List[Tuple[float, float]], ramp_s: float = 0.1) -> torch.Tensor:
    """
    Soft per-pose presence target (T,). 1 inside each labelled (start,end) span, linear
    ramp to 0 over ramp_s outside. Soft because hand-labelled boundaries are fuzzy.
    """
    tgt = torch.zeros_like(t)
    for s, e in spans_t:
        d = torch.where(t < s, s - t, torch.where(t > e, t - e, torch.zeros_like(t)))
        tgt = torch.maximum(tgt, (1 - d / ramp_s).clamp(0, 1))
    return tgt


def end_targets(t: torch.Tensor, spans_t: List[Tuple[float, float]], sigma_s: float = 0.05) -> torch.Tensor:
    """Gaussian bump (T,) centred on each span's end time = soft target for the 'end' head."""
    tgt = torch.zeros_like(t)
    for _, e in spans_t:
        tgt = torch.maximum(tgt, torch.exp(-0.5 * ((t - e) / sigma_s) ** 2))
    return tgt


def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return x.mean()
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def focal_presence_loss(logit, target, mask=None, gamma: float = 2.0):
    """Focal BCE with SOFT targets (down-weights the many easy 'nothing happening' poses)."""
    p = torch.sigmoid(logit)
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    pt = target * p + (1 - target) * (1 - p)
    return _masked_mean(((1 - pt) ** gamma) * bce, mask)


def layer_conf_loss(layer_conf, presence_target, mask=None):
    """Deep supervision: every layer's confidence should also predict presence."""
    tgt = presence_target.unsqueeze(-1).expand_as(layer_conf)
    bce = F.binary_cross_entropy(layer_conf.clamp(1e-5, 1 - 1e-5), tgt, reduction="none")
    m = None if mask is None else mask.unsqueeze(-1).expand_as(layer_conf)
    return _masked_mean(bce, m)


def soft_count_loss(end_logit, n_words, mask=None):
    """
    Weak supervision (stage 3): expected number of segments = sum of end probabilities
    (+1 for the final segment, which closes at end of stream) should equal the word count.
    end_logit (B,T), n_words (B,) long.
    """
    q = torch.sigmoid(end_logit)
    if mask is not None:
        q = q * mask
    expected = q.sum(1) + 1.0
    return ((expected - n_words.float()) ** 2).mean()


def min_gap_penalty(end_logit, k: int, mask=None):
    """Two cuts closer than k poses are implausible: penalise > 1 cut-probability mass in any k-window."""
    q = torch.sigmoid(end_logit)
    if mask is not None:
        q = q * mask
    win = F.avg_pool1d(q.unsqueeze(1), kernel_size=k, stride=1) * k          # window sums
    return F.relu(win - 1.0).mean()


def dip_prior_loss(end_logit, speed, mask=None):
    """Prior: cuts belong where the hand slows down (end prob mass weighted by speed is penalised)."""
    q = torch.sigmoid(end_logit)
    return _masked_mean(q * speed / (speed.mean().clamp_min(1e-6)), mask)


# ==============================================================================
# CONFORMER  (the transformer part)
# ==============================================================================
@dataclass
class ConformerConfig:
    num_words: int = 100        # vocabulary size; class 0 is reserved for BLANK/no-word, words are 1..num_words
    d_model: int = 256
    heads: int = 4
    layers: int = 4
    ff_mult: int = 4
    conv_kernel: int = 7        # depthwise conv over neighbouring entities (odd number)
    dropout: float = 0.1
    max_entities: int = 128     # size of the position table


def sinusoidal_positions(n: int, d: int, device) -> torch.Tensor:
    pos = torch.arange(n, device=device).float().unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, device=device).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe


class FeedForward(nn.Module):
    def __init__(self, d, mult, drop):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d * mult), nn.SiLU(), nn.Dropout(drop),
                                 nn.Linear(d * mult, d), nn.Dropout(drop))

    def forward(self, x):
        return self.net(x)


class ConvModule(nn.Module):
    """Conformer convolution module: pointwise(2d)+GLU -> depthwise conv -> LayerNorm -> SiLU -> pointwise."""

    def __init__(self, d, k, drop):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.pw1 = nn.Conv1d(d, 2 * d, 1)
        self.dw = nn.Conv1d(d, d, k, padding=k // 2, groups=d)
        self.norm2 = nn.LayerNorm(d)
        self.pw2 = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask):                       # x (B,K,d)  mask (B,K) True = real token
        m = mask.unsqueeze(-1).float()
        y = self.norm(x) * m
        y = F.glu(self.pw1(y.transpose(1, 2)), dim=1)
        y = self.dw(y * m.transpose(1, 2))
        y = F.silu(self.norm2(y.transpose(1, 2)))
        y = self.pw2((y * m).transpose(1, 2)).transpose(1, 2)
        return self.drop(y)


class ConformerBlock(nn.Module):
    """x += 0.5*FFN ; x += MHSA ; x += Conv ; x += 0.5*FFN ; LayerNorm   (Gulati et al. 2020)."""

    def __init__(self, c: ConformerConfig):
        super().__init__()
        d = c.d_model
        self.ff1, self.ff2 = FeedForward(d, c.ff_mult, c.dropout), FeedForward(d, c.ff_mult, c.dropout)
        self.attn_norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, c.heads, dropout=c.dropout, batch_first=True)
        self.conv = ConvModule(d, c.conv_kernel, c.dropout)
        self.out_norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, mask):
        x = x + 0.5 * self.ff1(x)
        h = self.attn_norm(x)
        a, _ = self.attn(h, h, h, key_padding_mask=~mask, need_weights=False)
        x = x + self.drop(a)
        x = x + self.conv(x, mask)
        x = x + 0.5 * self.ff2(x)
        return self.out_norm(x)


class YAPpingConformer(nn.Module):
    """
    YAP (front end) -> entities -> Conformer -> per-entity word logits.

    forward(pose, conf, valid, t, lengths=None, segments=None, reset_mask=None) returns dict:
      word_logits (B,K,num_words+1)  class 0 = blank
      token_mask  (B,K) bool         which of the K slots are real entities
      segments    list of list of Segment (what was used)
      yap         dict of per-pose YAP outputs (presence/end/... for the YAP losses)
    segments=None -> YAP finds them itself (inference / stage 3).
    segments=[...] -> teacher forcing with ground-truth boundaries (stages 1-2);
    build them with make_segments().
    A learned BOS token is always prepended so attention never sees an all-padding row.
    """

    def __init__(self, ycfg: YAPConfig, ccfg: ConformerConfig):
        super().__init__()
        self.ycfg, self.ccfg = ycfg, ccfg
        self.yap = YAP(ycfg)
        d = ccfg.d_model
        self.ent_proj = nn.Linear(ycfg.entity_dim, d)
        self.meta_proj = nn.Sequential(nn.Linear(3, d), nn.SiLU(), nn.Linear(d, d))
        self.bos = nn.Parameter(torch.zeros(1, 1, d))
        self.blocks = nn.ModuleList([ConformerBlock(ccfg) for _ in range(ccfg.layers)])
        self.head = nn.Linear(d, ccfg.num_words + 1)

    def forward(self, pose, conf, valid, t, lengths=None, segments=None, reset_mask=None):
        out = self.yap.forward_frames(pose, conf, valid, t, reset_mask)
        if segments is None:
            segments = self.yap.segment_batch(out, t, lengths)
        ent, meta, owner = self.yap.encode_segments(out, t, segments)
        B, d = pose.shape[0], self.ccfg.d_model
        counts = [len(s) for s in segments]
        K = min(max(max(counts), 1), self.ccfg.max_entities)
        tok = ent.new_zeros(B, K, d)
        mask = torch.zeros(B, K, dtype=torch.bool, device=pose.device)
        fill = [0] * B
        proj = self.ent_proj(ent) + self.meta_proj(meta) if len(owner) else ent.new_zeros(0, d)
        for n, b in enumerate(owner):
            k = fill[b]
            if k < K:
                tok[b, k] = proj[n]
                mask[b, k] = True
                fill[b] += 1
        tok = tok + sinusoidal_positions(K, d, tok.device).unsqueeze(0)
        x = torch.cat([self.bos.expand(B, -1, -1), tok], dim=1)
        full_mask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=x.device), mask], dim=1)
        for blk in self.blocks:
            x = blk(x, full_mask)
        return dict(word_logits=self.head(x[:, 1:]), token_mask=mask, segments=segments, yap=out)


def sentence_ctc_loss(word_logits, token_mask, targets, target_lengths):
    """
    Stage 3: only the word SEQUENCE is known, not which entity is which word.
    CTC lets the number of entities differ from the number of words.
    targets: (B, Lmax) ints in 1..num_words (0 is blank), target_lengths (B,).
    """
    logp = word_logits.log_softmax(-1).transpose(0, 1)                # (K,B,V+1)
    in_len = token_mask.sum(1).clamp_min(1)
    return F.ctc_loss(logp, targets, in_len, target_lengths, blank=0, zero_infinity=True)


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ==============================================================================
# __main__  : smoke tests (run `python yap_conformer.py`)
# ==============================================================================
def _fake_hand_template() -> torch.Tensor:
    """A rough 21-point hand: wrist at origin, five fingers pointing +y (just for testing)."""
    pts = torch.zeros(21, 3)
    xs = [-0.30, -0.15, 0.0, 0.15, 0.30]
    for f in range(5):
        base = 1 + 4 * f
        for k in range(4):
            pts[base + k] = torch.tensor([xs[f], (0.20 if f == 0 else 0.45) + 0.12 * k, 0.0])
    return pts


def _generic_template(spec: PartSpec, seed: int) -> torch.Tensor:
    """Fixed pseudo-random points for a non-hand part (just for testing); scale pair kept well separated."""
    g = torch.Generator().manual_seed(1000 + seed)
    pts = torch.randn(spec.num_joints, 3, generator=g) * 0.1
    a, b = spec.scale_pair
    pts[b] = pts[a] + torch.tensor([0.4, 0.0, 0.0])
    return pts


def _synthetic_batch(B: int, T: int, cfg: YAPConfig, seed: int = 0, fps: float = 30.0):
    """
    Moving noisy fake parts (any layout in cfg.parts) with random landmark dropouts and a
    whole-part dropout of the FIRST part for 8 poses. Returns pose, conf, valid, t.
    """
    g = torch.Generator().manual_seed(seed)
    J = cfg.total_joints
    t = (torch.arange(T).float() / fps).unsqueeze(0).repeat(B, 1) + 0.002 * torch.rand(B, T, generator=g)
    t, _ = torch.sort(t, dim=1)
    pose = torch.zeros(B, T, J, 3)
    offs = cfg.part_offsets()
    for k, spec in enumerate(cfg.parts):
        o, n = offs[spec.name], spec.num_joints
        phase = torch.rand(B, 1, generator=g) * 6.28
        move = torch.stack([0.3 * torch.sin(t * 3 + phase) + 0.5 + 0.8 * k,
                            0.2 * torch.cos(t * 2 + phase) + 0.5,
                            0.05 * torch.sin(t * 5)], dim=-1)             # (B,T,3)
        base = _fake_hand_template() * 0.5 if spec.kind == "hand" else _generic_template(spec, k)
        pose[:, :, o:o + n] = base.view(1, 1, n, 3) + move.unsqueeze(2)
    pose = pose + 0.004 * torch.randn(B, T, J, 3, generator=g)
    conf = 0.9 + 0.1 * torch.rand(B, T, J, generator=g)
    valid = torch.rand(B, T, J, generator=g) > 0.05                        # 5% random landmark dropouts
    valid[:, T // 3: T // 3 + 8, :cfg.parts[0].num_joints] = False         # first part vanishes for 8 poses
    return pose, conf, valid, t


def _run_tests():
    torch.manual_seed(0)
    # ---------------- test 1: tracker on a hand-made presence curve -----------------
    cfg = YAPConfig()
    n, fps = 200, 30.0
    t = [i / fps for i in range(n)]
    dt = [1 / fps] * n
    p = [0.95 if (20 <= i < 60 or 100 <= i < 140) else 0.05 for i in range(n)]
    q = [0.0] * n
    segs = segment_sequence(cfg, t, dt, p, q, [1.0] * n, [True] * n)
    assert [(s.start_idx, s.end_idx, s.reason) for s in segs] == [(20, 60, "presence_drop"), (100, 140, "presence_drop")], segs
    # back-to-back signs separated only by an end-prob peak
    p2 = [0.95 if 20 <= i < 140 else 0.05 for i in range(n)]
    q2 = [0.9 if i == 80 else 0.0 for i in range(n)]
    segs2 = segment_sequence(cfg, t, dt, p2, q2, [1.0] * n, [True] * n)
    assert [(s.start_idx, s.end_idx, s.reason) for s in segs2][:2] == [(20, 81, "cut"), (81, 140, "presence_drop")], segs2
    # a 4-pose tracker dropout inside a sign must NOT split it
    ok3 = [not (50 <= i < 54) for i in range(n)]
    p3 = [0.95 if 20 <= i < 50 or 54 <= i < 90 else (0.2 if 50 <= i < 54 else 0.05) for i in range(n)]
    segs3 = segment_sequence(cfg, t, dt, p3, q, [1.0] * n, ok3)
    assert len(segs3) == 1 and segs3[0].start_idx == 20, segs3
    # long stillness -> reset flag
    tr = SegmentTracker(cfg)
    flags = [tr.update(i / fps, 1 / fps, 0.0, 0.0, 0.0, False)[1] for i in range(200)]
    assert sum(flags) == 1 and flags.index(True) >= int(3.5 * fps) - 1
    print("[ok] tracker: boundaries, back-to-back cut, dropout tolerance, still-flush reset")

    # ---------------- test 2: shapes + gradients through YAP ----------------
    cfg = YAPConfig(d_model=64, summary_channels=16, hippo_order=8, entity_dim=64, n_mem_layers=2)
    ccfg = ConformerConfig(num_words=10, d_model=64, heads=4, layers=2, conv_kernel=3)
    model = YAPpingConformer(cfg, ccfg)
    pose, conf, valid, t = _synthetic_batch(2, 90, cfg)
    out = model.yap.forward_frames(pose, conf, valid, t)
    assert out["presence"].shape == (2, 90) and out["feat"].shape == (2, 90, 16) and out["layer_conf"].shape == (2, 90, 2)
    assert torch.isfinite(out["feat"]).all() and torch.isfinite(out["presence"]).all()
    clean = out["clean"]
    assert torch.isfinite(clean).all()
    print("[ok] YAP per-pose shapes, finite values; params =", count_parameters(model))

    # ---------------- test 3: Conformer with ground-truth segments + all losses ---------------
    gt_spans = [[(0.3, 1.0), (1.2, 2.0)], [(0.4, 1.5)]]
    segs_gt = [make_segments(t[b].tolist(), gt_spans[b], cfg) for b in range(2)]
    res = model(pose, conf, valid, t, segments=segs_gt)
    assert res["word_logits"].shape[0] == 2 and res["token_mask"].sum().item() == 3
    pt = torch.stack([presence_targets(t[b], gt_spans[b]) for b in range(2)])
    et = torch.stack([end_targets(t[b], gt_spans[b]) for b in range(2)])
    y = res["yap"]
    loss = (focal_presence_loss(y["presence"], pt) + F.binary_cross_entropy_with_logits(y["end"], et)
            + layer_conf_loss(y["layer_conf"], pt)
            + soft_count_loss(y["end"], torch.tensor([2, 1])) + min_gap_penalty(y["end"], 6)
            + dip_prior_loss(y["end"], y["speed"]))
    word_t = torch.tensor([[3, 5], [7, 0]])
    ce = F.cross_entropy(res["word_logits"].reshape(-1, 11), word_t.reshape(-1), reduction="none").view(2, -1)
    loss = loss + (ce * res["token_mask"]).sum() / res["token_mask"].sum()
    loss.backward()
    gnorm = sum(p.grad.abs().sum().item() for p in model.yap.core.layers.parameters() if p.grad is not None)
    assert gnorm > 0, "no gradient reached the YAP memory layers"
    assert model.yap.ctx_vec.grad is not None and model.yap.readout[0].weight.grad is not None
    # CTC path (stage 3) with predicted segments
    model.zero_grad()
    res2 = model(pose, conf, valid, t)
    ctc = sentence_ctc_loss(res2["word_logits"], res2["token_mask"], torch.tensor([[3, 5], [7, 1]]), torch.tensor([2, 2]))
    assert torch.isfinite(ctc)
    print("[ok] losses + gradients reach YAP layers, readout and ctx flag; CTC finite")

    # ---------------- test 4: LIVE stream == OFFLINE ----------------
    model.eval()
    pose, conf, valid, t = _synthetic_batch(1, 150, cfg, seed=3)
    with torch.no_grad():
        off = model.yap.forward_frames(pose, conf, valid, t)
        pr = torch.sigmoid(off["presence"])[0]
        thr = pr.median().item()                         # untrained heads: choose thresholds that yield segments
        cfg.on_thr, cfg.off_thr, cfg.on_hold_s, cfg.off_hold_s = thr, thr - 1e-5, 0.03, 0.1
        cfg.min_seg_s = 0.1
        segs_off = model.yap.segment_batch(off, t)[0]
        ents_off, meta_off, _ = model.yap.encode_segments(off, t, [segs_off])
        stream = YAPStream(model.yap)
        got, pres_live = [], []
        for k in range(150):
            got += stream.push(pose[0, k], conf[0, k], valid[0, k], float(t[0, k]))
            pres_live.append(stream.last_out["presence"][0])
        got += stream.flush()
    pres_live = torch.stack(pres_live)
    assert torch.allclose(pres_live, off["presence"][0], atol=1e-5), (pres_live - off["presence"][0]).abs().max()
    assert len(got) == len(segs_off) > 0, (len(got), len(segs_off))
    for e, s, vec in zip(got, segs_off, ents_off):
        assert abs(e.start_t - s.start_t) < 1e-9 and abs(e.end_t - s.end_t) < 1e-9 and e.reason == s.reason
        assert torch.allclose(e.vector, vec, atol=1e-4), (e.vector - vec).abs().max()
    print(f"[ok] live stream == offline: per-pose outputs identical, {len(got)} entities identical")
    # ---------------- test 5: configurable parts (holistic = 2 hands + upper body + face) ----------------
    hcfg = YAPConfig.holistic(d_model=64, summary_channels=16, hippo_order=8, entity_dim=64, n_mem_layers=2)
    hm = YAPpingConformer(hcfg, ccfg).eval()
    assert hcfg.total_joints == 21 + 21 + 9 + 28
    offs = hcfg.part_offsets()
    pose, conf, valid, t = _synthetic_batch(2, 90, hcfg, seed=5)
    core = hm.yap.core
    dts = frame_dts(t, hcfg.nominal_fps)
    # (a) declared feature sizes match what is really produced
    st = core.init_state(2, "cpu")
    with torch.no_grad():
        st["clean"], cl = core.cleaner.step(st["clean"], pose[:, 0], conf[:, 0], valid[:, 0], dts[:, 0])
        st["exp"], c0, r0, _ = core.expander.step(st["exp"], pose[:, 0], cl["x"], cl["v"], cl["trust"], cl["gap"], dts[:, 0])
    assert c0.shape[1] == core.expander.core_dim and r0.shape[1] == core.expander.rich_dim
    # (b) a part that is completely absent (face) must not silence the others
    v_noface = valid.clone()
    v_noface[:, :, offs["face"]:offs["face"] + 28] = False
    with torch.no_grad():
        o_nof = hm.yap.forward_frames(pose, conf, v_noface, t)
    assert (o_nof["hand_ok"] > 0.5).float().mean() > 0.9 and torch.isfinite(o_nof["presence"]).all()
    # (c) body frame: moving the whole body in the image must not change the features
    pv = valid.clone()
    pv[:, 0] = True                                           # all joints seen on pose 0 (clean start)
    shift = torch.tensor([0.31, -0.22, 0.13])

    def run_feats(ps):
        s = core.init_state(2, "cpu")
        cs, rs = [], []
        with torch.no_grad():
            for k in range(40):
                s["clean"], c_ = core.cleaner.step(s["clean"], ps[:, k], conf[:, k], pv[:, k], dts[:, k])
                s["exp"], co, ri, _ = core.expander.step(s["exp"], ps[:, k], c_["x"], c_["v"], c_["trust"], c_["gap"], dts[:, k])
                cs.append(co)
                rs.append(ri)
        return torch.stack(cs), torch.stack(rs)

    c_a, r_a = run_feats(pose)
    c_b, r_b = run_feats(pose + shift)
    assert (c_a - c_b).abs().max() < 1e-3 and (r_a - r_b).abs().max() < 1e-3, ((c_a - c_b).abs().max(), (r_a - r_b).abs().max())
    # (d) gradients + live == offline with the holistic layout
    hm.train()
    res = hm(pose, conf, valid, t, segments=[make_segments(t[b].tolist(), [(0.3, 1.0)], hcfg) for b in range(2)])
    res["word_logits"].sum().backward()
    assert all(p.grad is not None for p in hm.yap.core.layers[0].parameters())
    hm.eval()
    pose1, conf1, valid1, t1 = _synthetic_batch(1, 120, hcfg, seed=9)
    with torch.no_grad():
        off = hm.yap.forward_frames(pose1, conf1, valid1, t1)
        thr = torch.sigmoid(off["presence"])[0].median().item()
        hcfg.on_thr, hcfg.off_thr, hcfg.on_hold_s, hcfg.off_hold_s, hcfg.min_seg_s = thr, thr - 1e-5, 0.03, 0.1, 0.1
        stream = YAPStream(hm.yap)
        pres, got = [], []
        for k in range(120):
            got += stream.push(pose1[0, k], conf1[0, k], valid1[0, k], float(t1[0, k]))
            pres.append(stream.last_out["presence"][0])
        got += stream.flush()
        ref = hm.yap.segment_batch(off, t1)[0]
    assert torch.allclose(torch.stack(pres), off["presence"][0], atol=1e-5) and len(got) == len(ref) > 0
    # (e) assemble_parts: feed several separate models' outputs; a missing part becomes valid=False
    ap, ac, av = assemble_parts(hcfg, {"left_hand": (pose[:, :, :21], None, None),
                                       "upper_body": (pose[:, :, 42:51], None, None)})
    assert ap.shape == pose.shape and av[..., :21].all() and not av[..., 21:42].any() and not av[..., 51:].any()
    # (f) "any tracker": arbitrary custom layouts run too (2D-style single object, long chain, tiny part)
    ocfg = YAPConfig(parts=[PartSpec("obj", 5, motion=True),
                            PartSpec("tail", 40, parents=[-1] + list(range(39)), scale_pair=(0, 1)),
                            PartSpec("pair", 2, motion=False)],
                     d_model=64, summary_channels=16, hippo_order=8, entity_dim=64, n_mem_layers=2)
    om = YAPpingConformer(ocfg, ccfg)
    po, co_, vo, to = _synthetic_batch(2, 60, ocfg, seed=2)
    ro = om(po, co_, vo, to, segments=[make_segments(to[b].tolist(), [(0.2, 1.0)], ocfg) for b in range(2)])
    assert torch.isfinite(ro["word_logits"]).all()
    try:
        YAPConfig(parts=[PartSpec("a", 3), PartSpec("a", 3)])
        raise AssertionError("duplicate part names should be rejected")
    except ValueError:
        pass
    print("[ok] configurable parts: holistic dims, absent face, body-frame translation invariance, "
          "live==offline, assemble_parts, custom layouts, config validation")
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    _run_tests()
