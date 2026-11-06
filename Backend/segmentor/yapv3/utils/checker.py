"""
yapv3.utils.checker — universal YAP model collapse checker.

Works across every YAP model family:
  * YAP presence head only          (Stage 1)
  * YAP boundary / end head only    (Stage 1 boundary)
  * YAP gesture segmentation        (Stage 2 / Phase-3)
  * YAPpingConformer + CTC          (Stage 3 / Phase-4 real sentences)

Detects collapse from TWO independent angles:

  1. FORWARD-PASS CHECK  — run the model on real data and inspect the
     raw outputs: zero variance, all-blank presence, all-zero end logits,
     fully-zero output clips, meaningless predictions.

  2. HISTORY CHECK        — if a training history JSON is supplied, inspect
     the recorded metrics: NaN loss components, loss divergence (negative
     → positive), accuracy near random, blank ratio trending to 1.0,
     boundary F1 stuck at 0.0, plateaued / stagnant loss.

Returns a single collapse report dict with:
  collapsed:   bool        — overall collapse verdict
  severity:    str         — "none" | "warning" | "critical"
  reasons:     list[str]   — human-readable reasons
  checks:      dict        — per-check results (forward, history, ...)

Exposed functions:
  check_model(model, data_loader, device, ...)       → forward-pass collapse
  check_history(history_path | history_dict, ...)    → history-based collapse
  run_universal_check(model, data_loader, device, history_path=None, ...)
                                                       → combined report

All functions are importable and return plain dicts — no side effects except
the optional JSON report file.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from yap_conformer import YAP, YAPpingConformer


# ===========================================================================
# Thresholds (tunable)
# ===========================================================================
class CollapseThresholds:
    """Tunable collapse-detection thresholds."""

    # Forward-pass
    zero_variance_std: float = 1e-6          # output std below this = collapsed
    blank_ratio_max: float = 0.95            # P(blank) above this = collapsed
    zero_output_clip_frac_max: float = 0.5   # >half fully-zero clips = collapsed
    presence_mean_max: float = 0.05          # mean presence prob below this = collapsed
    end_mean_max: float = 0.02               # mean end prob below this = collapsed

    # History
    acc_max: float = 0.02                    # accuracy below this = collapsed
    f1_max: float = 1e-3                     # F1 below this = collapsed
    boundary_f1_max: float = 0.05            # boundary F1 below this = collapsed
    blank_ratio_trend_max: float = 0.70      # blank ratio above this = collapsed
    loss_divergence_threshold: float = 0.0   # loss going from neg → pos


# ===========================================================================
# Helpers
# ===========================================================================
def _safe_float(x) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _is_nan_or_inf(v) -> bool:
    try:
        return not math.isfinite(float(v))
    except (TypeError, ValueError):
        return True


def _load_history(history_path: Union[Path, str, dict, None]) -> Optional[dict]:
    """Load a history JSON or accept a dict directly."""
    if history_path is None:
        return None
    if isinstance(history_path, dict):
        return history_path
    path = Path(history_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# ===========================================================================
# 1. FORWARD-PASS COLLAPSE CHECK
# ===========================================================================
@torch.no_grad()
def check_model_forward(
    model,
    data_loader: DataLoader,
    device: torch.device,
    max_batches: int = 25,
    thresholds: Optional[CollapseThresholds] = None,
) -> dict:
    """
    Run the model on real data and check for forward-pass collapse.

    Works with YAP (presence/end heads) and YAPpingConformer (word_logits).
    Inspects: output variance, blank/presence ratio, fully-zero clips,
    end-head activity.

    Returns dict with:
      collapsed, severity, reasons, checks
    """
    th = thresholds or CollapseThresholds()
    reasons = []
    checks = {}

    model.eval()
    model.to(device)

    total_frames = 0
    total_blank_frames = 0
    total_presence_sum = 0.0
    total_end_sum = 0.0
    all_outputs_std = []
    fully_zero_clips = 0
    total_clips = 0
    word_logit_variance = []
    end_activity = []  # fraction of frames where end prob > 0.5

    batches_seen = 0
    for batch in data_loader:
        if batches_seen >= max_batches:
            break
        batches_seen += 1

        # Unpack batch flexibly: (pose, ...) or (pose, conf, valid, t, ...)
        if isinstance(batch, (list, tuple)):
            pose = batch[0]
        else:
            pose = batch

        B = pose.shape[0]
        J = 53
        # pose: (B, T, 163) → slice coords → (B, T, J, 3)
        pose_coords = pose.to(device)[:, :, :159].view(B, -1, J, 3)
        T = pose_coords.shape[1]
        conf = torch.ones(B, T, J, device=device)
        valid = torch.ones(B, T, J, dtype=torch.bool, device=device)
        t = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1) / 30.0

        if isinstance(model, YAPpingConformer):
            out = model(pose_coords, conf, valid, t)
            # Word logits variance (Conformer output)
            wl = out.get("word_logits", None)
            if wl is not None:
                word_logit_variance.append(wl.detach().float().var().item())
            # YAP sub-outputs
            yap_out = out.get("yap", out)
        elif isinstance(model, YAP):
            yap_out = model.forward_frames(pose_coords, conf, valid, t)
        else:
            # Generic: try forward_frames, fall back to __call__
            if hasattr(model, "forward_frames"):
                yap_out = model.forward_frames(pose_coords, conf, valid, t)
            else:
                yap_out = model(pose_coords, conf, valid, t)

        presence = yap_out.get("presence", None)
        end_logits = yap_out.get("end", None)

        # Presence / blank analysis
        if presence is not None:
            pres_probs = torch.sigmoid(presence.detach())
            pmask = torch.ones_like(pres_probs)  # all frames valid in this synthetic loader
            total_frames += pres_probs.numel()
            total_blank_frames += (pres_probs < 0.5).sum().item()
            total_presence_sum += pres_probs.sum().item()
            all_outputs_std.append(pres_probs.float().std().item())

        # End-head analysis
        if end_logits is not None:
            end_probs = torch.sigmoid(end_logits.detach())
            total_end_sum += end_probs.sum().item()
            end_activity.append((end_probs > 0.5).float().mean().item())
            all_outputs_std.append(end_probs.float().std().item())

        # Fully-zero clip detection (presence all zero)
        if presence is not None:
            pres_probs = torch.sigmoid(presence.detach())
            for b in range(B):
                total_clips += 1
                if pres_probs[b].max().item() < 0.01:
                    fully_zero_clips += 1

    checks = {}
    collapsed = False
    severity = "none"

    # ---- Output variance ----
    if all_outputs_std:
        overall_std = float(np.mean(all_outputs_std))
        checks["output_std"] = overall_std
        if overall_std < th.zero_variance_std:
            collapsed = True
            reasons.append(f"output std {overall_std:.2e} below {th.zero_variance_std:.0e} — no variance")

    # ---- Blank ratio (presence) ----
    blank_ratio = total_blank_frames / max(total_frames, 1)
    checks["blank_ratio"] = blank_ratio
    if blank_ratio > th.blank_ratio_max:
        collapsed = True
        reasons.append(f"blank ratio {blank_ratio:.3f} above {th.blank_ratio_max:.2f} — predicting blank almost everywhere")

    # ---- Mean presence ----
    mean_presence = total_presence_sum / max(total_frames, 1)
    checks["mean_presence"] = mean_presence
    if mean_presence < th.presence_mean_max and total_frames > 0:
        collapsed = True
        reasons.append(f"mean presence {mean_presence:.4f} below {th.presence_mean_max:.2f} — near total blank")

    # ---- End-head activity ----
    if end_activity:
        mean_end_activity = float(np.mean(end_activity))
        checks["end_activity"] = mean_end_activity
        if mean_end_activity < th.end_mean_max:
            collapsed = True
            reasons.append(f"end-head active frames {mean_end_activity:.4f} below {th.end_mean_max:.2f} — no cuts detected")

    # ---- Fully zero clips ----
    zero_clip_frac = fully_zero_clips / max(total_clips, 1)
    checks["zero_clip_fraction"] = zero_clip_frac
    if zero_clip_frac > th.zero_output_clip_frac_max and total_clips > 0:
        collapsed = True
        reasons.append(f"{zero_clip_frac:.1%} of clips fully zero — model outputting nothing")

    # ---- Word logit variance (Conformer) ----
    if word_logit_variance:
        mean_wl_var = float(np.mean(word_logit_variance))
        checks["word_logit_variance"] = mean_wl_var
        if mean_wl_var < th.zero_variance_std:
            collapsed = True
            reasons.append(f"word logit variance {mean_wl_var:.2e} — Conformer output collapsed")

    # ---- Severity ----
    if collapsed:
        if len(reasons) >= 3 or blank_ratio > 0.95:
            severity = "critical"
        else:
            severity = "warning"

    return {
        "collapsed": collapsed,
        "severity": severity,
        "reasons": reasons,
        "checks": checks,
        "type": "forward_pass",
        "batches_evaluated": batches_seen,
    }


# ===========================================================================
# 2. HISTORY-BASED COLLAPSE CHECK
# ===========================================================================
def check_history(
    history: Union[Path, str, dict, None],
    thresholds: Optional[CollapseThresholds] = None,
) -> dict:
    """
    Inspect a training history JSON (or dict) for collapse signatures.

    Checks: NaN loss components, loss divergence (negative → positive),
    accuracy near random, F1 near zero, boundary F1 stuck at 0.0,
    blank ratio trending to 1.0, loss plateau/stagnation.

    Returns dict with:
      collapsed, severity, reasons, checks
    """
    th = thresholds or CollapseThresholds()
    reasons = []
    checks = {}

    hist = _load_history(history)
    if hist is None:
        return {
            "collapsed": False,
            "severity": "none",
            "reasons": ["no history available to check"],
            "checks": {},
            "type": "history",
        }

    # Normalize: history may be a dict of lists or a list of epoch dicts
    if isinstance(hist, list):
        # List of per-epoch dicts → transpose to dict of lists
        transposed: dict = {}
        for epoch_dict in hist:
            for k, v in epoch_dict.items():
                transposed.setdefault(k, []).append(v)
        hist = transposed

    def get_series(key: str) -> list:
        return hist.get(key, [])

    def last_val(key: str) -> float:
        s = get_series(key)
        return _safe_float(s[-1]) if s else float("nan")

    def first_val(key: str) -> float:
        s = get_series(key)
        return _safe_float(s[0]) if s else float("nan")

    # ---- NaN loss components ----
    nan_keys = []
    for key in ("train_pres", "train_end", "train_layer", "train_count",
                "val_pres", "val_end", "val_layer", "val_count",
                "train_loss", "val_loss", "loss"):
        series = get_series(key)
        nan_count = sum(1 for v in series if _is_nan_or_inf(v))
        if nan_count > 0:
            nan_keys.append(f"{key} ({nan_count} NaN)")
    checks["nan_components"] = nan_keys
    if nan_keys:
        collapsed = True
        reasons.append(f"NaN loss components: {', '.join(nan_keys)}")
    else:
        collapsed = False

    # ---- Loss divergence (negative → positive) ----
    train_loss_series = get_series("train_loss")
    if len(train_loss_series) >= 2:
        first = _safe_float(train_loss_series[0])
        last = _safe_float(train_loss_series[-1])
        checks["train_loss_first"] = first
        checks["train_loss_last"] = last
        if math.isfinite(first) and math.isfinite(last):
            if first < 0 and last > 0:
                collapsed = True
                reasons.append(f"loss diverged: {first:.3f} → {last:.3f} (negative to positive)")
            elif last > 0 and last > abs(first) * 3:
                collapsed = True
                reasons.append(f"loss grew: {first:.3f} → {last:.3f}")

    # ---- Accuracy near random ----
    for acc_key in ("train_acc", "val_acc", "accuracy"):
        series = get_series(acc_key)
        if series:
            last_acc = _safe_float(series[-1])
            checks[f"{acc_key}_last"] = last_acc
            if math.isfinite(last_acc) and last_acc < th.acc_max:
                collapsed = True
                reasons.append(f"{acc_key} {last_acc:.4f} below {th.acc_max:.2f} — near random")
            break

    # ---- F1 near zero ----
    for f1_key in ("train_f1", "val_f1", "f1"):
        series = get_series(f1_key)
        if series:
            last_f1 = _safe_float(series[-1])
            checks[f"{f1_key}_last"] = last_f1
            if math.isfinite(last_f1) and last_f1 < th.f1_max:
                collapsed = True
                reasons.append(f"{f1_key} {last_f1:.2e} below {th.f1_max:.0e} — F1 near zero")
            break

    # ---- Boundary F1 stuck at 0 ----
    for bf_key in ("boundary_f1_200ms", "val_boundary_f1_200ms", "boundary_f1"):
        series = get_series(bf_key)
        if series:
            last_bf = _safe_float(series[-1])
            checks[f"{bf_key}_last"] = last_bf
            if math.isfinite(last_bf) and last_bf < th.boundary_f1_max:
                collapsed = True
                reasons.append(f"{bf_key} {last_bf:.4f} below {th.boundary_f1_max:.2f} — no boundaries detected")
            break

    # ---- Blank ratio trending to 1.0 ----
    for br_key in ("blank", "val_blank", "blank_ratio"):
        series = get_series(br_key)
        if series:
            last_br = _safe_float(series[-1])
            first_br = _safe_float(series[0]) if len(series) > 1 else last_br
            checks[f"{br_key}_last"] = last_br
            checks[f"{br_key}_first"] = first_br
            if math.isfinite(last_br) and last_br > th.blank_ratio_trend_max:
                collapsed = True
                trend = f" (was {first_br:.2f})" if len(series) > 1 else ""
                reasons.append(f"blank ratio {last_br:.2f}{trend} — model predicting blank most frames")
            break

    # ---- Loss plateau / stagnation ----
    for loss_key in ("train_loss", "val_loss", "loss"):
        series = get_series(loss_key)
        if len(series) >= 10:
            finite_vals = [_safe_float(v) for v in series if math.isfinite(_safe_float(v))]
            if len(finite_vals) >= 10:
                recent = finite_vals[-10:]
                recent_std = float(np.std(recent)) if len(recent) > 1 else 0.0
                checks[f"{loss_key}_recent_std"] = recent_std
                if recent_std < 1e-4:
                    collapsed = True
                    reasons.append(f"{loss_key} plateaued (std {recent_std:.2e}) over last 10 epochs")
                break

    # ---- Severity ----
    if collapsed:
        severity = "critical" if len(reasons) >= 3 else "warning"
    else:
        severity = "none"

    return {
        "collapsed": collapsed,
        "severity": severity,
        "reasons": reasons,
        "checks": checks,
        "type": "history",
        "epochs_recorded": len(get_series("train_loss")) if get_series("train_loss") else 0,
    }


# ===========================================================================
# 3. UNIVERSAL COMBINED CHECK
# ===========================================================================
def run_universal_check(
    model: Optional[torch.nn.Module] = None,
    data_loader: Optional[DataLoader] = None,
    device: Optional[torch.device] = None,
    history: Union[Path, str, dict, None] = None,
    max_batches: int = 25,
    thresholds: Optional[CollapseThresholds] = None,
) -> dict:
    """
    Run a universal collapse check on a YAP model.

    Combines forward-pass check (if model + data_loader given) and
    history check (if history given). Returns a single unified report.

    Parameters
    ----------
    model : YAP or YAPpingConformer, or None to skip forward check
    data_loader : DataLoader of real data, or None to skip forward check
    device : torch.device, or None to auto-select cuda/cpu
    history : path to history JSON, dict, or None to skip history check
    max_batches : max batches for forward check
    thresholds : CollapseThresholds, or None for defaults

    Returns
    -------
    dict with keys:
      collapsed, severity, reasons, forward_check, history_check, overall_score
    """
    th = thresholds or CollapseThresholds()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    forward_result = None
    history_result = None

    if model is not None and data_loader is not None:
        forward_result = check_model_forward(
            model, data_loader, device,
            max_batches=max_batches, thresholds=th,
        )

    if history is not None:
        history_result = check_history(history, thresholds=th)

    # Combine
    all_reasons = []
    sub_results = {}
    collapsed_any = False
    severity_order = {"none": 0, "warning": 1, "critical": 2}
    max_severity = "none"

    if forward_result is not None:
        sub_results["forward_pass"] = forward_result
        all_reasons.extend([f"[forward] {r}" for r in forward_result["reasons"]])
        if forward_result["collapsed"]:
            collapsed_any = True
        if severity_order[forward_result["severity"]] > severity_order[max_severity]:
            max_severity = forward_result["severity"]

    if history_result is not None:
        sub_results["history"] = history_result
        all_reasons.extend([f"[history] {r}" for r in history_result["reasons"]])
        if history_result["collapsed"]:
            collapsed_any = True
        if severity_order[history_result["severity"]] > severity_order[max_severity]:
            max_severity = history_result["severity"]

    # Overall score: 0 = healthy, 1 = warning, 2 = critical
    overall_score = severity_order[max_severity]

    return {
        "collapsed": collapsed_any,
        "severity": max_severity,
        "overall_score": overall_score,
        "reasons": all_reasons,
        "forward_check": forward_result,
        "history_check": history_result,
        "thresholds": {
            "zero_variance_std": th.zero_variance_std,
            "blank_ratio_max": th.blank_ratio_max,
            "acc_max": th.acc_max,
            "f1_max": th.f1_max,
            "boundary_f1_max": th.boundary_f1_max,
            "blank_ratio_trend_max": th.blank_ratio_trend_max,
        },
    }


# ===========================================================================
# Convenience: save report to JSON
# ===========================================================================
def save_check_report(report: dict, out_path: Union[Path, str]) -> None:
    """Save a collapse check report to JSON (with numpy → python conversion)."""
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(report), indent=2), encoding="utf-8")


__all__ = [
    "CollapseThresholds",
    "check_model_forward",
    "check_history",
    "run_universal_check",
    "save_check_report",
]
