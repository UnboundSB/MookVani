"""
yapv3.utils.trainer — phase-3 training loop for YAP gesture segmentation.

Trains YAP (presence + end heads only, everything else frozen) on stitched
word-sentences.  Reports boundary F1 / precision / recall using the production
SegmentTracker, not just thresholded logits.

No Conformer, no CTC, no word-classifier head — pure gesture-boundary
pretraining (Stage 2 / Phase A style).
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import (AugmentationConfig, SentenceClipDataset, build_sentence_dataset,
                   collate_sentence_batch)
from .losses import build_end_targets, build_presence_targets, total_segmentation_loss
from .model import YAPModelWrapper


# ===========================================================================
# Boundary metrics (production SegmentTracker)
# ===========================================================================
def _boundary_matches(pred_times, true_times, tolerance_s):
    """One-to-one time matching; duplicate detections count as false positives."""
    pairs = sorted((abs(p - g), i, j)
                   for i, p in enumerate(pred_times)
                   for j, g in enumerate(true_times)
                   if abs(p - g) <= tolerance_s)
    used_p, used_g = set(), set()
    for _, i, j in pairs:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
    return len(used_p)


def _prf(matches, predicted, truth):
    precision = matches / predicted if predicted else 0.0
    recall = matches / truth if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "matched": matches, "predicted": predicted, "ground_truth": truth}


def evaluate_boundaries(yap: YAPModelWrapper, output, times, mask, spans,
                        tolerance_s: float = 0.2):
    """Evaluate the REAL production SegmentTracker on a batch.

    Returns dict with start/end/combined boundary metrics, IoU, count MAE,
    and cut reasons.
    """
    lengths = mask.sum(dim=1).long()
    predicted = yap.segment_batch(output, times, lengths)

    result = dict(matches=0, predicted=0, truth=0,
                  start_matches=0, start_predicted=0, start_truth=0,
                  end_matches=0, end_predicted=0, end_truth=0,
                  iou_sum=0.0, iou_count=0, reasons={})
    count_mae_total = 0.0

    for segments, gt in zip(predicted, spans):
        # EOS is artificial — do not count as a learned cut
        segments = [seg for seg in segments if getattr(seg, "reason", "") != "eos"]
        pred_starts = [float(seg.start_t) for seg in segments]
        pred_ends = [float(seg.end_t) for seg in segments]
        gt_starts = [float(s) for s, _ in gt]
        gt_ends = [float(e) for _, e in gt]

        for kind, ps, gs in (("start", pred_starts, gt_starts),
                             ("end", pred_ends, gt_ends)):
            m = _boundary_matches(ps, gs, tolerance_s)
            result[f"{kind}_matches"] += m
            result[f"{kind}_predicted"] += len(ps)
            result[f"{kind}_truth"] += len(gs)
            result["matches"] += m
            result["predicted"] += len(ps)
            result["truth"] += len(gs)

        # Segment IoU between predicted and GT intervals
        for ps_, pe in zip(pred_starts, pred_ends):
            best = 0.0
            for gs_, ge in zip(gt_starts, gt_ends):
                inter = max(0.0, min(pe, ge) - max(ps_, gs_))
                uni = max(pe, ge) - min(ps_, gs_)
                if uni > 0:
                    best = max(best, inter / uni)
            result["iou_sum"] += best
            result["iou_count"] += 1

        for seg in segments:
            r = getattr(seg, "reason", "unknown")
            result["reasons"][r] = result["reasons"].get(r, 0) + 1

    # Count MAE: sigmoid(end) peaks above cut_thr
    q = torch.sigmoid(output["end"])
    B, T = q.shape
    for b in range(B):
        n = int(lengths[b].item())
        pred = int((q[b, :n] > 0.7).sum().item())
        true = len(spans[b])
        count_mae_total += abs(pred - true)

    return result, count_mae_total / max(B, 1)


# ===========================================================================
# Training step
# ===========================================================================
def train_epoch(yap, optimizer, loader, device, epoch, total_epochs):
    """One training epoch. Returns (avg_loss, parts_avg)."""
    yap.train()
    loss_names = ("presence", "end", "layer", "count")
    train_sum = 0.0
    train_parts = {k: 0.0 for k in loss_names}
    n_batches = 0

    pbar = tqdm(loader, desc=f"Phase3 {epoch + 1:03d}/{total_epochs:03d} [train]",
                leave=False)
    for batch in pbar:
        pose, conf, valid, times, mask, spans, labels = batch
        pose, conf, valid, times, mask = (x.to(device) for x in
                                          (pose, conf, valid, times, mask))
        B, T = mask.shape

        # Build targets per sequence
        presence_tgt = torch.zeros(B, T, device=device)
        end_tgt = torch.zeros(B, T, device=device)
        counts = torch.zeros(B, device=device)
        for b, ss in enumerate(spans):
            n = int(mask[b].sum().item())
            presence_tgt[b, :n] = build_presence_targets(times[b, :n], ss)
            end_tgt[b, :n] = build_end_targets(times[b, :n], ss)
            counts[b] = len(ss)

        output = yap.forward_frames(pose, conf, valid, times)
        total, parts = total_segmentation_loss(output, presence_tgt, end_tgt,
                                               counts, mask.float())

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(yap.parameters(), 1.0)
        optimizer.step()

        train_sum += total.item()
        for k in loss_names:
            train_parts[k] += parts[k]
        n_batches += 1
        pbar.set_postfix({k: f"{v / max(1, n_batches):.4f}" for k, v in train_parts.items()})

    pbar.close()
    avg = {k: train_parts[k] / max(n_batches, 1) for k in loss_names}
    return train_sum / max(n_batches, 1), avg


# ===========================================================================
# Validation step
# ===========================================================================
def validate(yap, loader, device):
    """One validation epoch. Returns (avg_loss, parts_avg, boundary_metrics)."""
    yap.eval()
    loss_names = ("presence", "end", "layer", "count")
    val_sum = 0.0
    metrics = {k: 0.0 for k in loss_names}
    boundary_totals = dict(matches=0, predicted=0, truth=0,
                           start_matches=0, start_predicted=0, start_truth=0,
                           end_matches=0, end_predicted=0, end_truth=0,
                           iou_sum=0.0, iou_count=0, reasons={})
    count_mae_total = 0.0
    count_batches = 0

    with torch.no_grad():
        for batch in loader:
            pose, conf, valid, times, mask, spans, labels = batch
            pose, conf, valid, times, mask = (x.to(device) for x in
                                              (pose, conf, valid, times, mask))
            total, parts, output = None, None, None
            # Reuse total_segmentation_loss for parts; return_output not needed
            presence_tgt = torch.zeros(pose.shape[0], pose.shape[1], device=device)
            end_tgt = torch.zeros(pose.shape[0], pose.shape[1], device=device)
            counts = torch.zeros(pose.shape[0], device=device)
            for b, ss in enumerate(spans):
                n = int(mask[b].sum().item())
                presence_tgt[b, :n] = build_presence_targets(times[b, :n], ss)
                end_tgt[b, :n] = build_end_targets(times[b, :n], ss)
                counts[b] = len(ss)

            output = yap.forward_frames(pose, conf, valid, times)
            total, parts = total_segmentation_loss(output, presence_tgt, end_tgt,
                                                   counts, mask.float())
            val_sum += total.item()
            for k in metrics:
                metrics[k] += parts[k]

            bc, cm = evaluate_boundaries(yap, output, times, mask, spans)
            count_mae_total += cm
            count_batches += 1
            for k in boundary_totals:
                if k == "reasons":
                    for reason, n in bc[k].items():
                        boundary_totals[k][reason] = boundary_totals[k].get(reason, 0) + n
                else:
                    boundary_totals[k] += bc[k]

    n_batches = max(len(loader), 1)
    val_avg = {k: metrics[k] / n_batches for k in metrics}
    boundary_all = _prf(boundary_totals["matches"], boundary_totals["predicted"],
                        boundary_totals["truth"])
    boundary_start = _prf(boundary_totals["start_matches"],
                          boundary_totals["start_predicted"],
                          boundary_totals["start_truth"])
    boundary_end = _prf(boundary_totals["end_matches"],
                        boundary_totals["end_predicted"],
                        boundary_totals["end_truth"])
    iou = boundary_totals["iou_sum"] / max(boundary_totals["iou_count"], 1)
    cm = count_mae_total / max(count_batches, 1)

    return (val_sum / n_batches, val_avg,
            {"boundary_f1_200ms": boundary_all["f1"],
             "boundary_precision_200ms": boundary_all["precision"],
             "boundary_recall_200ms": boundary_all["recall"],
             "boundary_start_f1_200ms": boundary_start["f1"],
             "boundary_end_f1_200ms": boundary_end["f1"],
             "boundary_iou": iou,
             "count_mae": cm,
             "boundary_cut_reasons": boundary_totals["reasons"],
             "matched": boundary_all["matched"],
             "predicted": boundary_all["predicted"],
             "ground_truth": boundary_all["ground_truth"]})


# ===========================================================================
# Main training loop
# ===========================================================================
def run_phase3(data_root: Path | str, classes_path: Path | str,
               out_dir: Path | str = "models/yap_phase3",
               epochs: int = 10, batch_size: int = 4,
               train_sentences: int = 4000, val_sentences: int = 400,
               max_per_class: Optional[int] = None,
               max_len: int = 900, fps: float = 30.0,
               lr: float = 1e-4, seed: int = 42,
               d_model: int = 128, n_mem_layers: int = 3,
               summary_channels: int = 32, hippo_order: int = 16,
               entity_dim: int = 256,
               init: Optional[Path | str] = None,
               cpu: bool = False,
               aug: Optional[AugmentationConfig] = None):
    """Run phase-3 gesture-segmentation pretraining.

    Stitches 2-5 random words per sentence, trains presence + end heads,
    validates with boundary F1 / IoU / count MAE.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    print(f"[DEVICE] {device}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- datasets (80/20 split handled inside build_sentence_dataset via separate counts) ----
    aug_train = aug or AugmentationConfig(seed=seed)
    aug_val = AugmentationConfig(seed=seed, mirror=False, shift=False,
                                  tilt=False, time_warp=False,
                                  hold_last_valid=True)

    train_ds = build_sentence_dataset(data_root, classes_path, train_sentences,
                                      fps=fps, max_len=max_len, aug=aug_train,
                                      seed=seed, training=True,
                                      max_per_class=max_per_class)
    val_ds = build_sentence_dataset(data_root, classes_path, val_sentences,
                                    fps=fps, max_len=max_len, aug=aug_val,
                                    seed=seed + 7, training=False,
                                    max_per_class=max_per_class)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_sentence_batch, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_sentence_batch, num_workers=0)

    print(f"[DATA] Train sentences: {len(train_ds)} | Val sentences: {len(val_ds)}")
    print(f"[DATA] Classes available: {len(train_ds.classes)}")
    print(f"[DATA] Classes with clips: {len(train_ds.clips_by_class)}")

    # ---- model ----
    yap = YAPModelWrapper.from_checkpoint(init, device=device,
                                          d_model=d_model, n_mem_layers=n_mem_layers,
                                          summary_channels=summary_channels,
                                          hippo_order=hippo_order,
                                          entity_dim=entity_dim)

    # Freeze everything except presence + end heads
    for p in yap.parameters():
        p.requires_grad = False
    # Unfreeze presence_head and end_head inside YAPCore
    for p in yap.yap.core.presence_head.parameters():
        p.requires_grad = True
    for p in yap.yap.core.end_head.parameters():
        p.requires_grad = True

    n_trainable = sum(p.numel() for p in yap.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in yap.parameters())
    print(f"[FREEZE] Trainable: {n_trainable:,} / {n_total:,} "
          f"({100.0 * n_trainable / n_total:.2f}%) — presence + end heads only")

    optimizer = torch.optim.AdamW([p for p in yap.parameters() if p.requires_grad],
                                  lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best = math.inf
    history = []

    for epoch in range(epochs):
        train_ds.epoch = epoch
        val_ds.epoch = epoch

        tr_loss, tr_parts = train_epoch(yap, optimizer, train_loader, device,
                                        epoch, epochs)
        v_loss, v_parts, bm = validate(yap, val_loader, device)

        history.append({"epoch": epoch + 1,
                        "train": tr_loss, "val": v_loss,
                        **{f"train_{k}": tr_parts[k] for k in tr_parts},
                        **{f"val_{k}": v_parts[k] for k in v_parts},
                        **bm,
                        "lr": scheduler.get_last_lr()[0]})

        improved = v_loss < best
        if improved:
            best = v_loss
            torch.save({"yap": yap.state_dict(), "epoch": epoch + 1,
                        "history": history}, out / "yap_phase3_best.pth")

        print("\n" + "=" * 76)
        print(f"  YAP PHASE-3 GESTURE SEGMENTATION | EPOCH {epoch + 1:03d}/{epochs:03d} | "
              f"LR {scheduler.get_last_lr()[0]:.2e}")
        print("-" * 76)
        print(f'  {"Loss":<20} {"Train":>10} {"Val":>10}')
        print("-" * 76)
        for k in ("presence", "end", "layer", "count"):
            print(f'  {k:<20} {tr_parts[k]:>10.4f} {v_parts[k]:>10.4f}')
        print(f'  {"TOTAL":<20} {tr_loss:>10.4f} {v_loss:>10.4f}')
        print("-" * 76)
        print(f'  Boundary F1 (200ms): {bm["boundary_f1_200ms"]:.4f} | '
              f'Precision: {bm["boundary_precision_200ms"]:.4f} | '
              f'Recall: {bm["boundary_recall_200ms"]:.4f}')
        print(f'  Start F1: {bm["boundary_start_f1_200ms"]:.4f} | '
              f'End F1: {bm["boundary_end_f1_200ms"]:.4f}')
        print(f'  Segment IoU: {bm["boundary_iou"]:.4f} | Count MAE: {bm["count_mae"]:.3f}')
        print(f'  Matched: {bm["matched"]} / Predicted: {bm["predicted"]} / '
              f'GT: {bm["ground_truth"]}')
        print(f'  Cut reasons: {bm["boundary_cut_reasons"]}')
        print(f'  Best val: {best:.4f}' +
              (' [new checkpoint saved]' if improved else ''))
        print("=" * 76 + "\n", flush=True)

        # Resume checkpoint
        torch.save({"yap": yap.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1, "best": best, "history": history},
                   out / "yap_phase3_resume.pth")
        (out / "history.json").write_text(json.dumps(history, indent=2),
                                          encoding="utf-8")
        scheduler.step()

    # Final weights
    torch.save(yap.state_dict(), out / "yap_phase3_final.pth")
    print(f"[DONE] Best validation loss = {best:.4f}")
    return best, history
