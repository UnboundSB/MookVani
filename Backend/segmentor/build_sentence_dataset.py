"""
build_sentence_dataset.py
=========================

Per-sentence processed dataset for YAPpingConformer (Conformer-attached Yap)
fine-tuning on real sentences.

Runs YAPpingConformer (YAP front-end + Conformer + CTC head) on each real
sentence tensor, detects gesture segments/boundaries via the production
SegmentTracker, and shows the detected boundaries to the user for accept/reject.

Accepted sentences are saved with their entity vectors (processed.pt) so the
Conformer-attached Yap can be fine-tuned on real sentence boundaries.

Structure under --out_root (segmentor/data_sentence/):

    <sentence_slug>/
        1/
            sentence.pt   # raw (T, 163) tensor for this take
            processed.pt  # entity vectors (K, entity_dim) — YAP+Conformer output
            segments.json # detected segments with start/end times + accept flag
        2/
            ...

Writes mapping.json: {"sentence_slug": ["1", "2", ...], ...}

Usage (from D:\MookVani\Backend):

    # Build sentence dataset with interactive accept/reject:
    .\venv\Scripts\python.exe segmentor\build_sentence_dataset.py `
        --init segmentor/models/yap_stage1/yap_noise_robust.pth `
        --out_root segmentor/data_sentence `
        --max_sentences 50 `
        --interactive

    # Non-interactive: auto-accept all detected segments:
    .\venv\Scripts\python.exe segmentor\build_sentence_dataset.py `
        --init segmentor/models/yap_stage1/yap_noise_robust.pth `
        --out_root segmentor/data_sentence `
        --max_sentences 50 `
        --auto-accept
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.checkpoint_utils import load_flexible

from yap_conformer import (
    YAPConfig, YAPpingConformer, ConformerConfig,
    Segment,
)

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INDEX = os.path.join(BACKEND_DIR, "data", "tensors_sentence_level", "index.json")
DEFAULT_CLASSES = os.path.join(BACKEND_DIR, "models", "word_vocabulary", "classes.json")
DEFAULT_INIT = os.path.join(BACKEND_DIR, "segmentor", "models", "yap_stage1", "yap_noise_robust.pth")
DEFAULT_OUT_ROOT = os.path.join(BACKEND_DIR, "segmentor", "data_sentence")

FPS = 30.0
MAX_LEN = 900


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def slugify(s: str) -> str:
    """Convert sentence text to a filesystem-safe slug."""
    return "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in s).strip().replace(" ", "_")


def split_163_to_pose(x163: torch.Tensor, J: int):
    T = x163.shape[0]
    coords = x163[:, :J * 3].contiguous().view(T, J, 3)
    conf = torch.ones(T, J)
    valid = torch.ones(T, J, dtype=torch.bool)
    return coords, conf, valid


def load_sentence_index(index_path: Path) -> list[dict]:
    with open(index_path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------
# Sentence detection + display
# ----------------------------------------------------------------------------
def detect_segments(yap: YAPpingConformer, pose: torch.Tensor, conf: torch.Tensor,
                    valid: torch.Tensor, t: torch.Tensor, device: torch.device,
                    max_len: int = MAX_LEN):
    """
    Run Yap front-end + segment_batch on a sentence tensor.
    Returns (segments, output_dict) where segments is a list of Segment objects.
    """
    T = pose.shape[1]
    if T > max_len:
        pose = pose[:, :max_len]
        conf = conf[:, :max_len]
        valid = valid[:, :max_len]
        t = t[:, :max_len]

    pose = pose.to(device)
    conf = conf.to(device)
    valid = valid.to(device)
    t = t.to(device)

    with torch.no_grad():
        # YAPpingConformer wraps YAP; per-pose output lives on yap.yap
        out = yap.yap.forward_frames(pose, conf, valid, t)
        segments = yap.yap.segment_batch(out, t)[0]

    return segments, out


def display_segments(sentence: str, segments: list[Segment], t_tensor: torch.Tensor,
                     fps: float = FPS) -> None:
    """Print detected segments in a human-readable format."""
    print(f"\n{'='*70}")
    print(f"  SENTENCE: {sentence}")
    print(f"{'='*70}")
    if not segments:
        print("  [NO SEGMENTS DETECTED]")
    else:
        print(f"  Detected {len(segments)} segment(s):")
        for i, seg in enumerate(segments, 1):
            start_frame = int(seg.start_t * fps)
            end_frame = int(seg.end_t * fps)
            start_s = seg.start_t
            end_s = seg.end_t
            duration = end_s - start_s
            print(f"    [{i}] frames {start_frame:4d}-{end_frame:4d}  "
                  f"({start_s:5.2f}s - {end_s:5.2f}s, {duration:4.2f}s)  "
                  f"reason={seg.reason}")
    print(f"{'='*70}")


# ----------------------------------------------------------------------------
# Interactive accept/reject
# ----------------------------------------------------------------------------
def prompt_accept(sentence: str, segments: list[Segment]) -> bool:
    """Ask user to accept or reject this sentence's detected segments.
    Returns True if accepted, False if rejected."""
    while True:
        choice = input(f"  Accept this sentence's segments? (y/n/s=skip): ").strip().lower()
        if choice in ("y", "yes"):
            return True
        elif choice in ("n", "no"):
            return False
        elif choice in ("s", "skip"):
            return "skip"
        else:
            print("    Please enter y (accept), n (reject), or s (skip).")


# ----------------------------------------------------------------------------
# Save accepted sentence
# ----------------------------------------------------------------------------
def save_sentence(sentence: str, take_idx: int, x163: torch.Tensor,
                  entities: torch.Tensor, segments: list[Segment],
                  out_root: Path, fps: float = FPS) -> str:
    """Save sentence.pt, processed.pt, segments.json. Returns take slug."""
    slug = slugify(sentence)
    take_dir = out_root / slug / str(take_idx)
    take_dir.mkdir(parents=True, exist_ok=True)

    # Save raw sentence tensor
    torch.save(x163, take_dir / "sentence.pt")

    # Save entity vectors
    torch.save(entities.cpu(), take_dir / "processed.pt")

    # Save segment info
    seg_data = []
    for seg in segments:
        seg_data.append({
            "start_t": seg.start_t,
            "end_t": seg.end_t,
            "start_frame": int(seg.start_t * fps),
            "end_frame": int(seg.end_t * fps),
            "reason": seg.reason,
            "confidence": getattr(seg, "confidence", 1.0),
        })
    with open(take_dir / "segments.json", "w", encoding="utf-8") as f:
        json.dump(seg_data, f, indent=2)

    return str(take_idx)


# ----------------------------------------------------------------------------
# Main processing
# ----------------------------------------------------------------------------
def process_sentences(index_entries: list[dict], classes: list[str],
                      yap: YAPpingConformer, device: torch.device,
                      out_root: Path, max_sentences: int | None = None,
                      interactive: bool = True, auto_accept: bool = False,
                      max_len: int = MAX_LEN, fps: float = FPS):
    """
    Run Yap+Conformer on each sentence, detect segments, accept/reject,
    save accepted sentences with entity vectors.

    Returns mapping dict: {sentence_slug: [take_idx, ...]}
    """
    class_to_idx = {c: i for i, c in enumerate(classes)}
    mapping: dict[str, list[str]] = {}
    stats = {"total": 0, "accepted": 0, "rejected": 0, "skipped": 0,
             "no_entity": 0, "saved": 0}

    # Deduplicate by sentence text (keep first occurrence per sentence)
    seen = set()
    unique_entries = []
    for entry in index_entries:
        s = entry.get("sentence", "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        unique_entries.append(entry)

    if max_sentences is not None:
        unique_entries = unique_entries[:max_sentences]

    print(f"[DATA] {len(unique_entries)} unique sentences to process")
    print(f"[MODE] {'interactive' if interactive else 'auto-accept' if auto_accept else 'headless'}")

    for i, entry in enumerate(unique_entries):
        sentence = entry.get("sentence", "").strip()
        tensor_path = entry.get("tensor_path", "")
        stats["total"] += 1

        print(f"\n[{i+1}/{len(unique_entries)}] {sentence}")
        print(f"  tensor: {tensor_path}")

        # Load sentence tensor
        try:
            x163 = torch.load(tensor_path, weights_only=True).float()
        except Exception as e:
            print(f"  [WARN] could not load {tensor_path}: {e}")
            stats["skipped"] += 1
            continue

        T = x163.shape[0]
        J = 53
        pose, conf, valid = split_163_to_pose(x163, J)
        pose = pose.unsqueeze(0)   # (1, T, J, 3)
        conf = conf.unsqueeze(0)   # (1, T, J)
        valid = valid.unsqueeze(0) # (1, T, J)
        t = torch.arange(T, dtype=torch.float32).unsqueeze(0) / fps  # (1, T)

        # Detect segments
        segments, out = detect_segments(yap, pose, conf, valid, t, device, max_len=max_len)

        # Encode segments to entity vectors
        if len(segments) == 0:
            stats["no_entity"] += 1
            print(f"  [NO ENTITY] no segments detected")
            if interactive:
                decision = prompt_accept(sentence, segments)
            elif auto_accept:
                decision = False
            else:
                decision = False
            if decision is True:
                # Save with empty entities
                entities = torch.zeros(0, yap.cfg.entity_dim)
                take_slug = save_sentence(sentence, stats["saved"] + 1, x163,
                                          entities, segments, out_root, fps)
                slug = slugify(sentence)
                mapping.setdefault(slug, []).append(take_slug)
                stats["saved"] += 1
                stats["accepted"] += 1
            elif decision == "skip":
                stats["skipped"] += 1
            else:
                stats["rejected"] += 1
            continue

        # Encode segments
        ent, _, _ = yap.yap.encode_segments(out, t, [segments])
        entities = ent.cpu()[0]  # (K, entity_dim)

        # Display segments
        display_segments(sentence, segments, t, fps=fps)

        # Accept/reject
        if interactive:
            decision = prompt_accept(sentence, segments)
        elif auto_accept:
            decision = True
        else:
            decision = False

        if decision is True:
            take_slug = save_sentence(sentence, stats["saved"] + 1, x163,
                                      entities, segments, out_root, fps)
            slug = slugify(sentence)
            mapping.setdefault(slug, []).append(take_slug)
            stats["saved"] += 1
            stats["accepted"] += 1
            print(f"  [ACCEPTED] saved as take {take_slug}")
        elif decision == "skip":
            stats["skipped"] += 1
            print(f"  [SKIPPED]")
        else:
            stats["rejected"] += 1
            print(f"  [REJECTED]")

    # Save mapping
    with open(out_root / "mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  BUILD COMPLETE")
    print(f"{'='*70}")
    print(f"  Total sentences:  {stats['total']}")
    print(f"  Accepted:         {stats['accepted']}")
    print(f"  Rejected:         {stats['rejected']}")
    print(f"  Skipped:          {stats['skipped']}")
    print(f"  No entity:        {stats['no_entity']}")
    print(f"  Saved takes:      {stats['saved']}")
    print(f"  Mapping saved to: {out_root / 'mapping.json'}")
    print(f"{'='*70}")

    return mapping


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Build per-sentence processed dataset with Yap+Conformer + accept/reject")
    p.add_argument("--index",       default=DEFAULT_INDEX,
                   help="Path to tensors_sentence_level/index.json")
    p.add_argument("--classes",     default=DEFAULT_CLASSES,
                   help="Path to classes.json")
    p.add_argument("--init",        default=DEFAULT_INIT,
                   help="Pretrained Yap checkpoint")
    p.add_argument("--out_root",    default=DEFAULT_OUT_ROOT,
                   help="Output root for sentence dataset")
    p.add_argument("--max_sentences", type=int, default=None,
                   help="Max number of unique sentences to process")
    p.add_argument("--max_len",     type=int,   default=MAX_LEN,
                   help="Max frames per sentence (truncate longer)")
    p.add_argument("--fps",         type=float, default=FPS,
                   help="Frames per second")
    p.add_argument("--d_model",     type=int,   default=128)
    p.add_argument("--n_mem_layers",type=int,   default=3)
    p.add_argument("--summary_channels", type=int, default=32)
    p.add_argument("--hippo_order", type=int,   default=16)
    p.add_argument("--entity_dim",  type=int,   default=256)
    p.add_argument("--heads",       type=int,   default=4)
    p.add_argument("--layers",      type=int,   default=4)
    p.add_argument("--interactive", action="store_true",
                   help="Prompt accept/reject for each sentence")
    p.add_argument("--auto-accept", action="store_true",
                   help="Auto-accept all detected segments (no prompts)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    with open(args.classes, encoding="utf-8") as f:
        classes = json.load(f)
    num_classes = len(classes)
    print(f"[CLASSES] {num_classes} classes")

    index = load_sentence_index(Path(args.index))
    print(f"[INDEX] {len(index)} sentence entries")

    # Build Yap+Conformer model
    kw = dict(d_model=args.d_model, n_mem_layers=args.n_mem_layers,
              summary_channels=args.summary_channels, hippo_order=args.hippo_order,
              entity_dim=args.entity_dim)
    ycfg = YAPConfig.corpus(**kw)
    J = ycfg.total_joints
    print(f"[LAYOUT] corpus: J = {J}")

    ccfg = ConformerConfig(num_words=num_classes, d_model=args.d_model,
                           heads=args.heads, layers=args.layers)
    yap = YAPpingConformer(ycfg, ccfg).to(device).eval()

    if args.init and os.path.exists(args.init):
        sd = torch.load(args.init, map_location=device, weights_only=True)
        m, u = load_flexible(yap, sd, label="SENTENCE_BUILDER")
        print(f"[INIT] loaded weights: missing {len(m)}, unexpected {len(u)}")
    else:
        print("[WARN] no --init given, using random weights")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    interactive = args.interactive and not args.auto_accept
    auto_accept = args.auto_accept and not args.interactive

    mapping = process_sentences(index, classes, yap, device, out_root,
                                max_sentences=args.max_sentences,
                                interactive=interactive,
                                auto_accept=auto_accept,
                                max_len=args.max_len, fps=args.fps)

    print(f"\n[DONE] Sentence dataset at {out_root}")


if __name__ == "__main__":
    main()
