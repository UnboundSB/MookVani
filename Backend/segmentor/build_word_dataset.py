"""
build_word_dataset.py
=====================

Two-phase dataset builder for per-word processed YAP outputs.

Structure under --out_root (default: segmentor/data/):

    <word_name>/
        1/
            word.pt       # raw (T, 163) tensor for this take
            processed.pt  # YAP Stage 1 output: (n_entities, entity_dim) or cleaned (T, J, 3)
        2/
            word.pt
            processed.pt
        ...

Writes --out_root/mapping.json:
    {"good afternoon": ["1","2"], "alright": ["1","2","3"], ...}

Phase 1 (instant): creates all folders and copies word.pt files.
Phase 2 (slow, ~4s/clip on CUDA): runs YAP Stage 1 front end on each clip
       and saves processed.pt. Resumable — skips takes that already have
       both files.

Usage (from D:\MookVani\Backend):

    # Phase 1 only (instant, creates the folder structure):
    .\venv\Scripts\python.exe segmentor\build_word_dataset.py --phase 1

    # Phase 1 + Phase 2 (runs YAP on every clip, takes hours):
    .\venv\Scripts\python.exe segmentor\build_word_dataset.py `
        --init segmentor/models/yap_stage1/yap_noise_robust.pth `
        --out_root segmentor/data `
        --layout corpus `
        --max_per_class 15

    # Resume Phase 2 after interruption:
    .\venv\Scripts\python.exe segmentor\build_word_dataset.py `
        --init segmentor/models/yap_stage1/yap_noise_robust.pth `
        --out_root segmentor/data `
        --layout corpus `
        --max_per_class 15
"""

from __future__ import annotations

import json, os, random, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.checkpoint_utils import load_flexible

from yap_conformer import (
    YAPConfig, YAPpingConformer, ConformerConfig,
)

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_SRC = os.path.join(BACKEND_DIR, "data", "tensors_include50_word_level")
DEFAULT_CLASSES  = os.path.join(BACKEND_DIR, "models", "word_vocabulary", "classes.json")
DEFAULT_INIT     = os.path.join(BACKEND_DIR, "segmentor", "models", "yap_stage1", "yap_noise_robust.pth")
DEFAULT_OUT_ROOT = os.path.join(BACKEND_DIR, "segmentor", "data")


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_clips(data_src: str, classes: list[str], max_per_class: int = 15):
    """Load .pt clips grouped by class index.

    Returns dict[class_idx] -> list[(clip_name, tensor (T, 163))].
    """
    label_map = {c: i for i, c in enumerate(classes)}
    clips_by_class = {}
    d = Path(data_src)
    for folder in d.iterdir():
        if folder.name not in label_map:
            continue
        idx = label_map[folder.name]
        pts = sorted(folder.rglob("*.pt"))
        random.shuffle(pts)
        items = []
        for p in pts[:max_per_class]:
            t = torch.load(p, weights_only=True).float()
            items.append((p.name, t))
        clips_by_class[idx] = items
    return clips_by_class


def split_163_to_pose(x163: torch.Tensor, J: int):
    """Split (T, 163) -> pose (T, J, 3), conf (T, J), valid (T, J)."""
    T = x163.shape[0]
    coords = x163[:, :J * 3].contiguous().view(T, J, 3)
    conf = torch.ones(T, J)
    valid = torch.ones(T, J, dtype=torch.bool)
    return coords, conf, valid


# ----------------------------------------------------------------------------
# Phase 1: create folders + save word.pt (instant)
# ----------------------------------------------------------------------------
def phase1(args, clips_by_class, classes, out_root):
    mapping = {}
    total = 0
    for cls_idx in range(len(classes)):
        word = classes[cls_idx]
        items = clips_by_class.get(cls_idx, [])
        if not items:
            continue
        mapping[word] = []
        take_num = 0
        for clip_name, x163 in items:
            take_num += 1
            take_dir = out_root / word / str(take_num)
            take_dir.mkdir(parents=True, exist_ok=True)
            torch.save(x163, take_dir / "word.pt")
            mapping[word].append(str(take_num))
            total += 1
    mapping_path = out_root / "mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    print(f"[PHASE 1] {total} word.pt files created across {len(mapping)} words")
    print(f"[PHASE 1] mapping saved to {mapping_path}")
    return mapping


# ----------------------------------------------------------------------------
# Phase 2: run YAP Stage 1 front end on each clip (slow, resumable)
# ----------------------------------------------------------------------------
def phase2(args, clips_by_class, classes, out_root, ycfg, model, device, mapping):
    J = ycfg.total_joints
    stats = {"processed": 0, "skipped": 0, "no_entity": 0, "entities_total": 0}
    t0 = time.time()

    with torch.no_grad():
        for cls_idx in range(len(classes)):
            word = classes[cls_idx]
            items = clips_by_class.get(cls_idx, [])
            if not items:
                continue
            for take_idx, (clip_name, x163) in enumerate(items, start=1):
                take_dir = out_root / word / str(take_idx)
                word_path = take_dir / "word.pt"
                proc_path = take_dir / "processed.pt"

                # Resume: skip if both files exist
                if word_path.exists() and proc_path.exists():
                    stats["skipped"] += 1
                    continue

                # Make sure word.pt exists (Phase 1 may not have run)
                if not word_path.exists():
                    take_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(x163, word_path)

                # Run through YAP Stage 1 front end
                T = min(x163.shape[0], args.max_len)
                clip = x163[:T]
                pose, conf, valid = split_163_to_pose(clip, J)
                pose = pose.unsqueeze(0).to(device)
                conf = conf.unsqueeze(0).to(device)
                valid = valid.unsqueeze(0).to(device)
                t = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(0) / args.fps

                out = model.yap.forward_frames(pose, conf, valid, t)
                segments = model.yap.segment_batch(out, t)[0]

                if len(segments) == 0:
                    stats["no_entity"] += 1
                    # Save cleaned poses instead of empty entity
                    clean = out["clean"][0].cpu()  # (T, J, 3)
                    torch.save(clean, proc_path)
                else:
                    ent, meta, _ = model.yap.encode_segments(out, t, [segments])
                    processed = ent.cpu()  # (n_entities, entity_dim)
                    torch.save(processed, proc_path)
                    stats["entities_total"] += len(segments)

                stats["processed"] += 1

                # Progress every clip (with timing)
                elapsed = time.time() - t0
                rate = stats["processed"] / max(elapsed, 1e-6)
                remaining = (len(items) - take_idx) / max(rate, 1e-6)
                print(f"  [{stats['processed']+stats['skipped']}] {word}/{take_idx} "
                      f"({rate:.2f} clip/s, ~{remaining:.0f}s remaining)",
                      end="\r", flush=True)

    elapsed = time.time() - t0
    print(f"\n[PHASE 2] {stats['processed']} new clips processed "
          f"(skipped {stats['skipped']}, no-entity {stats['no_entity']})")
    print(f"[PHASE 2] total entities: {stats['entities_total']}, time: {elapsed:.0f}s")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def build_dataset(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    with open(args.classes, encoding="utf-8") as f:
        classes = json.load(f)
    num_classes = len(classes)

    clips_by_class = load_clips(args.data_src, classes, args.max_per_class)
    n_classes_with_clips = sum(1 for v in clips_by_class.values() if v)
    print(f"[DATA] {n_classes_with_clips} classes with clips / {num_classes} total")
    if n_classes_with_clips == 0:
        raise RuntimeError("No word clips found in data source")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- Phase 1: create folders + save word.pt ----
    if args.phase in (1, 3):
        mapping = phase1(args, clips_by_class, classes, out_root)
    else:
        # Load existing mapping
        mapping_path = out_root / "mapping.json"
        if mapping_path.exists():
            with open(mapping_path, encoding="utf-8") as f:
                mapping = json.load(f)
        else:
            raise RuntimeError("mapping.json not found. Run --phase 1 first.")

    # ---- Phase 2: run YAP on each clip ----
    if args.phase in (2, 3):
        cfg_kwargs = dict(
            d_model=args.d_model, n_mem_layers=args.n_mem_layers,
            summary_channels=args.summary_channels, hippo_order=args.hippo_order,
            entity_dim=args.entity_dim,
        )
        if args.layout == "corpus":
            ycfg = YAPConfig.corpus(**cfg_kwargs)
        elif args.layout == "holistic":
            ycfg = YAPConfig.holistic(**cfg_kwargs)
        else:
            ycfg = YAPConfig(**cfg_kwargs)
        J = ycfg.total_joints
        print(f"[LAYOUT] {args.layout}: J = {J}")

        ccfg = ConformerConfig(num_words=num_classes, d_model=args.d_model,
                               heads=args.heads, layers=args.layers)
        model = YAPpingConformer(ycfg, ccfg).to(device)
        model.eval()

        if args.init and os.path.exists(args.init):
            sd = torch.load(args.init, map_location=device, weights_only=True)
            missing, unexpected = load_flexible(model, sd, label="STAGE1")
            print(f"[INIT] {args.init} | missing {len(missing)}, unexpected {len(unexpected)}")
        else:
            print("[WARN] no --init given, using random weights")

        phase2(args, clips_by_class, classes, out_root, ycfg, model, device, mapping)

        # Re-save mapping (may have new entries from Phase 2 creating dirs)
        mapping_path = out_root / "mapping.json"
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)

    total_files = sum(len(items) for items in clips_by_class.values())
    print(f"\n[DONE] Dataset at {out_root}")
    print(f"  total clips : {total_files}")
    print(f"  words       : {len(mapping)} / {num_classes}")


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Build per-word processed dataset from Stage 1 YAP")
    p.add_argument("--data_src",       default=DEFAULT_DATA_SRC)
    p.add_argument("--classes",        default=DEFAULT_CLASSES)
    p.add_argument("--init",           default=DEFAULT_INIT)
    p.add_argument("--out_root",       default=DEFAULT_OUT_ROOT)
    p.add_argument("--layout",         type=str,   default="corpus",
                   choices=["corpus", "holistic", "hands"])
    p.add_argument("--fps",            type=float, default=30.0)
    p.add_argument("--max_len",        type=int,   default=900)
    p.add_argument("--max_per_class",  type=int,   default=15)
    p.add_argument("--phase",          type=int,   default=3,
                   choices=[1, 2, 3],
                   help="1=create folders+word.pt only (instant), "
                        "2=run YAP+processed.pt (slow, resumable), "
                        "3=both")
    p.add_argument("--d_model",        type=int, default=128)
    p.add_argument("--n_mem_layers",   type=int, default=3)
    p.add_argument("--summary_channels", type=int, default=32)
    p.add_argument("--hippo_order",    type=int, default=16)
    p.add_argument("--entity_dim",     type=int, default=256)
    p.add_argument("--heads",          type=int, default=4)
    p.add_argument("--layers",         type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
