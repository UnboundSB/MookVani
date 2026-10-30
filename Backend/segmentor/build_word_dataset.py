"""
build_word_dataset.py
=====================

Per-word processed dataset for YAP Stage 1.5 (filteration + entity matching).

Structure under --out_root (segmentor/data/):

    <word_name>/
        1/
            word.pt       # raw (T, 163) tensor for this take
            processed.pt  # entity vector (entity_dim,) — YAP Stage 1 output
        2/
            word.pt
            processed.pt
        ...

Writes mapping.json: {"word": ["1", "2", ...], ...}

Phase 1 (instant): creates folders + copies word.pt files.
Phase 2 (GPU, single process): runs YAP Stage 1 front end on each clip
   and saves processed.pt. Uses 2 model instances on GPU + 2-6 CPU worker
   threads for data loading/preprocessing and writing results back.

Usage (from D:\MookVani\Backend):

    # Phase 1 only (instant):
    .\venv\Scripts\python.exe segmentor\build_word_dataset.py --phase 1

    # Phase 2 only (GPU, 2 model instances, 4 data-loading workers):
    .\venv\Scripts\python.exe segmentor\build_word_dataset.py `
        --phase 2 `
        --init segmentor/models/yap_stage1/yap_noise_robust.pth `
        --out_root segmentor/data `
        --layout corpus `
        --max_per_class 22 `
        --num_workers 4

    # Both phases:
    .\venv\Scripts\python.exe segmentor\build_word_dataset.py --phase 3 ...
"""
from __future__ import annotations

import json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F

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
DEFAULT_DATA_SRC = os.path.join(BACKEND_DIR, "data", "tensors_include50_word_level")
DEFAULT_CLASSES  = os.path.join(BACKEND_DIR, "models", "word_vocabulary", "classes.json")
DEFAULT_INIT     = os.path.join(BACKEND_DIR, "segmentor", "models", "yap_stage1", "yap_noise_robust.pth")
DEFAULT_OUT_ROOT = os.path.join(BACKEND_DIR, "segmentor", "data")


# ----------------------------------------------------------------------------
# Missing-coordinate fill  (prev + next coord interpolation)
# ----------------------------------------------------------------------------
def fill_missing_coords(pose: torch.Tensor, valid: torch.Tensor, max_gap_frames: int = 15):
    T, J, _ = pose.shape
    pose_f = pose.clone()
    for j in range(J):
        v = valid[:, j]
        idx_valid = torch.nonzero(v).flatten()
        if idx_valid.numel() == 0:
            continue
        first = int(idx_valid[0])
        last = int(idx_valid[-1])
        if first > 0:
            pose_f[:first, j] = pose[first, j]
        if last < T - 1:
            pose_f[last + 1:, j] = pose[last, j]
        for a, b in zip(idx_valid[:-1], idx_valid[1:]):
            a, b = int(a), int(b)
            gap = b - a - 1
            if gap == 0 or gap > max_gap_frames:
                continue
            t = torch.arange(1, gap + 1, dtype=pose.dtype, device=pose.device) / (gap + 1)
            pose_f[a + 1:b, j] = pose[a, j].unsqueeze(0) * (1 - t).unsqueeze(-1) + \
                                 pose[b, j].unsqueeze(0) * t.unsqueeze(-1)
    return pose_f


def split_163_to_pose(x163: torch.Tensor, J: int):
    T = x163.shape[0]
    coords = x163[:, :J * 3].contiguous().view(T, J, 3)
    conf = torch.ones(T, J)
    valid = torch.ones(T, J, dtype=torch.bool)
    return coords, conf, valid


# ----------------------------------------------------------------------------
# Worker thread: CPU-bound data loading + preprocessing
# ----------------------------------------------------------------------------
def prepare_clip(work_item, J, max_len, max_gap_frames, fps):
    """
    Load a clip from disk, build validity mask, fill missing coords.
    Returns (word, take_idx, word_path, proc_path, pose_tensor, t_tensor) or None if skipped.
    Runs in a worker thread (CPU-bound, releases GIL during torch ops).
    """
    word, take_idx, word_path, proc_path = work_item

    # Resume: skip if already done
    if Path(proc_path).exists():
        return ("skip", word, take_idx, None, None)

    # Load raw clip
    x163 = torch.load(word_path, weights_only=True).float()
    T = min(x163.shape[0], max_len)
    clip = x163[:T]

    # Build per-joint valid from modality flags [159:163]
    flags = clip[:, 159:163]                                          # (T,4)
    valid_joint = torch.zeros(T, J, dtype=torch.bool)
    valid_joint[:, 0:21]  = flags[:, 0:1] > 0.5                       # left hand
    valid_joint[:, 21:42] = flags[:, 1:2] > 0.5                       # right hand
    valid_joint[:, 42:46] = flags[:, 2:3] > 0.5                       # arms
    valid_joint[:, 46:53] = flags[:, 3:4] > 0.5                       # face

    # Fill missing coords by prev/next interpolation
    pose_np, _, _ = split_163_to_pose(clip, J)
    pose_filled = fill_missing_coords(pose_np, valid_joint,
                                      max_gap_frames=max_gap_frames)
    pose = pose_filled.unsqueeze(0)                                   # (1,T,J,3)
    t = torch.arange(T, dtype=torch.float32).unsqueeze(0) / fps       # (1,T)

    return ("ready", word, take_idx, word_path, proc_path, pose, t)


def write_result(proc_path: str, processed: torch.Tensor):
    """Write entity vector to disk (worker thread, CPU-bound)."""
    torch.save(processed, proc_path)


# ----------------------------------------------------------------------------
# Phase 2: run YAP front end on each clip (GPU, single process, 2 model instances)
# ============================================================================
def phase2(args, clips_by_class, classes, out_root, mapping):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    print(f"[MODELS] loading 2 model instances on GPU")

    # Build work list
    work_items = []
    already_done = 0
    for cls_idx in range(len(classes)):
        word = classes[cls_idx]
        items = clips_by_class.get(cls_idx, [])
        if not items:
            continue
        for take_idx, (clip_name, x163) in enumerate(items, start=1):
            take_dir = out_root / word / str(take_idx)
            word_path = take_dir / "word.pt"
            proc_path = take_dir / "processed.pt"
            if word_path.exists() and proc_path.exists():
                already_done += 1
                continue
            if not word_path.exists():
                take_dir.mkdir(parents=True, exist_ok=True)
                torch.save(x163, word_path)
            work_items.append((word, take_idx, str(word_path), str(proc_path)))

    total_todo = len(work_items)
    print(f"[PHASE 2] {total_todo} clips to process "
          f"({already_done} already done), device={device}")

    # ---- Build TWO model instances on GPU ----
    kw = dict(d_model=args.d_model, n_mem_layers=args.n_mem_layers,
              summary_channels=args.summary_channels, hippo_order=args.hippo_order,
              entity_dim=args.entity_dim)
    if args.layout == "corpus":
        ycfg = YAPConfig.corpus(**kw)
    elif args.layout == "holistic":
        ycfg = YAPConfig.holistic(**kw)
    else:
        ycfg = YAPConfig(**kw)
    J = ycfg.total_joints
    print(f"[LAYOUT] {args.layout}: J = {J}")

    ccfg = ConformerConfig(num_words=len(classes), d_model=args.d_model,
                           heads=args.heads, layers=args.layers)

    model_0 = YAPpingConformer(ycfg, ccfg).to(device).eval()
    model_1 = YAPpingConformer(ycfg, ccfg).to(device).eval()

    if args.init and os.path.exists(args.init):
        sd = torch.load(args.init, map_location=device, weights_only=True)
        m0, u0 = load_flexible(model_0, sd, label="MODEL_0")
        m1, u1 = load_flexible(model_1, sd, label="MODEL_1")
        print(f"[INIT] model_0: missing {len(m0)}, unexpected {len(u0)} | "
              f"model_1: missing {len(m1)}, unexpected {len(u1)}")
    else:
        print("[WARN] no --init given, using random weights")

    models = [model_0, model_1]
    model_idx = 0  # alternate between the two

    stats = {"processed": 0, "skipped": already_done, "no_entity": 0, "entities_total": 0}
    t0 = time.time()

    # ThreadPoolExecutor for data loading + writing back (CPU cores)
    num_workers = max(1, min(args.num_workers, 6))
    print(f"[WORKERS] {num_workers} CPU threads for data loading/writing")

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        # Pre-submit the first batch of clips for preparation
        prep_futures = {}
        prep_batch = min(num_workers * 2, total_todo)
        for i in range(prep_batch):
            fut = pool.submit(prepare_clip, work_items[i], J, args.max_len,
                              args.max_gap_frames, args.fps)
            prep_futures[fut] = i

        completed = 0
        next_to_prep = prep_batch

        while prep_futures:
            # Wait for the next prepared clip to finish
            done_fut = None
            for fut in list(prep_futures.keys()):
                if fut.done():
                    done_fut = fut
                    break
            if done_fut is None:
                # None ready yet, wait a bit
                import threading
                threading.Event().wait(0.001)
                continue

            idx = prep_futures.pop(done_fut)
            result = done_fut.result()
            status = result[0]
            if status == "skip":
                stats["skipped"] += 1
                completed += 1
                # Submit a new prep task to replace this one
                if next_to_prep < total_todo:
                    new_fut = pool.submit(prepare_clip, work_items[next_to_prep], J,
                                          args.max_len, args.max_gap_frames, args.fps)
                    prep_futures[new_fut] = next_to_prep
                    next_to_prep += 1
                continue

            # "ready" path: unpack the remaining 6 values
            _, word, take_idx, wp, pp, pose, t = result

            # ---- Run inference on GPU (main thread, alternates models) ----
            model = models[model_idx]
            model_idx = (model_idx + 1) % 2

            conf = torch.ones(pose.shape[1], J, device=device).unsqueeze(0)
            valid = torch.ones(pose.shape[1], J, dtype=torch.bool, device=device).unsqueeze(0)
            pose = pose.to(device)
            t = t.to(device)

            out = model.yap.forward_frames(pose, conf, valid, t)
            segments = model.yap.segment_batch(out, t)[0]

            if len(segments) == 0:
                stats["no_entity"] += 1
                fake_seg = Segment(0, pose.shape[1], float(t[0, 0]),
                                   float(t[0, pose.shape[1] - 1]), "fallback", 0.0)
                ent, _, _ = model.yap.encode_segments(out, t, [[fake_seg]])
                processed = ent.cpu()[0]
            else:
                ent, _, _ = model.yap.encode_segments(out, t, [segments])
                processed = ent.cpu()[0]

            # ---- Write result back via worker thread ----
            pool.submit(write_result, pp, processed)
            stats["processed"] += 1
            stats["entities_total"] += 1
            completed += 1

            # Submit a new prep task to keep the pipeline full
            if next_to_prep < total_todo:
                new_fut = pool.submit(prepare_clip, work_items[next_to_prep], J,
                                      args.max_len, args.max_gap_frames, args.fps)
                prep_futures[new_fut] = next_to_prep
                next_to_prep += 1

            if completed % 10 == 0 or completed == total_todo:
                elapsed = time.time() - t0
                rate = completed / max(elapsed, 1e-6)
                remaining = (total_todo - completed) / max(rate, 1e-6)
                print(f"  [{completed}/{total_todo}] {rate:.2f} clip/s, ~{remaining:.0f}s remaining",
                      flush=True)

    elapsed = time.time() - t0
    print(f"\n[PHASE 2] {stats['processed']} new clips processed "
          f"(skipped {stats['skipped']}, no-entity {stats['no_entity']})")
    print(f"[PHASE 2] total entities: {stats['entities_total']}, time: {elapsed:.0f}s")


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_clips(data_src: str, classes: list[str], max_per_class: int = 22):
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
        for _, x163 in items:
            take_num += 1
            td = out_root / word / str(take_num)
            td.mkdir(parents=True, exist_ok=True)
            torch.save(x163, td / "word.pt")
            mapping[word].append(str(take_num))
            total += 1
    with open(out_root / "mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    print(f"[PHASE 1] {total} word.pt across {len(mapping)} words")
    print(f"[PHASE 1] mapping saved to {out_root / 'mapping.json'}")
    return mapping


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
        mapping_path = out_root / "mapping.json"
        if mapping_path.exists():
            with open(mapping_path, encoding="utf-8") as f:
                mapping = json.load(f)
        else:
            raise RuntimeError("mapping.json not found. Run --phase 1 first.")

    # ---- Phase 2: run YAP on each clip (GPU, 2 models, CPU workers for I/O) ----
    if args.phase in (2, 3):
        phase2(args, clips_by_class, classes, out_root, mapping)

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
    p.add_argument("--max_per_class",  type=int,   default=22,
                   help="max clips per word class (was 15, now 22)")
    p.add_argument("--phase",          type=int,   default=3,
                   choices=[1, 2, 3],
                   help="1=create folders+word.pt only (instant), "
                        "2=run YAP+processed.pt (GPU, 2 models, CPU I/O workers), "
                        "3=both")
    p.add_argument("--num_workers",    type=int,   default=4,
                   help="CPU worker threads for data loading/writing (default 4, range 2-6)")
    p.add_argument("--max_gap_frames", type=int, default=15,
                   help="max gap frames to interpolate across; longer gaps held")
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
