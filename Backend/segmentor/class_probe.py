"""
class_probe.py — One-class word probe for YAP segmentor checkpoints.

Pick one target class (index into classes.json, 0..227), load ANY video from
that class, run it through MediaPipe once and cache the (T,163) tensor, then
forward it through a trained YAPpingConformer.

Output:
    1  if predicted_word_class == target_class  AND  target_class is not blank
    0  otherwise (wrong class, no segments, or target is blank)

Usage (from D:\MookVani\Backend):
    python segmentor\class_probe.py --target-class 5
    python segmentor\class_probe.py --target-class 5 --video Backend\data\tensors_include50_word_level\actor\MVI_4516.pt --checkpoint segmentor\models\yap_stage1\yap_noise_robust.pth
    python segmentor\class_probe.py --target-class 0 --blank          # blank always -> 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

BACKEND_DIR = Path(__file__).resolve().parent.parent
SEGMENTOR_DIR = Path(__file__).resolve().parent
for _p in (str(BACKEND_DIR), str(SEGMENTOR_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from yap_conformer import (  # noqa: E402
    YAPConfig, YAPpingConformer, ConformerConfig, hand_part,
)

# word_logits class 0 = CTC blank; classes.json[i] <-> word_logits index i+1
BLANK_WL_IDX = 0
WORD_CORPORA = [
    BACKEND_DIR / "data" / "tensors_include50_word_level",
    BACKEND_DIR / "data" / "tensors_isl_csltr_word_level",
]
CLASSES_PATH = BACKEND_DIR / "models" / "word_vocabulary" / "classes.json"
CACHE_DIR = SEGMENTOR_DIR / "features_cache" / "class_probe"


def load_classes() -> list:
    with open(CLASSES_PATH, encoding="utf-8") as f:
        return json.load(f)


def infer_architecture(state):
    """Detect d_model and joint count J from a checkpoint state dict."""
    d_model, J = 128, 42
    for k in ("yap.core.lift.0.weight", "yap.core.layers.0.inp.weight"):
        if k in state and isinstance(state[k], torch.Tensor):
            d_model = int(state[k].shape[0]); break
    for k in ("yap.core.cleaner.parent_idx", "yap.core.cleaner.part_of_joint",
              "yap.core.cleaner.project_mask"):
        if k in state and isinstance(state[k], torch.Tensor):
            J = int(state[k].shape[0]); break
    layout = "corpus" if J >= 50 else "hands"
    return d_model, J, layout


def build_model(device, layout, d_model):
    if layout == "corpus":
        ycfg = YAPConfig.corpus(d_model=d_model, n_mem_layers=3, summary_channels=32,
                                hippo_order=16, entity_dim=256)
    else:
        ycfg = YAPConfig(parts=[hand_part("left_hand"), hand_part("right_hand")],
                         d_model=d_model, n_mem_layers=3, summary_channels=32,
                         hippo_order=16, entity_dim=256)
    J = ycfg.total_joints
    ccfg = ConformerConfig(num_words=228, d_model=d_model, heads=4, layers=4)
    model = YAPpingConformer(ycfg, ccfg).to(device)
    model.eval()
    return model, J


def find_video_for_class(target_class: int, classes: list):
    """Return (Path, class_name) for the first .pt video under the target class."""
    class_name = classes[target_class]
    for root in WORD_CORPORA:
        class_dir = root / class_name
        if class_dir.is_dir():
            pts = sorted(class_dir.rglob("*.pt"))
            if pts:
                return pts[0], class_name
    return None, class_name


def load_or_cache(video_path: Path, cache_dir: Path) -> torch.Tensor:
    """Load a (T,163) tensor, store a processed copy in cache, return tensor."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{video_path.stem}.pt"
    if cache_path.exists():
        return torch.load(cache_path, weights_only=True).float()
    x = torch.load(video_path, weights_only=True).float()
    torch.save(x, cache_path)
    return x


def forward_word_logits(model, J, x163, device, layout):
    """Forward a (T,163) tensor through YAPpingConformer; return word_logits (K, num_words+1)."""
    T = x163.shape[0]
    pose = x163[:, :J * 3].contiguous().view(T, J, 3).unsqueeze(0).to(device)
    conf = torch.ones(1, T, J, device=device)
    valid = torch.ones(1, T, J, dtype=torch.bool, device=device)
    t = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(0) / 25.0

    with torch.no_grad():
        out = model(pose=pose, conf=conf, valid=valid, t=t, segments=None, reset_mask=None)

    wl = out.get("word_logits", None)
    if wl is None:
        return None
    return wl[0]  # (K, num_words+1)


def main():
    parser = argparse.ArgumentParser(description="One-class word probe for YAP segmentor")
    parser.add_argument("--target-class", type=int, required=True,
                        help="Target class index into classes.json (0..227)")
    parser.add_argument("--blank", action="store_true",
                        help="Treat target as CTC blank (always -> 0)")
    parser.add_argument("--video", default=None, help="Specific .pt video path (auto-picked if omitted)")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint .pth (auto-discovered if omitted)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = load_classes()
    num_classes = len(classes)
    print(f"[DEVICE] {device} | {num_classes} classes")

    if not (0 <= args.target_class < num_classes):
        print(f"[ERROR] target-class {args.target_class} out of range [0,{num_classes-1}]")
        return 1

    # ---- checkpoint ----
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        for c in [
            SEGMENTOR_DIR / "models" / "yap_stage1" / "yap_noise_robust.pth",
            SEGMENTOR_DIR / "models" / "yap_stage1_presence" / "yap_noise_robust.pth",
            BACKEND_DIR / "models" / "yap_segmentor" / "yap_segmentor_final.pth",
        ]:
            if c.is_file():
                ckpt_path = c; break
    if ckpt_path is None or not Path(ckpt_path).is_file():
        print("[ERROR] no checkpoint found")
        return 1
    print(f"[CKPT] {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict):
        for k in ("model_state_dict", "state_dict", "model"):
            if k in state and isinstance(state[k], dict):
                state = state[k]; break

    d_model, J, layout = infer_architecture(state)
    print(f"[ARCH] d_model={d_model}, J={J}, layout={layout}")

    model, Jm = build_model(device, layout, d_model)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[LOAD] missing={len(missing)}, unexpected={len(unexpected)}")

    # ---- video ----
    if args.video:
        video_path = Path(args.video)
        class_name = classes[args.target_class]
    else:
        video_path, class_name = find_video_for_class(args.target_class, classes)
        if video_path is None:
            print(f"[ERROR] no video for class '{classes[args.target_class]}' (idx {args.target_class})")
            return 1
        print(f"[VIDEO] {video_path}  (class={class_name})")

    x = load_or_cache(video_path, CACHE_DIR)
    print(f"[DATA] shape={tuple(x.shape)} frames={x.shape[0]}")

    # ---- forward ----
    wl = forward_word_logits(model, Jm, x, device, layout)
    if wl is None:
        print("[WARN] model produced no word_logits")
        return 1

    # wl: (K, num_words+1). Aggregate over entities: mean then argmax.
    mean_logits = wl.mean(dim=0)
    probs = torch.softmax(mean_logits, dim=0)
    pred_wl_idx = int(torch.argmax(probs).item())
    pred_conf = float(probs[pred_wl_idx].item())

    # Map word_logits index -> classes.json index (wl 0 = blank, wl i+1 = classes[i])
    if pred_wl_idx == 0:
        pred_class = None  # blank
        pred_class_name = "<blank>"
    else:
        pred_class = pred_wl_idx - 1
        pred_class_name = classes[pred_class] if pred_class < num_classes else f"#{pred_class}"

    target = args.target_class
    is_blank = args.blank or (target == BLANK_WL_IDX and classes[target].lower() in ("blank", ""))

    if is_blank:
        result = 0
        reason = "target is blank"
    elif pred_wl_idx == target + 1:
        result = 1
        reason = f"predicted {pred_class_name} == target {classes[target]}"
    else:
        result = 0
        reason = f"predicted {pred_class_name} != target {classes[target]}"

    print(f"\n{'='*52}")
    print(f"TARGET CLASS : {target} ({classes[target]})  blank={is_blank}")
    print(f"PREDICTED    : {pred_class_name}  (wl_idx={pred_wl_idx}, conf={pred_conf:.3f})")
    print(f"RESULT       : {result}  ({reason})")
    print(f"{'='*52}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
