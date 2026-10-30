"""
Generate synthetic noise/garbage clips for word-presence training.

Two types of noise clips (50/50 split):
  1. ALL-BLANK: completely still pose (zeros everywhere, flags=0)
     -> simulates no one in frame / camera covered
  2. RANDOM GARBAGE: random hand jitter, still body/face, flags=0
     -> simulates tracker noise / scratching / non-sign movement

Each clip is (T, 163) float32 — same layout as the word corpus tensors.
Saved as .npy so train_noise_filter.py can load them via index.json.

Usage:
    python D:\MookVani\backend\segmentor\data_builder\generate_noise_clips.py
"""
import json
from pathlib import Path

import numpy as np

OUT_DIR = Path(r"D:\MookVani\backend\segmentor\data_builder\noise_clips")
NOISE_COUNT = 500          # total noise clips (250 blank + 250 garbage)
MIN_LEN, MAX_LEN = 20, 120 # frames per clip
SEED = 42

# 163-dim layout offsets
HAND_DIM = 63
BODY_START = 126
BODY_DIM = 12
FACE_START = 138
FACE_DIM = 21
FLAGS_START = 159
FLAGS_DIM = 4
TOTAL_DIM = 163

rng_np = np.random.default_rng(SEED)


def make_blank_clip(t: int) -> np.ndarray:
    """Completely still clip — all zeros, presence flags = 0."""
    return np.zeros((t, TOTAL_DIM), dtype=np.float32)


def make_garbage_clip(t: int) -> np.ndarray:
    """Random hand jitter, still body/face — NOT a sign."""
    clip = np.zeros((t, TOTAL_DIM), dtype=np.float32)

    # Hands: random walk jitter (simulates tracker noise / non-sign movement)
    for h in range(2):
        base = h * HAND_DIM
        hand = np.zeros((t, 21, 3), dtype=np.float32)
        hand[:, 0, :] = rng_np.uniform(-0.3, 0.3, size=(t, 3))  # wrist
        for j in range(1, 21):
            hand[:, j, :] = hand[:, 0, :] + rng_np.normal(0, 0.05, size=(t, 3))
        clip[:, base:base + 63] = hand.reshape(t, 63)

    # Body: still
    clip[:, BODY_START:BODY_START + BODY_DIM] = rng_np.normal(
        0, 0.01, size=(t, 4, 3)).astype(np.float32).reshape(t, 12)

    # Face: still
    clip[:, FACE_START:FACE_START + FACE_DIM] = rng_np.normal(
        0, 0.01, size=(t, 7, 3)).astype(np.float32).reshape(t, 21)

    # Presence flags: all 0
    clip[:, FLAGS_START:FLAGS_START + FLAGS_DIM] = 0.0

    return clip


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    half = NOISE_COUNT // 2
    print(f"[INFO] Generating {half} all-blank + {NOISE_COUNT - half} random-garbage "
          f"noise clips -> {OUT_DIR}")

    entries = []
    idx = 0
    for i in range(half):
        t = int(rng_np.integers(MIN_LEN, MAX_LEN + 1))
        clip = make_blank_clip(t)
        fname = f"blank_{i:04d}.npy"
        fpath = OUT_DIR / fname
        np.save(fpath, clip)
        entries.append({"path": str(fpath), "label": "NOISE", "frames": t, "type": "blank"})
        idx += 1

    for i in range(NOISE_COUNT - half):
        t = int(rng_np.integers(MIN_LEN, MAX_LEN + 1))
        clip = make_garbage_clip(t)
        fname = f"garbage_{i:04d}.npy"
        fpath = OUT_DIR / fname
        np.save(fpath, clip)
        entries.append({"path": str(fpath), "label": "NOISE", "frames": t, "type": "garbage"})
        idx += 1

    index_path = OUT_DIR / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)

    print(f"[INFO] Wrote {len(entries)} noise clips + index.json")
    print("[INFO] Label 'NOISE' is NOT in classes.json - existing loader skips them")
    print(f"[INFO] Training mix: word clips (presence=1 during sign) + noise clips (presence=0 everywhere)")


if __name__ == "__main__":
    main()
