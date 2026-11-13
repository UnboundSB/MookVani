"""
MookVani — Dummy 5-Word Sentence Formation Visualizer
=====================================================

Simulates how a 5-word ISL sentence is formed, word by word, with dummy
pose stick-figure frames. Each word gets its own short "sign clip" of
synthetic landmark frames, displayed sequentially while the word text
is shown. The sentence builds up cumulatively at the bottom.

Run locally:

    .\Backend\venv\Scripts\python.exe Backend/scripts/demo_5word_sentence.py

Press 'q' to quit early, any other key to advance to the next word.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Dummy pose generation (synthetic, not real MediaPipe data)
# ---------------------------------------------------------------------------

def make_word_clip(word: str, frames: int = 45, seed: int = 0) -> np.ndarray:
    """
    Generate synthetic (T, 163) dummy landmark data for one word.

    Layout (163 dims):
      [0:63]    left hand   (21 pts x 3)
      [63:126]  right hand  (21 pts x 3)
      [126:138] arms        (4 pts x 3)
      [138:159] face        (7 pts x 3)
      [159:163] flags       [lh, rh, arms, face]

    We only animate the hands and arms; face stays at origin. Returns
    float32 (T, 163).
    """
    rng = np.random.RandomState(seed)
    T = frames
    pose = np.zeros((T, 163), dtype=np.float32)

    # Base hand positions (image-normalized-ish, mapped later to pixels)
    lh_base = np.array([-1.2, 2.0, -0.5], dtype=np.float32)
    rh_base = np.array([+1.2, 2.0, -0.5], dtype=np.float32)

    # Word-specific "gesture" offset so each word looks slightly different
    gesture = {
        "I":      np.array([0.0,  0.0, 0.0]),
        "LOVE":   np.array([0.0, -0.8, 0.0]),   # hands toward chest
        "YOU":    np.array([0.6,  0.0, 0.0]),   # hands outward
        "VERY":   np.array([0.0,  0.6, 0.0]),   # hands up
        "MUCH":   np.array([-0.6, 0.0, 0.0]),   # hands wide
    }.get(word.upper(), np.array([0.0, 0.0, 0.0]))

    for t in range(T):
        prog = t / max(T - 1, 1)
        # ease-in-out
        ease = 0.5 - 0.5 * math.cos(math.pi * prog)

        # Left hand trajectory
        lh = lh_base + gesture + np.array([
            0.3 * math.sin(ease * math.pi),   # x sway
            -0.2 * ease,                      # y down then up
            0.0,
        ]) + rng.normal(0, 0.03, 3)

        # Right hand trajectory (mirror-ish)
        rh = rh_base + gesture + np.array([
            -0.3 * math.sin(ease * math.pi),
            -0.2 * ease,
            0.0,
        ]) + rng.normal(0, 0.03, 3)

        # Arms (shoulders + elbows) - relatively static
        arms = np.array([
            [-0.9, 3.2, -0.6],   # left shoulder
            [+0.9, 3.2, -0.6],   # right shoulder
            [-0.7, 2.4, -0.5],   # left elbow
            [+0.7, 2.4, -0.5],   # right elbow
        ], dtype=np.float32) + rng.normal(0, 0.02, (4, 3))

        # Face (7 points) - static center
        face = np.array([
            [-0.2, 4.2, -0.8],   # left eye
            [+0.2, 4.2, -0.8],   # right eye
            [0.0,  4.0, -0.9],   # nose tip
            [-0.4, 3.6, -0.7],   # left cheek
            [+0.4, 3.6, -0.7],   # right cheek
            [-0.2, 3.3, -0.6],   # left mouth
            [+0.2, 3.3, -0.6],   # right mouth
        ], dtype=np.float32) + rng.normal(0, 0.01, (7, 3))

        # Write into 163-dim layout
        pose[t, 0:63] = lh.reshape(21, 3).flatten() if False else np.tile(lh, 21)  # placeholder
        # Actually fill properly:
        # left hand: repeat same dummy point 21 times (for visualization only)
        pose[t, 0:63] = np.tile(lh, 21)
        pose[t, 63:126] = np.tile(rh, 21)
        pose[t, 126:138] = arms.flatten()
        pose[t, 138:159] = face.flatten()
        pose[t, 159:163] = [1.0, 1.0, 1.0, 0.0]  # lh, rh, arms present, face absent

    return pose


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def dims_to_px(coords: np.ndarray, w: int, h: int) -> list[tuple[int, int]]:
    """Map normalized coords (roughly [-2,+2]) to pixel space."""
    pts = []
    for i in range(0, len(coords), 3):
        x, y, _ = coords[i], coords[i + 1], coords[i + 2]
        px = int((x + 2.0) / 4.0 * w)
        py = int((y + 2.0) / 4.0 * h)
        px = max(0, min(w - 1, px))
        py = max(0, min(h - 1, py))
        pts.append((px, py))
    return pts


def draw_frame(
    w: int,
    h: int,
    pose: np.ndarray,
    current_word: str,
    sentence_so_far: str,
    word_idx: int,
    total_words: int,
    frame_idx: int,
    total_frames: int,
) -> np.ndarray:
    """Draw one dummy pose frame + HUD."""
    bg = np.full((h, w, 3), 20, dtype=np.uint8)
    overlay = np.zeros_like(bg)

    # --- draw left hand (green) ---
    lh_pts = dims_to_px(pose[0:63], w, h)
    for px, py in lh_pts:
        cv2.circle(overlay, (px, py), 4, (0, 255, 0), -1)

    # --- draw right hand (yellow) ---
    rh_pts = dims_to_px(pose[63:126], w, h)
    for px, py in rh_pts:
        cv2.circle(overlay, (px, py), 4, (0, 255, 255), -1)

    # --- draw arms (cyan) ---
    arm_pts = dims_to_px(pose[126:138], w, h)
    for i in range(0, len(arm_pts), 2):
        if i + 1 < len(arm_pts):
            cv2.line(overlay, arm_pts[i], arm_pts[i + 1], (255, 255, 0), 2)
    for px, py in arm_pts:
        cv2.circle(overlay, (px, py), 4, (255, 255, 0), -1)

    # --- draw face (magenta) ---
    face_pts = dims_to_px(pose[138:159], w, h)
    for px, py in face_pts:
        cv2.circle(overlay, (px, py), 4, (255, 0, 255), -1)

    # --- HUD panel ---
    cv2.rectangle(overlay, (0, 0), (w, 140), (15, 15, 15), -1)
    cv2.putText(
        overlay,
        f"WORD {word_idx + 1}/{total_words}:  {current_word}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 255, 0),
        3,
    )
    cv2.putText(
        overlay,
        f"SENTENCE SO FAR:  {sentence_so_far}",
        (20, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2,
    )
    cv2.putText(
        overlay,
        f"frame {frame_idx + 1}/{total_frames}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
    )
    cv2.putText(
        overlay,
        "q=quit   any key=next word",
        (w - 320, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (150, 150, 150),
        1,
    )

    # blend
    out = cv2.addWeighted(bg, 0.5, overlay, 0.9, 0)
    return out


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    words = ["I", "LOVE", "YOU", "VERY", "MUCH"]
    frames_per_word = 40
    w, h = 900, 600

    sentence_so_far = ""

    print("=" * 60)
    print(" MOOKVANI — DUMMY 5-WORD SENTENCE FORMATION")
    print("=" * 60)
    print(f"Sentence: {' '.join(words)}")
    print("Press 'q' to quit, any other key to advance to the next word.")
    print()

    cv2.namedWindow("MookVani — Dummy Sentence Formation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("MookVani — Dummy Sentence Formation", w, h)

    for wi, word in enumerate(words):
        clip = make_word_clip(word, frames=frames_per_word, seed=100 + wi)

        for fi in range(frames_per_word):
            # Show cumulative sentence including current word partially
            display_sentence = sentence_so_far + (" " + word if sentence_so_far else word)
            frame = draw_frame(
                w, h, clip[fi],
                current_word=word,
                sentence_so_far=display_sentence,
                word_idx=wi,
                total_words=len(words),
                frame_idx=fi,
                total_frames=frames_per_word,
            )
            cv2.imshow("MookVani — Dummy Sentence Formation", frame)
            key = cv2.waitKey(60) & 0xFF  # ~60ms per frame (~16 fps)
            if key == ord("q"):
                cv2.destroyAllWindows()
                return

        # Brief pause at end of word before next word
        end_frame = draw_frame(
            w, h, clip[-1],
            current_word=word,
            sentence_so_far=sentence_so_far + (" " + word if sentence_so_far else word),
            word_idx=wi,
            total_words=len(words),
            frame_idx=frames_per_word - 1,
            total_frames=frames_per_word,
        )
        cv2.imshow("MookVani — Dummy Sentence Formation", end_frame)
        key = cv2.waitKey(700) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            return

        sentence_so_far = sentence_so_far + (" " + word if sentence_so_far else word)
        print(f"[DONE] Word {wi + 1}: '{word}'  |  Sentence so far: '{sentence_so_far}'")

    # Final hold showing complete sentence
    final_clip = make_word_clip(words[-1], frames=frames_per_word, seed=200)
    for _ in range(30):
        frame = draw_frame(
            w, h, final_clip[-1],
            current_word="COMPLETE",
            sentence_so_far=sentence_so_far,
            word_idx=len(words),
            total_words=len(words),
            frame_idx=0,
            total_frames=1,
        )
        cv2.imshow("MookVani — Dummy Sentence Formation", frame)
        key = cv2.waitKey(40) & 0xFF
        if key == ord("q"):
            break

    print(f"\n[COMPLETE] Final sentence: '{sentence_so_far}'")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
