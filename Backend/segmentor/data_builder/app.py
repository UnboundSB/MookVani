"""
Minimal Streamlit segmentation UI.

Flow per sentence (in dataset order):
  1. Word videos play large, one after another (stacked, autoplay).
  2. User presses "r - repeat words" or "space - begin sentence".
  3. Composed sentence video plays in slow motion.
  4. User presses "SPACE - mark segmentation" to record frame/timestamp.
  5. User presses "r - redo sentence" or "space - continue to next sentence".

Output appended to segmentation.csv.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List

import cv2
import streamlit as st

from builder import (
    COMPOSED_VIDEOS_DIR,
    COMPOSED_FPS,
    SLOWDOWN_FACTOR,
    SEGMENTATION_CSV,
    SentenceEntry,
    SegmentationDB,
    compose_sentence_video,
    hash_sentence,
    load_sentences_in_order,
    match_words_to_videos,
    scan_word_videos,
    load_checkpoint,
    save_checkpoint,
    clear_checkpoint,
)

st.set_page_config(page_title="Sentence Segmentor", page_icon="🎬", layout="wide")
st.title("🎬 Sentence Word Segmentor")
st.caption("Dataset: F:\\dataset  |  Output: segmentation.csv")

# ---------------- session state ----------------
def _init():
    if "db" not in st.session_state:
        st.session_state.db = SegmentationDB()
    if "word_index" not in st.session_state:
        st.session_state.word_index = scan_word_videos()
    if "sentences" not in st.session_state:
        st.session_state.sentences = load_sentences_in_order()

    # Resume from checkpoint if it exists
    ckpt = load_checkpoint()
    default_idx = 0
    default_stage = "words"
    if ckpt:
        try:
            default_idx = int(ckpt.get("sentence_index", 0))
            default_stage = ckpt.get("stage", "words")
        except Exception:
            default_idx = 0
            default_stage = "words"
        # Clamp index to valid range
        if default_idx < 0 or default_idx >= len(st.session_state.sentences):
            default_idx = 0

    if "idx" not in st.session_state:
        st.session_state.idx = default_idx
    if "stage" not in st.session_state:
        st.session_state.stage = default_stage  # words | playing | done
    if "entry" not in st.session_state:
        st.session_state.entry = None
    if "composed_path" not in st.session_state:
        st.session_state.composed_path = None

_init()

db: SegmentationDB = st.session_state.db
word_index = st.session_state.word_index
sentences: List[SentenceEntry] = st.session_state.sentences

if not sentences:
    st.error("No sentences found. Check F:\\dataset\\isign\\word-presence-dataset_v1.1.csv")
    st.stop()

# ---------------- helpers ----------------
def load_entry(pos: int):
    ent = sentences[pos]
    ent.matched_videos = match_words_to_videos(ent.sentence, word_index)
    ent.compute_hash()
    st.session_state.entry = ent
    st.session_state.composed_path = None

def compose_if_needed(ent: SentenceEntry) -> Path:
    if ent.composed_path and ent.composed_path.is_file():
        return ent.composed_path
    out = COMPOSED_VIDEOS_DIR / f"{ent.sentence_id}.mp4"
    if not out.is_file():
        compose_sentence_video(ent.matched_videos, out)
    ent.composed_path = out
    st.session_state.composed_path = out
    return out

def next_sentence():
    st.session_state.idx = (st.session_state.idx + 1) % len(sentences)
    st.session_state.stage = "words"
    load_entry(st.session_state.idx)

# ---------------- UI ----------------
pos = st.session_state.idx
ent: SentenceEntry = st.session_state.entry or sentences[pos]
if st.session_state.entry is None:
    load_entry(pos)
    ent = st.session_state.entry

st.subheader(f"Sentence {pos + 1} / {len(sentences)}  [{ent.sentence_id}]")
st.write(f"**{ent.sentence}**")
st.write(f"Hash: `{ent.hash}`")

# ---- Stage 1: word videos large, stacked ----
if st.session_state.stage == "words":
    st.markdown("### Word videos (autoplay, in order)")
    wvs = ent.matched_videos
    if not wvs:
        st.warning("No matching word videos found for this sentence.")
    else:
        for i, wv in enumerate(wvs):
            st.caption(f"{i+1}. {wv.word}  ({wv.category})")
            try:
                st.video(wv.path.read_bytes(), format="video/mp4")
            except Exception as e:
                st.error(f"Cannot play {wv.path.name}: {e}")

    st.divider()
    c1, c2 = st.columns(2)
    if c1.button("🔁 r - repeat words", use_container_width=True):
        st.experimental_rerun()
    if c2.button("␣ space - begin sentence (slow mo)", type="primary", use_container_width=True):
        compose_if_needed(ent)
        save_checkpoint(st.session_state.idx, ent.sentence_id, "playing")
        st.session_state.stage = "playing"
        st.experimental_rerun()

# ---- Stage 2: play sentence video slow mo ----
elif st.session_state.stage == "playing":
    st.markdown("### Sentence video (slow motion)")
    path = st.session_state.composed_path or compose_if_needed(ent)
    st.write(f"Video: `{path}`")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        st.error("Cannot open composed video.")
    else:
        fps = cap.get(cv2.CAP_PROP_FPS) or COMPOSED_FPS
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        placeholder = st.empty()
        prog = st.progress(0)
        status = st.empty()

        col_sp, col_skip = st.columns([2, 1])
        mark = col_sp.button("␣ SPACE - mark segmentation", use_container_width=True)
        skip = col_skip.button("⏭ skip", use_container_width=True)

        if skip:
            cap.release()
            save_checkpoint(st.session_state.idx, ent.sentence_id, "done")
            next_sentence()
            st.experimental_rerun()

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            placeholder.image(rgb, channels="RGB", use_container_width=True)
            prog.progress(min(frame_idx / max(total - 1, 1), 1.0))
            ts = frame_idx / fps
            status.text(f"Frame {frame_idx}/{total}  •  {ts:.3f}s  •  slow x{SLOWDOWN_FACTOR}")

            if mark:
                wvs = ent.matched_videos
                wi = min(int((frame_idx / max(total - 1, 1)) * len(wvs)), len(wvs) - 1) if wvs else 0
                wv = wvs[wi] if wvs else None
                if wv:
                    db.append(
                        sentence_hash=ent.hash,
                        sentence=ent.sentence,
                        sentence_video=str(path),
                        word_video=str(wv.path),
                        word=wv.word,
                        frame_index=frame_idx,
                        timestamp=ts,
                    )
                    st.success(f"Recorded frame {frame_idx} ({ts:.3f}s) for '{wv.word}'")
                st.session_state.mark_btn = False
            time.sleep(1.0 / fps * SLOWDOWN_FACTOR)

        cap.release()
        placeholder.empty()
        prog.empty()
        status.empty()
        st.session_state.stage = "done"

# ---- Stage 3: done - redo or continue ----
elif st.session_state.stage == "done":
    st.markdown("### Sentence complete")
    st.write(f"**{ent.sentence}**")
    st.write(f"Saved to: `{SEGMENTATION_CSV}`")

    c1, c2 = st.columns(2)
    if c1.button("🔁 r - redo this sentence", use_container_width=True):
        save_checkpoint(st.session_state.idx, ent.sentence_id, "words")
        st.session_state.stage = "words"
        st.experimental_rerun()
    if c2.button("␣ space - continue to next sentence", type="primary", use_container_width=True):
        save_checkpoint(st.session_state.idx, ent.sentence_id, "done")
        next_sentence()
        st.experimental_rerun()

# ---------------- CSV preview ----------------
st.divider()
st.subheader("📄 segmentation.csv (last 10 rows)")
try:
    df = db.load()
    st.dataframe(df.tail(10), use_container_width=True)
except Exception as e:
    st.info(f"CSV not yet created or unreadable: {e}")
