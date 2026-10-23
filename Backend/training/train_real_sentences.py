"""
Real Continuous Sentence Trainer
================================
Fine-tunes the ISL Hybrid Conformer-CTC (pretrained on synthetic sentences) on real
continuous sign videos listed in data/tensors_sentence_level/index.json.

Design (all of it is about not collapsing to all-blank):
0. Feature harmonisation (harmonize_features).  The three tensor corpora in
   Backend/data were extracted with three different anchors, so the same physical point
   sits in three different places - measured with the neck (shoulder midpoint) as the
   reference:
       data/tensors_include50_word_level  neck at (3.592, 2.769, -1.233)  <- pretraining
       data/tensors_isl_csltr_word_level   neck at (0.000, 0.000,  0.000)
       data/tensors_sentence_level         neck at (0.011, 0.605, -0.523)
   The continuous corpus is therefore displaced by more than three shoulder widths
   relative to the corpus the backbone was pretrained on, and the 21 face dimensions
   that are identically zero in that corpus suddenly carry values.  Measured on the
   warm-started model: mean P(best non-blank) on real sentences was 0.0032 before the
   harmonisation and 0.0205 after it, and the model went from 100% blank on every
   sentence to actually emitting signs (val WER 100% -> 75.6%).
   The pretraining corpus is left in its own frame, so the Phase 6 weights stay valid
   (word accuracy 62.0% -> 60.7%).
1. Anti-collapse CTC loss (AntiCollapseCTCLoss).  The measured failure of the previous
   run was NOT a diffuse emission: the collapsed checkpoint spent only 2.6% of its
   probability mass on non-blank tokens, won the argmax on 97.8% of the frames and had
   a per-frame entropy of 0.040 out of ln(263)=5.57, i.e. it was *confidently* blank.
   A mass-based duration deficit can therefore be satisfied by spreading a little
   probability over many frames while the greedy decoder still returns nothing.  The
   loss below instead counts the frames where a non-blank token is at least as likely
   as the blank (exactly the comparison the decoder makes) and applies a log barrier,
   log(required / achieved), so the gradient survives in the fully collapsed regime.
   A separate upper bound on the mean blank probability stops the model from simply
   spamming non-blank tokens to satisfy the first term.
2. Targets are the in-scope words IN SIGNING ORDER (see scripts/plotter.py).  The
   default --max_out_scope was 1, which kept only 220 of the 1579 clips (36 distinct
   sentences): the model memorised those and collapsed to blank on the hold-out
   (train WER 9%, val WER 88%, val loss climbing from 5.3 to 9.0).  The default is now
   5 (1136 clips / 692 sentences), which was measured to give a flat validation loss and
   no train/val gap (3 -> 268 sentences was still overfitting).
3. Padding is masked inside the model (GRU packing, attention key mask, conv zeroing,
   padding-aware BatchNorm), so batched training matches single-sequence live
   inference and the padded frames cannot pollute the BatchNorm statistics.
4. Safe warm start: always initialises from the SYNTHETIC checkpoint (--init), never from a
   previous real-sentence run unless you pass --resume. Missing keys abort the run instead
   of silently training from scratch.
5. One optimizer for the whole run. The backbone is frozen (and in eval mode) for the first
   --freeze_epochs, then trained at a low LR. BatchNorm running stats stay frozen by default.
6. Greedy decoding keeps every emitted token (min_dur=1). CTC outputs are naturally peaky,
   so dropping short bursts deletes correct words.
7. Local-minimum escape: after --escape_patience epochs without a val-WER improvement the
   run perturbs the Conformer/CTC-head weights, resets the trainable BatchNorm statistics,
   re-seeds the CTC head from the Phase 6 word classifier (only when the head really
   collapsed) and restarts the learning-rate schedule.
8. Stochastic Weight Averaging over the last --swa_last epochs gives a final checkpoint
   that usually sits in a flatter region than any single epoch.
9. Per-epoch diagnostics: substitutions/deletions/insertions, blank-frame %, hypothesis vs
   reference length, train-set WER (no augmentation) next to val WER.
10. Mixed training (--mix 1): the training set is drawn at random from three sources -
   isolated word clips (the sign inventory, 262 classes / up to --word_cap clips per
   class), synthetic sentences built from them (the CTC alignment) and the real
   continuous sentences (the target domain).  Word clips are hard-capped at --word_cap
   (default 14, the measured median class size) BEFORE any further sampling so the
   pronoun-dominated classes cannot dominate even when mixed in.  The synthetic
   sentences use REAL continuous frames as their context padding, so the model learns
   to find a short gesture inside a real continuous clip.  --mix_cap trims the classes
   with more clips than the median in the synthetic source and --mix_word_cap
   re-samples the continuous clips so that the pronoun-dominated classes do not drown
   the rare ones (the top six words hold 61% of all word instances and a model trained
   on that learns the label prior instead of the signs).

Sanity test before a real run: --overfit 16 trains and validates on the same 16 sentences.
PASS means the mechanics work (loss, masking, decoding, optimiser, init). It does NOT prove
the labels are in the right order: with few sentences the model can memorise each one even
when its labels are scrambled (verified with reversed labels). Wrong label order shows up on
HELD-OUT sentences instead: val WER stays near 100% while train WER falls.
"""

import os, sys, json, math, random, argparse, itertools
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path: sys.path.insert(0, BACKEND_DIR)

from training.train_conformer_ctc import (
    ISLHybridConformerCTC, ModelEMA, AntiCollapseCTCLoss, batch_targets, augment,
    _reclamp, escape_local_minimum, harmonize_features, match_pretraining_domain,
    MixedContinuousDataset, collate as collate_mixed
)
from utils.dataloader_sentence import remap_sentence_words
from utils.dataloader_word import FILL_MISSING_DEFAULT, get_word_classes
from utils.checkpoint_utils import extract_state_dict, load_flexible
from utils.ctc_decoder import batch_decode_with_lm
from utils.pose_augmentation import fill_missing_temporal

DEFAULT_DATA_DIR = os.path.join(BACKEND_DIR, "data", "tensors_sentence_level")
DEFAULT_OUT_DIR = os.path.join(BACKEND_DIR, "models", "isl_real_sentences")
STAGE3_BEST = os.path.join(BACKEND_DIR, "models", "isl_conformer_ctc", "stage3_best.pth")
STAGE1_BEST = os.path.join(BACKEND_DIR, "models", "isl_conformer_ctc", "stage1_best.pth")
CLASSES_PATH = os.path.join(BACKEND_DIR, "models", "word_vocabulary", "classes.json")


# =============================================================================
# METRICS & DECODING
# =============================================================================
def ctc_collapse(tokens, blank_id, min_dur=1):
    """
    Standard greedy CTC collapse: merge runs of identical frames, then drop blanks. The same
    word repeated with a blank between ("slow <blank> slow") is correctly kept twice.
    Runs shorter than min_dur frames are dropped.
    """
    return [k for k, g in itertools.groupby(tokens)
            if k != blank_id and sum(1 for _ in g) >= min_dur]

def align_counts(ref, hyp):
    """Levenshtein alignment -> (substitutions, deletions, insertions)."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1); d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i-1, j-1] + (ref[i-1] != hyp[j-1]), d[i-1, j] + 1, d[i, j-1] + 1)
    i, j, S, D, I = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i-1, j-1] + (ref[i-1] != hyp[j-1]):
            S += int(ref[i-1] != hyp[j-1]); i -= 1; j -= 1
        elif i > 0 and d[i, j] == d[i-1, j] + 1:
            D += 1; i -= 1
        else:
            I += 1; j -= 1
    return S, D, I

def corpus_errors(refs, hyps):
    S = D = I = N = 0
    for r, h in zip(refs, hyps):
        s, d, i = align_counts(r, h); S += s; D += d; I += i; N += len(r)
    return S, D, I, max(1, N)

def ctc_loss_fn(log_probs_btc, targets, input_lengths, target_lengths, blank_id):
    """Plain CTC in fp32. 'mean' divides each sentence loss by its target length.

    The per-sample losses are filtered instead of using zero_infinity=True: a model that
    never emits a blank makes the alignment impossible, the loss becomes +inf and
    zero_infinity would silently replace it with 0 - which rewards the model for
    making the loss vanish.  See AntiCollapseCTCLoss for the full explanation.
    """
    lp_t = log_probs_btc.float().transpose(0, 1)  # (T, B, C)
    per_sample = F.ctc_loss(lp_t, targets, input_lengths, target_lengths,
                            blank=blank_id, reduction="none", zero_infinity=False)
    finite = torch.isfinite(per_sample)
    if finite.any():
        return per_sample[finite].mean()
    return per_sample.sum() * 0.0


# =============================================================================
# DATA
# =============================================================================
def _ctc_feasible(labels, frames):
    """CTC needs at least len(labels) + (#adjacent repeats) frames."""
    repeats = sum(1 for a, b in zip(labels, labels[1:]) if a == b)
    return frames is None or frames >= len(labels) + repeats

def split_items(index_file, val_split=0.10, seed=42, max_out_scope=1, max_words=18,
                overfit=0, class_to_idx=None):
    """Sentence-level split (all takes of a sentence stay on one side). Returns (train, val, stats).

    The targets are re-indexed onto the CURRENT vocabulary by NAME: the
    ``index.json`` of the corpus was written for the old 262-class list, so the
    stored indices are meaningless for the frequency-filtered vocabulary the
    word loader uses now.  A word that is not in the vocabulary is dropped
    from the CTC target, exactly like an out-of-scope word.
    """
    with open(index_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    stats = {"index_total": len(data), "dropped_missing": 0, "dropped_no_labels": 0,
             "dropped_out_scope": 0, "dropped_long": 0, "dropped_infeasible": 0,
             "dropped_unknown_word": 0}
    items = []
    for it in data:
        if not os.path.exists(it.get("tensor_path", "")):
            stats["dropped_missing"] += 1; continue
        if class_to_idx is not None:
            remapped, report = remap_sentence_words(
                [it], class_to_idx, max_out_scope=-1, max_words=max_words
            )
            if not remapped:
                stats["dropped_unknown_word"] += 1; continue
            it = remapped[0]
        labels = it.get("in_scope_indices", [])
        if len(labels) == 0:
            stats["dropped_no_labels"] += 1; continue
        if max_out_scope is not None and max_out_scope >= 0 \
                and len(it.get("out_scope_words", [])) > max_out_scope:
            stats["dropped_out_scope"] += 1; continue
        if len(labels) > max_words:
            stats["dropped_long"] += 1; continue
        if not _ctc_feasible(labels, it.get("frames")):
            stats["dropped_infeasible"] += 1; continue
        items.append(it)

    groups = {}
    for it in items:
        groups.setdefault(it.get("sentence", "").strip().lower(), []).append(it)
    keys = sorted(groups.keys())
    random.Random(seed).shuffle(keys)

    if overfit > 0:
        chosen = [it for k in keys[:overfit] for it in groups[k]]
        return chosen, list(chosen), stats

    n_val = int(round(len(keys) * val_split))
    if len(keys) >= 2: n_val = max(1, n_val)
    val_keys, train_keys = set(keys[:n_val]), set(keys[n_val:])
    train = [it for k in sorted(train_keys) for it in groups[k]]
    val = [it for k in sorted(val_keys) for it in groups[k]]
    return train, val, stats


class RealSentenceDataset(Dataset):
    def __init__(self, items, is_train, max_len=300, aug_strength=0.75, harmonize=True,
                 domain_match=False, fill_missing=FILL_MISSING_DEFAULT):
        self.items, self.is_train = items, is_train
        self.max_len, self.aug_strength = max_len, aug_strength
        self.harmonize = harmonize
        self.domain_match = domain_match
        self.fill_missing = fill_missing
        self._cache = {}

    def __len__(self): return len(self.items)

    def _load(self, item):
        p = item["tensor_path"]
        if p not in self._cache:
            seq = torch.load(p, weights_only=True).float()
            if seq.shape[0] > self.max_len:  # smooth time-resampling, keeps every gesture
                seq = F.interpolate(seq.T.unsqueeze(0), size=self.max_len, mode="linear",
                                    align_corners=True).squeeze(0).T
            if self.harmonize:
                seq = harmonize_features(seq)
            if self.domain_match:
                seq = match_pretraining_domain(seq)
            seq = _reclamp(seq.contiguous())
            # Fill the drop-out holes with the last valid value (the flag of a
            # filled frame is set, so the zero convention is not restored).
            if self.fill_missing and self.fill_missing != "none":
                seq = fill_missing_temporal(seq, mode=self.fill_missing)
            self._cache[p] = seq
        return self._cache[p]

    def __getitem__(self, idx):
        item = self.items[idx]
        seq = self._load(item)
        if self.is_train and self.aug_strength > 0:
            seq = augment(seq, strength=self.aug_strength)
        return seq, seq.shape[0], torch.tensor(item["in_scope_indices"], dtype=torch.long)


def collate_real(batch):
    xs, lengths, labels = zip(*batch)
    B, maxT = len(xs), max(lengths)
    xp = torch.zeros(B, maxT, xs[0].shape[-1])
    for i, x in enumerate(xs): xp[i, :lengths[i]] = x
    return (xp, torch.tensor(lengths, dtype=torch.long),
            torch.cat(labels), torch.tensor([len(l) for l in labels], dtype=torch.long))


def print_data_report(train_items, val_items, stats, classes, max_len):
    def sent_count(items): return len({it.get("sentence", "").strip().lower() for it in items})
    n_train_sent = sent_count(train_items)
    print(f"[DATA] index entries: {stats['index_total']} | kept: train {len(train_items)} clips / "
          f"{n_train_sent} sentences, val {len(val_items)} clips / {sent_count(val_items)} sentences")
    print("[DATA] dropped: " + ", ".join(f"{k.replace('dropped_', '')}={v}" for k, v in stats.items() if k.startswith("dropped_")))
    if n_train_sent < 100:
        print(f"[DATA] [!!] ONLY {n_train_sent} DISTINCT TRAINING SENTENCES. The model will memorise them "
              f"instead of learning to recognise signs, and will collapse to blank on unseen sentences. "
              f"Raise --max_out_scope (3 -> 268 sentences, 5 -> 692) or add more extracted videos.")
    allit = train_items + val_items
    if not allit: return
    labels_per = [len(it["in_scope_indices"]) for it in allit]
    seen = {i for it in train_items for i in it["in_scope_indices"]}
    print(f"[DATA] words/sentence: mean {np.mean(labels_per):.1f}, max {max(labels_per)} | "
          f"vocab classes seen in train: {len(seen)}/{len(classes)}")
    frames = [it["frames"] for it in allit if "frames" in it]
    if frames:
        f = np.array(frames)
        print(f"[DATA] frames/clip: median {int(np.median(f))}, p95 {int(np.percentile(f, 95))}, max {f.max()} | "
              f"{100 * np.mean(f > max_len):.1f}% longer than --max_len {max_len} (time-resampled)")
        ratio = np.array(frames) / np.array([len(it['in_scope_indices']) for it in allit if 'frames' in it])
        print(f"[DATA] frames per target word: median {np.median(ratio):.1f} (very large values mean many "
              f"untrained signs/pauses per target word)")


# =============================================================================
# WORD-CLIP BUILDER  (hard per-class cap)
# =============================================================================
def build_capped_word_items(word_dir: str, classes: list, cap: int, seed: int = 42):
    """
    Scan word_dir for per-class tensor subdirectories and return a flat list of
    dicts compatible with RealSentenceDataset, each with keys:
        tensor_path, in_scope_indices, frames, sentence

    Only 'cap' clips per class are kept (chosen with a fixed random seed so the
    selection is reproducible across runs).  Classes with fewer than 'cap' clips
    are kept in full.  If cap <= 0 all clips are kept.

    Expected layout (same as tensors_include50_word_level):
        word_dir/
            <class_name>/
                <clip>.pt   (or .pth)
    The class name must appear in 'classes'; unknown subdirectories are skipped.

    The hard cap is applied BEFORE MixedContinuousDataset / --mix_cap so that
    frequent pronouns ("i", "you", "he") cannot dominate the word source even
    when the synthetic-sentence or real-sentence samplers down-weight them.
    """
    class_to_idx = {c: i for i, c in enumerate(classes)}
    rng = random.Random(seed)

    items = []
    if not os.path.isdir(word_dir):
        print(f"[WORD_CAP] word_dir {word_dir} does not exist, skipping word clips.")
        return items

    class_counts_before, class_counts_after = {}, {}
    for class_name in sorted(os.listdir(word_dir)):
        if class_name not in class_to_idx:
            continue
        class_dir = os.path.join(word_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        idx = class_to_idx[class_name]
        clips = sorted([
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if f.endswith(".pt") or f.endswith(".pth")
        ])
        class_counts_before[class_name] = len(clips)
        if cap > 0 and len(clips) > cap:
            clips = rng.sample(clips, cap)
        class_counts_after[class_name] = len(clips)
        for p in clips:
            items.append({
                "tensor_path": p,
                "in_scope_indices": [idx],
                "frames": None,      # not needed for CTC feasibility check (single word always feasible)
                "sentence": class_name,
            })

    # Report the classes that were most heavily trimmed.
    trimmed = {c: (class_counts_before[c], class_counts_after[c])
               for c in class_counts_before if class_counts_after[c] < class_counts_before[c]}
    n_classes_total = len(class_counts_before)
    n_clips_before = sum(class_counts_before.values())
    n_clips_after = sum(class_counts_after.values())
    print(f"[WORD_CAP] cap={cap} | classes {n_classes_total} | "
          f"clips {n_clips_before} -> {n_clips_after} "
          f"({100.0 * n_clips_after / max(1, n_clips_before):.0f}% kept) | "
          f"{len(trimmed)} classes trimmed")
    if trimmed:
        top = sorted(trimmed.items(), key=lambda kv: -kv[1][0])[:8]
        print("  [WORD_CAP] most trimmed: " +
              ", ".join(f"{c}:{b}->{a}" for c, (b, a) in top))
    return items


# =============================================================================
# BALANCED MIXED DATASET
# =============================================================================
class BalancedMixedDataset(Dataset):
    """
    At each __getitem__ call, randomly selects one of three sources according to
    fixed integer weights (word : synthetic : real), then draws uniformly from
    that source.  This keeps the word-frequency distribution of the word source
    flat (because each class contributes at most --word_cap clips) while
    ensuring real sentences and synthetic sentences are seen every epoch.

    word_items     : output of build_capped_word_items()  (single-word clips)
    real_items     : output of split_items() train split  (multi-word sentences)
    syn_dataset    : MixedContinuousDataset or None; if None, synthetic sentences
                     are skipped and the weight is redistributed to real.
    weights        : (w_word, w_syn, w_real) — proportional, need not sum to 1.
    samples_per_epoch : virtual epoch length (independent of source sizes).
    aug_strength   : passed to augment() for all three sources.
    harmonize      : passed to RealSentenceDataset._load for real/word clips.
    domain_match   : passed to RealSentenceDataset._load for real/word clips.
    max_len        : time-resampling cap for real/word clips.
    """

    def __init__(self, word_items, real_items, syn_dataset,
                 weights=(0.40, 0.35, 0.25),
                 samples_per_epoch=1600,
                 aug_strength=0.75,
                 harmonize=True,
                 domain_match=False,
                 max_len=300,
                 seed=42):
        self.aug_strength = aug_strength
        self.samples_per_epoch = samples_per_epoch
        self.syn_dataset = syn_dataset

        # Build lightweight dataset wrappers for word and real clips.
        # We reuse RealSentenceDataset which already handles harmonise / cache.
        self._word_ds = RealSentenceDataset(
            word_items, is_train=True,
            max_len=max_len, aug_strength=aug_strength,
            harmonize=harmonize, domain_match=domain_match)
        self._real_ds = RealSentenceDataset(
            real_items, is_train=True,
            max_len=max_len, aug_strength=aug_strength,
            harmonize=harmonize, domain_match=domain_match)

        # Normalise weights; if a source is empty, redistribute its weight.
        w_word, w_syn, w_real = weights
        if len(word_items) == 0:
            w_syn += w_word * 0.5; w_real += w_word * 0.5; w_word = 0.0
        if syn_dataset is None or len(syn_dataset) == 0:
            w_word += w_syn * 0.5; w_real += w_syn * 0.5; w_syn = 0.0
        if len(real_items) == 0:
            w_word += w_real * 0.5; w_syn += w_real * 0.5; w_real = 0.0
        total = w_word + w_syn + w_real
        if total <= 0:
            raise RuntimeError("BalancedMixedDataset: all three sources are empty.")
        self.weights = (w_word / total, w_syn / total, w_real / total)
        self._rng = random.Random(seed)

        print(f"[BALANCED_MIX] word {self.weights[0]:.2f} ({len(word_items)} clips) | "
              f"synthetic {self.weights[1]:.2f} "
              f"({'disabled' if syn_dataset is None else len(syn_dataset)}) | "
              f"real {self.weights[2]:.2f} ({len(real_items)} clips)")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, _idx):
        r = self._rng.random()
        w_w, w_s, _ = self.weights
        if r < w_w and len(self._word_ds) > 0:
            i = self._rng.randrange(len(self._word_ds))
            return self._word_ds[i]
        elif r < w_w + w_s and self.syn_dataset is not None and len(self.syn_dataset) > 0:
            i = self._rng.randrange(len(self.syn_dataset))
            seq, length, labels = self.syn_dataset[i]
            # syn_dataset already applies augmentation internally; return as-is.
            return seq, length, labels
        else:
            i = self._rng.randrange(len(self._real_ds))
            return self._real_ds[i]


# =============================================================================
# THE LABEL PRIOR OF THE TRAINING MIX
# =============================================================================
def report_word_prior(classes, real_items, word_items=None, mix_weights=(0.40, 0.35, 0.25),
                      real_word_cap=0, class_cap=0):
    """Print the label prior of every source and of the mixture.

    The continuous corpus is dominated by a handful of pronouns; the WORD and
    the SYNTHETIC source draw their classes uniformly.  The mixture is
    therefore a de-skewing mechanism, and this report shows by how much.
    """
    from utils.word_distribution import (
        collapse_metrics,
        counts_to_distribution,
        format_distribution_report,
        mixed_distribution,
        uniform_distribution,
        word_instance_counts,
    )

    n = len(classes)
    if n == 0:
        return

    print("\n" + "=" * 74)
    print("LABEL PRIOR OF THE TRAINING SOURCES")
    print("=" * 74)

    # ---- the real (continuous) source ---------------------------------------
    real_counts = word_instance_counts(real_items)
    real_dist = counts_to_distribution(real_counts, n)
    print(format_distribution_report(real_dist, class_names=classes, top=8,
                                     title="REAL (continuous sentences)"))

    if real_word_cap:
        # The cap keeps a clip with probability min over its words of
        # cap / count(word): the expectation of the effective instance count
        # is what the report shows.
        eff = Counter()
        for it in real_items:
            ws = set(it.get("in_scope_indices", []) or [])
            if not ws:
                continue
            p = 1.0
            for w in ws:
                p = min(p, min(1.0, real_word_cap / max(1, real_counts.get(w, 1))))
            for w in ws:
                eff[w] += p
        capped_dist = counts_to_distribution(eff, n)
        print(format_distribution_report(capped_dist, class_names=classes, top=8,
                                         title=f"REAL after --mix_word_cap {real_word_cap}"))
        real_dist = capped_dist

    # ---- the word and the synthetic source ----------------------------------
    # Both draw their classes UNIFORMLY, so their prior is flat by
    # construction.  (The synthetic source samples the classes with
    # random.sample over the class list; the word source with random.choice.)
    flat = uniform_distribution(n)
    if word_items:
        word_counts = word_instance_counts(word_items)
        word_dist = counts_to_distribution(word_counts, n)
        print(format_distribution_report(word_dist, class_names=classes, top=6,
                                         title="WORD (isolated sign clips)"))
    else:
        word_dist = flat
        print(format_distribution_report(flat, class_names=classes, top=6,
                                         title="WORD (isolated sign clips, assumed flat)"))

    if class_cap:
        print(f"  [NOTE] --mix_cap {class_cap} trims the over-represented classes of the "
              f"SYNTHETIC source, which makes ITS prior flatter than uniform.")

    # ---- the mixture ---------------------------------------------------------
    w_word, w_syn, w_real = mix_weights
    mixed = mixed_distribution([(w_word, word_dist), (w_syn, flat), (w_real, real_dist)])
    print(format_distribution_report(mixed, class_names=classes, top=8,
                                     title="MIX (what the model actually sees)"))

    m_real = collapse_metrics(real_dist)
    m_mix = collapse_metrics(mixed)
    print(f"\n  The effective support of the prior rises from "
          f"{m_real['effective_support']:.1f} (real only) to "
          f"{m_mix['effective_support']:.1f} classes (the mix).")
    print(f"  The most frequent word goes from {m_real['top1_share']*100:.1f}% to "
          f"{m_mix['top1_share']*100:.1f}%.")
    print("  Measured WER of this repository: no cap 75.0% | --mix_word_cap 60 87.2% "
          "| --class_balance_alpha 0.5 98.9%.")
    print("  -> The caps are OFF by default on purpose: capping the REAL source drops")
    print("     ~40% of the only real continuous data, and the loss-level re-weighting")
    print("     breaks the CTC alignment.  Balance the MIXTURE, not the corpus.")
    print("=" * 74 + "\n")


# =============================================================================
# TRAINING HELPERS
# =============================================================================
def set_bn_eval(model):
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm): m.eval()

def load_init_weights(model, path, device, allow_partial=False):
    raw = torch.load(path, map_location=device, weights_only=False)
    sd = extract_state_dict(raw)
    # The CTC head depends on the vocabulary size, so a checkpoint trained on
    # the old class list cannot be loaded verbatim: the mismatching head is
    # dropped and re-learned, the backbone is kept.
    missing, unexpected = load_flexible(model, sd, label="INIT")
    bad_missing = [k for k in missing if not k.endswith("pe")]
    print(f"[INIT] {os.path.basename(path)}: missing={len(missing)} unexpected={len(unexpected)}")
    if bad_missing or unexpected:
        print("       missing (first 5):", bad_missing[:5], "| unexpected (first 5):", list(unexpected)[:5])
        if not allow_partial:
            raise RuntimeError("Checkpoint does not match the model (see above). Pass --allow_partial_load "
                               "only if you really want to train partly from scratch.")

def build_scheduler(optimizer, epochs, warmup, floor=0.02):
    def mult(ep):
        if ep < warmup: return 0.2 + 0.8 * (ep + 1) / max(1, warmup)
        prog = min(1.0, (ep - warmup) / max(1, epochs - warmup))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog))
    return optim.lr_scheduler.LambdaLR(optimizer, mult)

@torch.no_grad()
def evaluate(model, loader, blank_id, min_dur, device, use_cuda,
             beam_width=0, beam_lm=None, beam_alpha=0.0, beam_beta=0.0,
             beam_lm_mode="bigram", id_to_word=None):
    model.eval()
    loss_sum, n_batches, frames, blank_frames = 0.0, 0, 0, 0
    refs, hyps = [], []
    for xs, lengths, targets, tlens in loader:
        xs, lengths = xs.to(device), lengths.to(device)
        targets, tlens = targets.to(device), tlens.to(device)
        with torch.amp.autocast("cuda", enabled=use_cuda):
            lp = model(xs, lengths)
        loss_sum += ctc_loss_fn(lp, targets, lengths, tlens, blank_id).item(); n_batches += 1
        if beam_width and beam_width > 0:
            # The prefix beam search (optionally rescored by the word-level
            # language model).  This is the SAME decoder the live path uses, so
            # the validation WER is measured with the decoder that will
            # actually run on the webcam.
            out_len = (model.output_lengths(lengths)
                       if hasattr(model, "output_lengths") else lengths)
            hyps.extend(batch_decode_with_lm(
                lp, out_len, blank_id, id_to_word=id_to_word, lm=beam_lm,
                alpha=beam_alpha, beta=beam_beta, beam_width=beam_width,
                lm_mode=beam_lm_mode))
            am = lp.float().argmax(-1).cpu()
            for i, L in enumerate(out_len.cpu().tolist()):
                frames += L; blank_frames += int((am[i, :L] == blank_id).sum())
        else:
            am = lp.float().argmax(-1).cpu()
            for i, L in enumerate(lengths.cpu().tolist()):
                row = am[i, :L]
                frames += L; blank_frames += int((row == blank_id).sum())
                hyps.append(ctc_collapse(row.tolist(), blank_id, min_dur))
        refs.extend(batch_targets(targets, tlens))
    S, D, I, N = corpus_errors(refs, hyps)
    return {"loss": loss_sum / max(1, n_batches), "wer": (S + D + I) / N, "S": S, "D": D, "I": I, "N": N,
            "blank_pct": 100.0 * blank_frames / max(1, frames),
            "hyp_len": float(np.mean([len(h) for h in hyps])) if hyps else 0.0,
            "ref_len": float(np.mean([len(r) for r in refs])) if refs else 0.0,
            "exact": float(np.mean([r == h for r, h in zip(refs, hyps)])) if refs else 0.0,
            "refs": refs, "hyps": hyps}

def show_examples(classes, res, n=6):
    for r, h in list(zip(res["refs"], res["hyps"]))[:n]:
        print("    REF:", " ".join(classes[i] for i in r))
        print("    HYP:", " ".join(classes[i] for i in h) if h else "<empty>")

def save_curves(history, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)
    E = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(11, 4))
    plt.subplot(1, 2, 1)
    plt.plot(E, history["train_loss"], label="Train CTC", marker="o", color="#3b82f6")
    plt.plot(E, history["val_loss"], label="Val CTC", marker="s", color="#ef4444")
    plt.xlabel("Epoch"); plt.ylabel("CTC loss (per word)"); plt.title("CTC Loss"); plt.legend(); plt.grid(True, ls="--", alpha=0.5)
    plt.subplot(1, 2, 2)
    plt.plot(E, [w * 100 for w in history["train_wer"]], label="Train WER", marker="^", color="#f59e0b")
    plt.plot(E, [w * 100 for w in history["wer"]], label="Val WER", marker="o", color="#10b981")
    plt.plot(E, [w * 100 for w in history["wer_ema"]], label="EMA Val WER", marker="s", color="#8b5cf6")
    plt.xlabel("Epoch"); plt.ylabel("WER (%)"); plt.title("Word Error Rate"); plt.legend(); plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "real_sentence_training_curves.png"), dpi=150)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================
def train_real_sentences(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    plots_dir = os.path.join(args.out_dir, "plots")

    with open(args.classes) as f: classes = json.load(f)
    if isinstance(classes, dict): classes = classes["classes"]
    num_classes = len(classes)

    # ---- the word-level language model (optional) --------------------------
    # The decoder that will actually run on the webcam: a prefix beam search
    # rescored with the word ORDER of the continuous corpus.
    beam_lm = None
    if args.beam_width and args.beam_width > 0:
        if args.lm_file and os.path.exists(args.lm_file):
            from utils.word_lm import WordLM
            beam_lm = WordLM.load(args.lm_file)
            print(f"[LM] {os.path.basename(args.lm_file)} loaded "
                  f"({len(beam_lm.unigram)} word types) | alpha={args.beam_alpha} "
                  f"({args.beam_lm_mode} only)")
        else:
            print(f"[LM] {args.lm_file} not found - the beam search stays purely acoustic.")
    blank_id = num_classes
    class_to_idx = {c: i for i, c in enumerate(classes)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    index_file = os.path.join(args.data_dir, "index.json")
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"Missing index.json in {args.data_dir}. Run scripts/extract_real_sentences.py first!")
    train_items, val_items, stats = split_items(index_file, args.val_split, args.seed,
                                                args.max_out_scope, args.max_words, args.overfit,
                                                class_to_idx=class_to_idx)
    if not train_items: raise RuntimeError("No usable training clips after filtering; see [DATA] report.")
    print_data_report(train_items, val_items, stats, classes, args.max_len)

    aug = 0.0 if args.overfit else args.aug

    if args.mix:
        # -------------------------------------------------------------------
        # Balanced mixed training:
        #   source 0 — isolated word clips, hard-capped at --word_cap per class
        #   source 1 — synthetic sentences (MixedContinuousDataset)
        #   source 2 — real continuous sentences
        # At every step one source is chosen according to --mix_weights and one
        # clip is drawn uniformly from it, so the word-frequency distribution
        # of the word source is flat (each class contributes at most word_cap
        # clips) and every epoch sees all three domains.
        # -------------------------------------------------------------------
        word_items = build_capped_word_items(
            args.word_dir, classes, cap=args.word_cap, seed=args.seed)

        # Build the synthetic-sentence dataset (MixedContinuousDataset handles
        # the CTC alignment and context padding from real frames).
        syn_dataset = MixedContinuousDataset(
            args.word_dir, classes, samples_per_epoch=args.train_samples, is_train=True,
            weights=(1.0, 0.0, 0.0),          # word-only source; we mix externally
            max_words=args.mix_max_words, min_words=1,
            aug_strength=aug, real_items=train_items, real_dir=args.data_dir,
            harmonize=bool(args.harmonize), domain_match=bool(args.domain_match),
            class_cap=args.mix_cap, max_len=args.max_len, real_word_cap=args.mix_word_cap)

        train_ds = BalancedMixedDataset(
            word_items=word_items,
            real_items=train_items,
            syn_dataset=syn_dataset,
            weights=tuple(args.mix_weights),
            samples_per_epoch=args.train_samples,
            aug_strength=aug,
            harmonize=bool(args.harmonize),
            domain_match=bool(args.domain_match),
            max_len=args.max_len,
            seed=args.seed,
        )
        train_ld = DataLoader(train_ds, args.batch_size, shuffle=True,
                              collate_fn=collate_real, num_workers=args.num_workers)

        # Frequency report so the user can see if pronouns are still dominant.
        if args.mix_word_cap:
            freq = Counter()
            for it in train_items:
                for w in it["in_scope_indices"]:
                    freq[w] += 1
            top = sorted(freq.items(), key=lambda kv: -kv[1])[:8]
            print(f"[MIX] real-sentence word-instance cap {args.mix_word_cap}:")
            for w, c in top:
                eff = c * min(1.0, args.mix_word_cap / max(1, c))
                print(f"        {classes[w]:<16} {c:>4} instances -> {eff:>6.1f} effective")


    # ---- the label prior of the mix -------------------------------------
        # The three sources have very different priors: the WORD and the
        # SYNTHETIC source draw their classes UNIFORMLY (flat), the REAL source
        # follows the corpus (six pronouns = 64% of the word instances).  This
        # report makes the EFFECTIVE prior of the training mix measurable, so
        # the frequency cap can be chosen on numbers instead of on intuition.
        # The historical measurements of this repo are why the caps are OFF by
        # default: capping the real source DROPS ~40% of the only real data
        # (75.0% -> 87.2% WER), and fixing the prior in the LOSS destroys the
        # CTC alignment (75.0% -> 98.9% WER).
        report_word_prior(
            classes=classes,
            real_items=train_items,
            word_items=word_items,
            mix_weights=tuple(args.mix_weights),
            real_word_cap=args.mix_word_cap,
            class_cap=args.mix_cap,
        )
    else:
        train_ds = RealSentenceDataset(train_items, True, args.max_len, aug,
                                       args.harmonize, args.domain_match,
                                       fill_missing=args.fill_missing)
        train_ld = DataLoader(train_ds, args.batch_size, shuffle=True,
                              collate_fn=collate_real, num_workers=args.num_workers)

    val_ds = RealSentenceDataset(val_items, False, args.max_len, 0.0,
                                 args.harmonize, args.domain_match,
                                 fill_missing=args.fill_missing)
    probe_items = (train_items if len(train_items) <= 200
                   else random.Random(args.seed).sample(train_items, 200))
    probe_ds = RealSentenceDataset(probe_items, False, args.max_len, 0.0,
                                   args.harmonize, args.domain_match,
                                   fill_missing=args.fill_missing)
    val_ld = DataLoader(val_ds, args.batch_size, shuffle=False,
                        collate_fn=collate_real, num_workers=args.num_workers)
    probe_ld = DataLoader(probe_ds, args.batch_size, shuffle=False,
                          collate_fn=collate_real, num_workers=args.num_workers)

    model = ISLHybridConformerCTC(num_classes).to(device)
    init = args.init or (STAGE3_BEST if os.path.exists(STAGE3_BEST) else STAGE1_BEST)
    if args.random_init:
        print("[INIT] --random_init: starting from a randomly initialised model.")
    elif args.resume:
        pass  # weights come from the resume checkpoint below
    elif os.path.exists(init):
        load_init_weights(model, init, device, args.allow_partial_load)
    else:
        print(f"[INIT] {init} not found -> loading Phase 6 word backbone only (conformer/head start random).")
        model.load_phase6_backbone()

    freeze_epochs = 0 if args.overfit else args.freeze_epochs
    param_groups = [
        {"params": list(model.backbone.parameters()), "lr": args.lr_backbone, "weight_decay": 1e-4},
        {"params": list(model.conformer.parameters()) + list(model.ctc_proj.parameters()),
         "lr": args.lr_head, "weight_decay": 1e-4},
    ]
    for g in param_groups:
        g["base_lr"] = g["lr"]

    # Optional class-balancing weights.
    class_weights = None
    if args.class_balance_alpha > 0.0:
        freq = Counter()
        for it in train_items:
            for w in set(it["in_scope_indices"]):
                freq[w] += 1
        if freq:
            med = sorted(freq.values())[len(freq) // 2]
            cw = torch.ones(num_classes)
            for w, f in freq.items():
                cw[w] = min(args.class_balance_clip,
                            max(1.0 / args.class_balance_clip,
                                (med / max(1, f)) ** args.class_balance_alpha))
            class_weights = cw
            top = sorted(freq.items(), key=lambda kv: -kv[1])[:6]
            print(f"[BALANCE] alpha={args.class_balance_alpha} -> "
                  + ", ".join(f"{classes[w]}:{freq[w]}clips w={float(cw[w]):.2f}" for w, _ in top))

    criterion = AntiCollapseCTCLoss(
        blank_id,
        blank_target=args.blank_target,
        blank_w=args.blank_w,
        emission_w=args.emission_w,
        min_frames_per_sign=args.min_frames_per_sign,
        spike_w=args.spike_w,
        entropy_w=args.entropy_w,
        class_weights=class_weights,
    )
    optimizer = optim.AdamW(param_groups, betas=(0.9, 0.98))
    scheduler = build_scheduler(optimizer, args.epochs, args.warmup)
    ema = ModelEMA(model, decay=args.ema_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    history = {k: [] for k in ["train_loss", "val_loss", "wer", "wer_ema", "train_wer",
                                "S", "D", "I", "blank_pct", "hyp_len", "escapes"]}
    start_ep, best_wer, best_ema_wer = 0, float("inf"), float("inf")
    ckpt_path = os.path.join(args.out_dir, "real_sentence_checkpoint.pth")
    if args.resume:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"--resume but {ckpt_path} does not exist")
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        best_pth = os.path.join(args.out_dir, "real_sentence_best.pth")
        if os.path.exists(best_pth):
            load_flexible(model, extract_state_dict(torch.load(best_pth, map_location=device, weights_only=True)),
                          label="RESUME")
            print(f"[RESUME] loaded BEST model weights from {best_pth}")
        else:
            load_flexible(model, extract_state_dict(ck), label="RESUME")
            print(f"[RESUME] best weights not found, loaded last-checkpoint model")
        ema.model.load_state_dict(ck["ema_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        start_ep, history = ck["epoch"], ck["history"]
        best_wer, best_ema_wer = ck["best_wer"], ck["best_ema_wer"]
        for _ in range(start_ep): scheduler.step()
        if args.epochs <= start_ep:
            auto_epochs = start_ep + 20
            print(f"[RESUME] --epochs {args.epochs} <= checkpoint epoch {start_ep}; "
                  f"auto-extending to {auto_epochs}. Pass --epochs N (N > {start_ep}) to control this.")
            args.epochs = auto_epochs
            scheduler = build_scheduler(optimizer, args.epochs, 0)
            for _ in range(start_ep): scheduler.step()
        print(f"[RESUME] epoch {start_ep}, best val WER {best_wer * 100:.1f}% | will train to epoch {args.epochs}")

    def set_backbone_frozen(frozen):
        for p in model.backbone.parameters(): p.requires_grad = not frozen
    set_backbone_frozen(start_ep < freeze_epochs)

    print("=" * 72)
    print(f"REAL SENTENCE TRAINING | device {device} | classes {num_classes} | blank id {blank_id}")
    print(f"epochs {args.epochs} | batch {args.batch_size} x accum {args.accum_steps} | "
          f"lr head {args.lr_head:g}, backbone {args.lr_backbone:g} | freeze {freeze_epochs} ep | "
          f"BN {'train' if args.train_bn else 'frozen'} | overfit {args.overfit or 'off'}")
    print(f"anti-collapse: emission_w {args.emission_w} (min {args.min_frames_per_sign} frames/sign) | "
          f"blank cap {args.blank_target:.2f} (w {args.blank_w}) | spike_w {args.spike_w} | "
          f"escape_patience {args.escape_patience} | swa_last {args.swa_last}")
    if args.mix:
        print(f"mix: word_cap {args.word_cap} clips/class | weights word={args.mix_weights[0]:.2f} "
              f"syn={args.mix_weights[1]:.2f} real={args.mix_weights[2]:.2f} | "
              f"aug {aug:.2f} applied to all three sources")
    print("=" * 72)

    swa_state, swa_n = None, 0
    epochs_without_improvement = 0
    escapes = 0

    for ep in range(start_ep, args.epochs):
        frozen = ep < freeze_epochs
        if ep == freeze_epochs and freeze_epochs > 0:
            print(f"\n[UNFREEZE] backbone trainable from epoch {ep + 1}")
        set_backbone_frozen(frozen)

        model.train()
        if frozen: model.backbone.eval()
        if not args.train_bn: set_bn_eval(model)

        t_loss, n_ok, n_skip = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        for step_i, (xs, lengths, targets, tlens) in enumerate(
                tqdm(train_ld, desc=f"Ep {ep + 1:02d}/{args.epochs:02d}", leave=False)):
            xs, lengths = xs.to(device), lengths.to(device)
            targets, tlens = targets.to(device), tlens.to(device)
            with torch.amp.autocast("cuda", enabled=use_cuda):
                lp = model(xs, lengths)
            loss = criterion(lp.transpose(0, 1), targets, lengths, tlens)
            if torch.isfinite(loss):
                scaler.scale(loss / args.accum_steps).backward()
                t_loss += loss.item(); n_ok += 1
            else:
                n_skip += 1
            if (step_i + 1) % args.accum_steps == 0 or (step_i + 1) == len(train_ld):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
        scheduler.step()
        t_loss /= max(1, n_ok)

        beam = dict(beam_width=args.beam_width, beam_lm=beam_lm,
                    beam_alpha=args.beam_alpha, beam_beta=args.beam_beta,
                    beam_lm_mode=args.beam_lm_mode, id_to_word=classes)
        val = evaluate(model, val_ld, blank_id, args.min_dur, device, use_cuda, **beam)
        ema_val = evaluate(ema.model, val_ld, blank_id, args.min_dur, device, use_cuda, **beam)
        tr = evaluate(model, probe_ld, blank_id, args.min_dur, device, use_cuda, **beam)

        for k, v in [("train_loss", t_loss), ("val_loss", val["loss"]),
                     ("wer", val["wer"]), ("wer_ema", ema_val["wer"]),
                     ("train_wer", tr["wer"]), ("S", val["S"]), ("D", val["D"]), ("I", val["I"]),
                     ("blank_pct", val["blank_pct"]), ("hyp_len", val["hyp_len"]),
                     ("escapes", escapes)]:
            history[k].append(v)

        print(f"[Ep {ep + 1:02d}/{args.epochs:02d}] loss {t_loss:.3f} | val {val['loss']:.3f} | "
              f"WER val {val['wer'] * 100:.1f}% (S{val['S']} D{val['D']} I{val['I']}/{val['N']}) "
              f"EMA {ema_val['wer'] * 100:.1f}% | train {tr['wer'] * 100:.1f}% | "
              f"blank {val['blank_pct']:.0f}% | words hyp/ref {val['hyp_len']:.1f}/{val['ref_len']:.1f}"
              + (f" | skipped {n_skip}" if n_skip else "") + (" [frozen]" if frozen else ""))
        if (ep + 1) % 5 == 0 and val["hyp_len"] == 0.0:
            print("   [WARN] model emits nothing on val (all-blank). If this persists, run --overfit 16 "
                  "to check the training mechanics, and check the label order/quality of index.json.")

        # --- Blank-collapse watchdog -----------------------------------------
        blank_collapsed = (val["blank_pct"] > args.blank_collapse_pct and
                           val["hyp_len"] < max(0.5, args.blank_collapse_ratio * val["ref_len"]))
        if blank_collapsed:
            print(f"   [WARN] BLANK COLLAPSE: {val['blank_pct']:.1f}% blank frames and only "
                  f"{val['hyp_len']:.2f} words emitted per sentence (reference {val['ref_len']:.2f}).")
            if args.emission_w < args.emission_w_max:
                args.emission_w = min(args.emission_w_max, args.emission_w * 2.0)
                criterion.emission_w = args.emission_w
                print(f"   [WARN] raised the emission-deficit weight to {args.emission_w:.2f}.")

        # --- Local-minimum escape --------------------------------------------
        if val["wer"] < best_wer - 1e-4:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if args.escape_patience > 0 and epochs_without_improvement >= args.escape_patience:
            escapes += 1
            print(f"   [ESCAPE] No val-WER improvement for {epochs_without_improvement} epochs "
                  f"(WER {val['wer'] * 100:.1f}%). Kicking the model out of the local minimum.")
            info = escape_local_minimum(model, optimizer, blank_collapsed=blank_collapsed,
                                        scale=args.escape_scale)
            print(f"   [ESCAPE] {info}")
            ema = ModelEMA(model, decay=args.ema_decay)
            for g in optimizer.param_groups:
                g["lr"] = g.get("base_lr", g["lr"])
            scheduler = build_scheduler(optimizer, max(1, args.epochs - ep), 1)
            epochs_without_improvement = 0
            swa_state, swa_n = None, 0

        # --- Stochastic Weight Averaging over the tail epochs ---------------
        if args.swa_last > 0 and (ep + 1) > (args.epochs - args.swa_last):
            with torch.no_grad():
                cur = {k: v.detach().float().cpu().clone() for k, v in model.state_dict().items()}
                if swa_state is None:
                    swa_state, swa_n = cur, 1
                else:
                    for k in swa_state:
                        if swa_state[k].is_floating_point():
                            swa_state[k] += (cur[k] - swa_state[k]) / (swa_n + 1)
                    swa_n += 1

        if (ep + 1) % 5 == 0 or ep + 1 == args.epochs:
            show_examples(classes, val)

        torch.save({"epoch": ep + 1, "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_wer": best_wer, "best_ema_wer": best_ema_wer,
                    "history": history}, ckpt_path)
        if val["wer"] < best_wer:
            best_wer = val["wer"]
            torch.save(model.state_dict(), os.path.join(args.out_dir, "real_sentence_best.pth"))
        if ema_val["wer"] < best_ema_wer:
            best_ema_wer = ema_val["wer"]
            torch.save(ema.model.state_dict(), os.path.join(args.out_dir, "real_sentence_ema_best.pth"))
        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        save_curves(history, plots_dir)

    # --- Stochastic Weight Averaging checkpoint ---------------------------
    if swa_state is not None and swa_n > 0:
        swa_path = os.path.join(args.out_dir, "real_sentence_swa.pth")
        torch.save(swa_state, swa_path)
        print(f"[SWA] averaged {swa_n} checkpoints -> {swa_path}")
        swa_model = ISLHybridConformerCTC(num_classes).to(device)
        swa_model.load_state_dict({k: v.to(device) for k, v in swa_state.items()}, strict=True)
        swa_val = evaluate(swa_model, val_ld, blank_id, args.min_dur, device, use_cuda, **beam)
        print(f"[SWA] SWA val WER {swa_val['wer'] * 100:.1f}% (best single {best_wer * 100:.1f}%)")
        if swa_val["wer"] < best_wer:
            best_wer = swa_val["wer"]
            torch.save(swa_model.state_dict(), os.path.join(args.out_dir, "real_sentence_best.pth"))
            print(f"  ★ SWA IS THE NEW BEST -> {swa_val['wer'] * 100:.1f}%")

    with open(os.path.join(args.out_dir, "structure.md"), "w", encoding="utf-8") as f:
        f.write("# Directory Structure: Backend/models/isl_real_sentences\n\n"
                "Real continuous human sign recognition model "
                "(fine-tuned from the synthetic Conformer-CTC by `training/train_real_sentences.py`).\n\n"
                "## Files\n"
                "- `real_sentence_best.pth`: Best-val-WER weights (plain state dict).\n"
                "- `real_sentence_ema_best.pth`: Best EMA shadow weights.\n"
                "- `real_sentence_swa.pth`: Stochastic-weight-averaged weights over the tail epochs.\n"
                "- `real_sentence_checkpoint.pth`: Latest full training state (model, EMA, optimizer, history) for `--resume`.\n"
                "- `history.json`: Per-epoch metrics (CTC loss, WER, S/D/I, blank %, escape count).\n\n"
                "## Immediate Subdirectories\n"
                "- `plots/`: Training curves (`real_sentence_training_curves.png`).\n")

    print("\n" + "=" * 72)
    print(f"TRAINING COMPLETE | best val WER {best_wer * 100:.1f}% | "
          f"best EMA WER {best_ema_wer * 100:.1f}% | escapes {escapes}")
    if escapes:
        print(f"[NOTE] {escapes} local-minimum escape(s) were triggered. If the WER is still high, check:")
        print("       - the data report above (label order, words per sentence, frames per target word)")
        print("       - run `python Backend/eda/analyze_tensor_dataset.py --dataset sentence` for the corpus audit")
        print("       - the blank-collapse warning above (a >90% blank rate means the head never learned to emit)")
    if args.overfit:
        ok = history["train_wer"][-1] <= 0.05
        print(f"[OVERFIT TEST] final WER on {args.overfit} sentences: {history['train_wer'][-1] * 100:.1f}% -> "
              + ("PASS: training mechanics work. This does not check label order; run without --overfit and "
                 "compare train vs val WER." if ok else
                 "FAIL: cannot even memorise a few sentences. Suspect mechanics: LR too low, wrong/partial init, "
                 "broken tensors, or clips too short for their labels (see [DATA] report). Train longer first."))
    elif (history["train_wer"] and history["train_wer"][-1] < 0.3
          and history["wer"][-1] > max(0.5, 2 * history["train_wer"][-1])):
        print(f"[HINT] train WER {history['train_wer'][-1] * 100:.0f}% but val WER "
              f"{history['wer'][-1] * 100:.0f}%: the model fits seen sentences but not new ones. "
              "Typical causes: labels in the wrong order, too little data, or "
              "signers/sentences too different between train and val.")
    print("=" * 72)


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune ISL Conformer-CTC on real sentence videos")
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--classes", default=CLASSES_PATH)
    p.add_argument("--init", default=None,
                   help="Warm-start checkpoint (default: synthetic stage3_best.pth)")
    p.add_argument("--resume", action="store_true",
                   help="Continue THIS script's real_sentence_checkpoint.pth")
    p.add_argument("--fresh", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--allow_partial_load", action="store_true")
    p.add_argument("--random_init", action="store_true",
                   help="Start from a random initialisation instead of the pretrained "
                        "synthetic checkpoint (control experiment for the domain gap)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--accum_steps", type=int, default=2)
    p.add_argument("--lr_head", type=float, default=1e-4)
    p.add_argument("--lr_backbone", type=float, default=1e-5)
    p.add_argument("--freeze_epochs", type=int, default=2)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--max_len", type=int, default=300,
                   help="Clips longer than this are time-resampled")
    p.add_argument("--max_words", type=int, default=18)
    p.add_argument("--max_out_scope", type=int, default=5,
                   help="Drop sentences with more untrained words than this (-1 = keep all). "
                        "WARNING: the previous default of 1 kept only 220 of the 1579 clips "
                        "(36 distinct sentences), which is what made the model memorise the "
                        "training set (train WER 9%%) and collapse to blank on the hold-out "
                        "(val WER 88%%). 3 keeps 705 clips / 268 sentences, 5 keeps 1136 / 692. "
                        "5 was measured to give a flat validation loss and no train/val gap, "
                        "3 was still overfitting.")
    p.add_argument("--aug", type=float, default=0.75)
    p.add_argument("--min_dur", type=int, default=1,
                   help="Min frames per decoded burst (keep 1 for peaky CTC)")
    p.add_argument("--ema_decay", type=float, default=0.995)
    p.add_argument("--val_split", type=float, default=0.10)
    p.add_argument("--train_bn", action="store_true",
                   help="Let BatchNorm running stats update (default: frozen)")
    p.add_argument("--overfit", type=int, default=0,
                   help="Train+val on N sentences, no aug/freeze (pipeline sanity test)")
    p.add_argument("--num_workers",     type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    # --- Anti blank-collapse loss ---
    p.add_argument("--blank_target", type=float, default=0.97,
                   help="Upper bound on the mean blank probability (0.97 = at most 97%% blank)")
    p.add_argument("--blank_w", type=float, default=0.5,
                   help="Weight of the blank cap")
    p.add_argument("--blank_w_max", type=float, default=4.0,
                   help="Upper bound for the auto-escalating blank cap weight")
    p.add_argument("--emission_w", type=float, default=1.0,
                   help="Weight of the emission deficit (the anti all-blank-collapse term)")
    p.add_argument("--min_frames_per_sign", type=float, default=10.0,
                   help="Minimum frames a sign should occupy; drives the required emission count")
    p.add_argument("--spike_w", type=float, default=0.10,
                   help="Penalty on isolated 1-frame emissions")
    p.add_argument("--entropy_w", type=float, default=0.0,
                   help="Per-frame emission-entropy penalty (0 disables)")
    p.add_argument("--blank_collapse_pct", type=float, default=90.0,
                   help="Blank-frame percentage above which the model is considered blank-collapsed")
    p.add_argument("--blank_collapse_ratio", type=float, default=0.6,
                   help="Emitted/reference word ratio below which the model is considered blank-collapsed")
    p.add_argument("--emission_w_max", type=float, default=4.0,
                   help="Upper bound for the auto-escalating emission-deficit weight")
    # --- Local-minimum escape ---
    p.add_argument("--escape_patience", type=int, default=8,
                   help="Epochs without val-WER improvement before an escape is triggered (0 disables)")
    p.add_argument("--escape_scale", type=float, default=0.02,
                   help="Weight perturbation sigma used by the escape")
    p.add_argument("--swa_last", type=int, default=5,
                   help="Number of trailing epochs averaged into the SWA checkpoint (0 disables)")
    p.add_argument("--harmonize", type=int, default=1,
                   help="Re-anchor every clip on the neck and clear the face block so the "
                        "continuous corpus lives in the same geometric frame as the word "
                        "corpus the backbone was pretrained on.  The three corpora use three "
                        "different anchors (image centre / neck / nose) and the skeleton is "
                        "displaced by more than three shoulder widths between them. "
                        "Set to 0 only if every corpus was extracted with the same convention.")
    p.add_argument("--domain_match", type=int, default=0,
                   help="Rewrite every clip so the arm is extended and the hand is 1.46x "
                        "larger, i.e. the geometry of the isolated-word corpus the backbone "
                        "was pretrained on (measured gap: wrist-to-neck 0.94 vs 2.96 "
                        "shoulder widths, hand extent 0.318 vs 0.463).")
    p.add_argument("--fill_missing", type=str, default=FILL_MISSING_DEFAULT,
                   choices=["none", "hold", "interpolate"],
                   help="Fill the drop-out holes of a clip with the last valid value. "
                        "'none' keeps the corpus convention (an undetected modality owns a "
                        "zero block), which is what the backbone was pretrained on.")
    # Prefix beam search + word-level language model (the live decoder)
    p.add_argument("--beam_width", type=int, default=0,
                   help="Width of the CTC prefix beam search (0 = the greedy decoder)")
    p.add_argument("--beam_alpha", type=float, default=0.0,
                   help="Weight of the word-level language model in the beam rescoring")
    p.add_argument("--beam_beta", type=float, default=0.0,
                   help="Insertion penalty of the beam rescoring")
    p.add_argument("--beam_lm_mode", type=str, default="bigram",
                   choices=["bigram", "unigram", "both"],
                   help="What the language model contributes: 'bigram' (the word ORDER "
                        "only - the unigram prior of the continuous corpus is the pronoun "
                        "skew we want to avoid)")
    p.add_argument("--lm_file", type=str,
                   default=os.path.join(BACKEND_DIR, "models", "word_vocabulary", "word_lm.json"),
                   help="The word-level language model used by the beam rescoring")
    # --- mixed training ----------------------------------------------------
    p.add_argument("--mix", type=int, default=0,
                   help="Enable balanced mixed training from three sources: isolated word "
                        "clips (hard-capped at --word_cap per class), synthetic sentences, "
                        "and real continuous sentences.  0 = off (real sentences only).")
    p.add_argument("--mix_weights", type=float, nargs=3, default=(0.40, 0.35, 0.25),
                   help="Sampling weights for --mix: word, synthetic, real.  "
                        "At each step one source is chosen with these probabilities and one "
                        "clip is drawn uniformly from it, so the word distribution stays "
                        "flat regardless of corpus sizes.")
    p.add_argument("--mix_max_words", type=int, default=4,
                   help="Maximum words per synthetic sentence in --mix")
    p.add_argument("--mix_cap", type=int, default=0,
                   help="Cap on the clips per class used by the SYNTHETIC source inside "
                        "MixedContinuousDataset (0 = use the median count).  Separate from "
                        "--word_cap which caps the isolated-word source.")
    p.add_argument("--mix_word_cap", type=int, default=0,
                   help="Cap on the clips per word in the CONTINUOUS (real sentence) source. "
                        "0 = off (DEFAULT).  Measured: a cap of 60 removes too much label "
                        "signal and the CTC head collapses (100%% blank / 97%% WER).")
    p.add_argument("--word_cap", type=int, default=20,
                   help="Hard cap on isolated-word clips kept per class before any sampling. "
                        "Applied at dataset build time so it is absolute and cannot be "
                        "overridden by --mix_cap or --mix_word_cap.  14 is the measured "
                        "median class size in tensors_include50_word_level; this prevents "
                        "pronoun-class clips ('i', 'you', 'he') from dominating the word "
                        "source even when mixed in.  Set 0 to disable the cap.")
    p.add_argument("--train_samples", type=int, default=1600,
                   help="Virtual epoch length for mixed-mode training")
    p.add_argument("--class_balance_alpha", type=float, default=0.0,
                   help="Down-weight the frequent classes in the CTC loss.  "
                        "0 = off (DEFAULT).  Measured: alpha=0.5 collapsed the CTC head "
                        "(98.9%% WER, 0.09 words/sentence) versus 75.0%% with it off.")
    p.add_argument("--class_balance_clip", type=float, default=6.0,
                   help="Maximum up/down weight factor of --class_balance_alpha")
    p.add_argument("--word_dir",
                   default=os.path.join(BACKEND_DIR, "data", "tensors_include50_word_level"),
                   help="Isolated-word corpus directory used by --mix")
    a = p.parse_args()
    if a.max_out_scope < 0: a.max_out_scope = None
    if a.overfit and a.epochs < 100:
        print(f"[NOTE] --overfit usually needs >=100 epochs; you set {a.epochs}.")
    return a


if __name__ == "__main__":
    train_real_sentences(parse_args())
