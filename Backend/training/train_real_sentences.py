"""
Real Continuous Sentence Trainer
================================
Fine-tunes the ISL Hybrid Conformer-CTC (pretrained on synthetic sentences) on real
continuous sign videos listed in data/tensors_sentence_level/index.json.

Design (all of it is about not collapsing to all-blank):
1. Plain CTC loss (mean over target length), computed in fp32. No focal weighting and no
   duration-deficit term: the deficit term was satisfiable by spreading a little
   probability over many frames, so it never prevented the all-blank solution.
2. Targets are the in-scope words IN SIGNING ORDER (see scripts/plotter.py). Sentences with
   too many untrained words are dropped (--max_out_scope) because each untrained word is a
   real sign the model would be told to treat as blank.
3. Padding is masked inside the model (GRU packing, attention key mask, conv zeroing), so
   batched training matches single-sequence live inference.
4. Safe warm start: always initialises from the SYNTHETIC checkpoint (--init), never from a
   previous real-sentence run unless you pass --resume. Missing keys abort the run instead
   of silently training from scratch.
5. One optimizer for the whole run. The backbone is frozen (and in eval mode) for the first
   --freeze_epochs, then trained at a low LR. BatchNorm running stats stay frozen by default.
6. Greedy decoding keeps every emitted token (min_dur=1). CTC outputs are naturally peaky,
   so dropping short bursts deletes correct words.
7. Per-epoch diagnostics: substitutions/deletions/insertions, blank-frame %, hypothesis vs
   reference length, train-set WER (no augmentation) next to val WER.

Sanity test before a real run: --overfit 16 trains and validates on the same 16 sentences.
PASS means the mechanics work (loss, masking, decoding, optimiser, init). It does NOT prove
the labels are in the right order: with few sentences the model can memorise each one even
when its labels are scrambled (verified with reversed labels). Wrong label order shows up on
HELD-OUT sentences instead: val WER stays near 100% while train WER falls.
"""

import os, sys, json, math, random, argparse, itertools
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
    ISLHybridConformerCTC, ModelEMA, batch_targets, augment
)

DEFAULT_DATA_DIR = os.path.join(BACKEND_DIR, "data", "tensors_sentence_level")
DEFAULT_OUT_DIR = os.path.join(BACKEND_DIR, "models", "isl_real_sentences")
STAGE3_BEST = os.path.join(BACKEND_DIR, "models", "isl_conformer_ctc", "stage3_best.pth")
STAGE1_BEST = os.path.join(BACKEND_DIR, "models", "isl_conformer_ctc", "stage1_best.pth")
CLASSES_PATH = os.path.join(BACKEND_DIR, "models", "phase6_word_model", "classes.json")


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
    """Plain CTC in fp32. 'mean' divides each sentence loss by its target length."""
    lp_t = log_probs_btc.float().transpose(0, 1)  # (T, B, C)
    return F.ctc_loss(lp_t, targets, input_lengths, target_lengths,
                      blank=blank_id, reduction="mean", zero_infinity=True)


# =============================================================================
# DATA
# =============================================================================
def _ctc_feasible(labels, frames):
    """CTC needs at least len(labels) + (#adjacent repeats) frames."""
    repeats = sum(1 for a, b in zip(labels, labels[1:]) if a == b)
    return frames is None or frames >= len(labels) + repeats

def split_items(index_file, val_split=0.10, seed=42, max_out_scope=1, max_words=18, overfit=0):
    """Sentence-level split (all takes of a sentence stay on one side). Returns (train, val, stats)."""
    with open(index_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    stats = {"index_total": len(data), "dropped_missing": 0, "dropped_no_labels": 0,
             "dropped_out_scope": 0, "dropped_long": 0, "dropped_infeasible": 0}
    items = []
    for it in data:
        if not os.path.exists(it.get("tensor_path", "")):
            stats["dropped_missing"] += 1; continue
        labels = it.get("in_scope_indices", [])
        if len(labels) == 0:
            stats["dropped_no_labels"] += 1; continue
        if max_out_scope is not None and len(it.get("out_scope_words", [])) > max_out_scope:
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
    def __init__(self, items, is_train, max_len=300, aug_strength=0.75):
        self.items, self.is_train = items, is_train
        self.max_len, self.aug_strength = max_len, aug_strength
        self._cache = {}

    def __len__(self): return len(self.items)

    def _load(self, item):
        p = item["tensor_path"]
        if p not in self._cache:
            seq = torch.load(p, weights_only=True).float()
            if seq.shape[0] > self.max_len:  # smooth time-resampling, keeps every gesture
                seq = F.interpolate(seq.T.unsqueeze(0), size=self.max_len, mode="linear",
                                    align_corners=True).squeeze(0).T
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
    print(f"[DATA] index entries: {stats['index_total']} | kept: train {len(train_items)} clips / "
          f"{sent_count(train_items)} sentences, val {len(val_items)} clips / {sent_count(val_items)} sentences")
    print("[DATA] dropped: " + ", ".join(f"{k.replace('dropped_', '')}={v}" for k, v in stats.items() if k.startswith("dropped_")))
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
# TRAINING HELPERS
# =============================================================================
def set_bn_eval(model):
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm): m.eval()

def load_init_weights(model, path, device, allow_partial=False):
    raw = torch.load(path, map_location=device, weights_only=False)
    sd = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
    res = model.load_state_dict(sd, strict=False)
    bad_missing = [k for k in res.missing_keys if not k.endswith("pe")]  # positional buffers are rebuilt
    print(f"[INIT] {os.path.basename(path)}: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
    if bad_missing or res.unexpected_keys:
        print("       missing (first 5):", bad_missing[:5], "| unexpected (first 5):", list(res.unexpected_keys)[:5])
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
def evaluate(model, loader, blank_id, min_dur, device, use_cuda):
    model.eval()
    loss_sum, n_batches, frames, blank_frames = 0.0, 0, 0, 0
    refs, hyps = [], []
    for xs, lengths, targets, tlens in loader:
        xs, lengths = xs.to(device), lengths.to(device)
        targets, tlens = targets.to(device), tlens.to(device)
        with torch.amp.autocast("cuda", enabled=use_cuda):
            lp = model(xs, lengths)
        loss_sum += ctc_loss_fn(lp, targets, lengths, tlens, blank_id).item(); n_batches += 1
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
    blank_id = num_classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    index_file = os.path.join(args.data_dir, "index.json")
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"Missing index.json in {args.data_dir}. Run scripts/extract_real_sentences.py first!")
    train_items, val_items, stats = split_items(index_file, args.val_split, args.seed,
                                                args.max_out_scope, args.max_words, args.overfit)
    if not train_items: raise RuntimeError("No usable training clips after filtering; see [DATA] report.")
    print_data_report(train_items, val_items, stats, classes, args.max_len)

    aug = 0.0 if args.overfit else args.aug
    train_ds = RealSentenceDataset(train_items, True, args.max_len, aug)
    val_ds = RealSentenceDataset(val_items, False, args.max_len)
    probe_items = train_items if len(train_items) <= 200 else random.Random(args.seed).sample(train_items, 200)
    probe_ds = RealSentenceDataset(probe_items, False, args.max_len)   # train clips, no augmentation
    train_ld = DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=collate_real, num_workers=args.num_workers)
    val_ld = DataLoader(val_ds, args.batch_size, shuffle=False, collate_fn=collate_real, num_workers=args.num_workers)
    probe_ld = DataLoader(probe_ds, args.batch_size, shuffle=False, collate_fn=collate_real, num_workers=args.num_workers)

    model = ISLHybridConformerCTC(num_classes).to(device)
    init = args.init or (STAGE3_BEST if os.path.exists(STAGE3_BEST) else STAGE1_BEST)
    if args.resume:
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
    optimizer = optim.AdamW(param_groups, betas=(0.9, 0.98))
    scheduler = build_scheduler(optimizer, args.epochs, args.warmup)
    ema = ModelEMA(model, decay=args.ema_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    history = {k: [] for k in ["train_loss", "val_loss", "wer", "wer_ema", "train_wer", "S", "D", "I", "blank_pct", "hyp_len"]}
    start_ep, best_wer, best_ema_wer = 0, float("inf"), float("inf")
    ckpt_path = os.path.join(args.out_dir, "real_sentence_checkpoint.pth")
    if args.resume:
        if not os.path.exists(ckpt_path): raise FileNotFoundError(f"--resume but {ckpt_path} does not exist")
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"]); ema.model.load_state_dict(ck["ema_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        start_ep, history = ck["epoch"], ck["history"]
        best_wer, best_ema_wer = ck["best_wer"], ck["best_ema_wer"]
        for _ in range(start_ep): scheduler.step()
        print(f"[RESUME] epoch {start_ep}, best val WER {best_wer * 100:.1f}%")

    def set_backbone_frozen(frozen):
        for p in model.backbone.parameters(): p.requires_grad = not frozen
    set_backbone_frozen(start_ep < freeze_epochs)

    print("=" * 72)
    print(f"REAL SENTENCE TRAINING | device {device} | classes {num_classes} | blank id {blank_id}")
    print(f"epochs {args.epochs} | batch {args.batch_size} x accum {args.accum_steps} | lr head {args.lr_head:g}, "
          f"backbone {args.lr_backbone:g} | freeze {freeze_epochs} ep | BN {'train' if args.train_bn else 'frozen'} | "
          f"overfit {args.overfit or 'off'}")
    print("=" * 72)

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
        for step_i, (xs, lengths, targets, tlens) in enumerate(tqdm(train_ld, desc=f"Ep {ep + 1:02d}/{args.epochs:02d}", leave=False)):
            xs, lengths = xs.to(device), lengths.to(device)
            targets, tlens = targets.to(device), tlens.to(device)
            with torch.amp.autocast("cuda", enabled=use_cuda):
                lp = model(xs, lengths)
            loss = ctc_loss_fn(lp, targets, lengths, tlens, blank_id)
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

        val = evaluate(model, val_ld, blank_id, args.min_dur, device, use_cuda)
        ema_val = evaluate(ema.model, val_ld, blank_id, args.min_dur, device, use_cuda)
        tr = evaluate(model, probe_ld, blank_id, args.min_dur, device, use_cuda)

        for k, v in [("train_loss", t_loss), ("val_loss", val["loss"]), ("wer", val["wer"]), ("wer_ema", ema_val["wer"]),
                     ("train_wer", tr["wer"]), ("S", val["S"]), ("D", val["D"]), ("I", val["I"]),
                     ("blank_pct", val["blank_pct"]), ("hyp_len", val["hyp_len"])]:
            history[k].append(v)

        print(f"[Ep {ep + 1:02d}/{args.epochs:02d}] loss {t_loss:.3f} | val {val['loss']:.3f} | "
              f"WER val {val['wer'] * 100:.1f}% (S{val['S']} D{val['D']} I{val['I']}/{val['N']}) "
              f"EMA {ema_val['wer'] * 100:.1f}% | train {tr['wer'] * 100:.1f}% | "
              f"blank {val['blank_pct']:.0f}% | words hyp/ref {val['hyp_len']:.1f}/{val['ref_len']:.1f}"
              + (f" | skipped {n_skip}" if n_skip else "") + (" [frozen]" if frozen else ""))
        if (ep + 1) % 5 == 0 and val["hyp_len"] == 0.0:
            print("   [WARN] model emits nothing on val (all-blank). If this persists, run --overfit 16 to check the "
                  "training mechanics, and check the label order/quality of index.json.")
        if (ep + 1) % 5 == 0 or ep + 1 == args.epochs:
            show_examples(classes, val)

        torch.save({"epoch": ep + 1, "model_state_dict": model.state_dict(), "ema_state_dict": ema.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(), "best_wer": best_wer,
                    "best_ema_wer": best_ema_wer, "history": history}, ckpt_path)
        if val["wer"] < best_wer:
            best_wer = val["wer"]
            torch.save(model.state_dict(), os.path.join(args.out_dir, "real_sentence_best.pth"))
        if ema_val["wer"] < best_ema_wer:
            best_ema_wer = ema_val["wer"]
            torch.save(ema.model.state_dict(), os.path.join(args.out_dir, "real_sentence_ema_best.pth"))
        with open(os.path.join(args.out_dir, "history.json"), "w") as f: json.dump(history, f, indent=2)
        save_curves(history, plots_dir)

    with open(os.path.join(args.out_dir, "structure.md"), "w", encoding="utf-8") as f:
        f.write("# Directory Structure: Backend/models/isl_real_sentences\n\nReal continuous human sign recognition model "
                "(fine-tuned from the synthetic Conformer-CTC by `training/train_real_sentences.py`).\n\n## Files\n"
                "- `real_sentence_best.pth`: Best-val-WER weights (plain state dict).\n"
                "- `real_sentence_ema_best.pth`: Best EMA shadow weights.\n"
                "- `real_sentence_checkpoint.pth`: Latest full training state (model, EMA, optimizer, history) for `--resume`.\n"
                "- `history.json`: Per-epoch metrics (CTC loss, WER, S/D/I, blank %).\n\n## Immediate Subdirectories\n"
                "- `plots/`: Training curves (`real_sentence_training_curves.png`).\n")

    print("\n" + "=" * 72)
    print(f"TRAINING COMPLETE | best val WER {best_wer * 100:.1f}% | best EMA WER {best_ema_wer * 100:.1f}%")
    if args.overfit:
        ok = history["train_wer"][-1] <= 0.05
        print(f"[OVERFIT TEST] final WER on {args.overfit} sentences: {history['train_wer'][-1] * 100:.1f}% -> "
              + ("PASS: training mechanics work. This does not check label order; run without --overfit and "
                 "compare train vs val WER." if ok else
                 "FAIL: cannot even memorise a few sentences. Suspect mechanics: LR too low, wrong/partial init, "
                 "broken tensors, or clips too short for their labels (see [DATA] report). Train longer first."))
    elif history["train_wer"] and history["train_wer"][-1] < 0.3 and history["wer"][-1] > max(0.5, 2 * history["train_wer"][-1]):
        print(f"[HINT] train WER {history['train_wer'][-1] * 100:.0f}% but val WER {history['wer'][-1] * 100:.0f}%: the model fits "
              "seen sentences but not new ones. Typical causes: labels in the wrong order, too little data, or "
              "signers/sentences too different between train and val.")
    print("=" * 72)


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune ISL Conformer-CTC on real sentence videos")
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--classes", default=CLASSES_PATH)
    p.add_argument("--init", default=None, help="Warm-start checkpoint (default: synthetic stage3_best.pth)")
    p.add_argument("--resume", action="store_true", help="Continue THIS script's real_sentence_checkpoint.pth")
    p.add_argument("--fresh", action="store_true", help=argparse.SUPPRESS)  # old flag, now the default behaviour
    p.add_argument("--allow_partial_load", action="store_true")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--accum_steps", type=int, default=2)
    p.add_argument("--lr_head", type=float, default=1e-4)
    p.add_argument("--lr_backbone", type=float, default=1e-5)
    p.add_argument("--freeze_epochs", type=int, default=2)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--max_len", type=int, default=300, help="Clips longer than this are time-resampled")
    p.add_argument("--max_words", type=int, default=18)
    p.add_argument("--max_out_scope", type=int, default=1, help="Drop sentences with more untrained words than this (-1 = keep all)")
    p.add_argument("--aug", type=float, default=0.75)
    p.add_argument("--min_dur", type=int, default=1, help="Min frames per decoded burst (keep 1 for peaky CTC)")
    p.add_argument("--ema_decay", type=float, default=0.995)
    p.add_argument("--val_split", type=float, default=0.10)
    p.add_argument("--train_bn", action="store_true", help="Let BatchNorm running stats update (default: frozen)")
    p.add_argument("--overfit", type=int, default=0, help="Train+val on N sentences, no aug/freeze (pipeline sanity test)")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if a.max_out_scope < 0: a.max_out_scope = None
    if a.overfit and a.epochs < 100:
        print(f"[NOTE] --overfit usually needs >=100 epochs; you set {a.epochs}.")
    return a


if __name__ == "__main__":
    train_real_sentences(parse_args())