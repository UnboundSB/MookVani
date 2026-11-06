"""
Evaluation for word-level CTC: top-1 / top-5 accuracy on held-out split.
Greedy decode (collapse repeats, remove blank) then compare to target class.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from segmentor.word_ctc.dataset import WordCTCDataset, collate_fn
from segmentor.word_ctc.model import ConformerCTC


def greedy_decode(log_probs: torch.Tensor, lengths: torch.Tensor):
    """
    log_probs: (B, T, C)
    lengths: (B,)
    returns: list of predicted class indices (int), blank=0 removed, repeats collapsed
    """
    preds = log_probs.argmax(dim=-1)  # (B, T)
    results = []
    for b in range(preds.shape[0]):
        T = int(lengths[b].item())
        seq = preds[b, :T].tolist()
        collapsed = []
        prev = None
        for p in seq:
            if p != prev:
                collapsed.append(p)
                prev = p
        # remove blank (0)
        word_seq = [p for p in collapsed if p != 0]
        # isolated word: take the most frequent non-blank, or first if tie
        if len(word_seq) == 0:
            results.append(None)
        else:
            results.append(word_seq[0])
    return results


def evaluate_word_ctc(checkpoint: str | None = None, batch_size: int = 8,
                      device: str | None = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    val_ds = WordCTCDataset(split="val", max_per_class=20)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_fn, pin_memory=True)

    model = ConformerCTC(num_words=val_ds.num_classes).to(device)
    ckpt_path = Path(checkpoint) if checkpoint else BACKEND_DIR / "models" / "word_ctc" / "best_model.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    correct_top1 = 0
    correct_top5 = 0
    total = 0
    blank_preds = 0

    with torch.no_grad():
        for x, lengths, labels in val_loader:
            x = x.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            log_probs = model(x, lengths)  # (B, T, C)
            probs = torch.exp(log_probs)

            # top-1 greedy decode
            pred_words = greedy_decode(log_probs, lengths)
            for p, t in zip(pred_words, labels.tolist()):
                total += 1
                if p is None:
                    blank_preds += 1
                    continue
                if p == t:
                    correct_top1 += 1

            # top-5: mean over frames of top-5 probs containing target
            top5 = torch.topk(probs, k=5, dim=-1).indices  # (B, T, 5)
            for b in range(x.shape[0]):
                T = int(lengths[b].item())
                target = int(labels[b].item())
                found = False
                for t in range(T):
                    if int(target) in top5[b, t].tolist():
                        found = True
                        break
                if found:
                    correct_top5 += 1

    acc1 = correct_top1 / max(total, 1)
    acc5 = correct_top5 / max(total, 1)
    print(f"[EVAL] total={total} top1={acc1:.4f} top5={acc5:.4f} blank_preds={blank_preds}")
    return {"top1": acc1, "top5": acc5, "total": total, "blank_preds": blank_preds}
