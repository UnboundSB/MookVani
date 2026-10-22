"""
Sequence-level knowledge distillation for the CTC sentence model.
==============================================================

The problem this solves
-----------------------
The CTC sentence model has to learn two things at once: WHAT sign is being
signed and WHERE it is in time.  The CTC loss only supervises the second one
(the alignment), and the first one has to be inferred from the (very weak)
evidence of the isolated-word corpus.  The result is the failure this
repository keeps hitting: the head collapses onto the blank and the training
loses 7-8 epochs before it escapes.

The student-teacher setup
-------------------------
Two networks look at the SAME sign in two different ways:

    TEACHER  an isolated word clip  ->  a clean class distribution over the
             vocabulary (the word model is trained on exactly that data and
             reaches ~69% accuracy, so its belief about a word is much better
             than anything the sentence model can extract from a stitched
             clip)

    STUDENT  a synthetic sentence (word clips patched together with
             co-articulation)  ->  per-frame log-probabilities

The synthetic generator KNOWS where every word sits in the stitched sequence
(it built the sequence by concatenating the clips), so no alignment has to be
guessed: the frames of word *i* are known exactly.  The consistency loss then
says: *inside the frames of word i, the student must express what the teacher
believes about word i*.

Two consequences:

1. **FILTERING.**  A word is only used as a target when the teacher agrees
   with the ground truth (``teacher_argmax == label``) and is confident
   enough (``teacher_conf >= threshold``).  The words the teacher gets wrong
   are exactly the ambiguous ones, and their synthetic sentences are exactly
   the ones that teach the student noise.  They are dropped.

2. **A DENSE SIGNAL.**  The teacher supplies a full distribution per word
   instead of a single label, so the student gets the "dark knowledge" of the
   word model: how much the teacher believes in the runners-up.  That is a
   much stronger training signal than the one-hot CTC target, and it is
   exactly what breaks the blank collapse.

The loss
--------
For the word *i* occupying the frames ``[s_i, e_i)``:

    p_teacher  = softmax(teacher(word_clip_i))            # (C,)
    p_student  = mean_t softmax(student_logprob[t])       # over t in [s_i, e_i)
    loss_i     = KL(p_teacher || p_student)

and the total loss is ``ctc_loss + lambda * mean_i loss_i`` over the words
that passed the filter.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# The teacher
# --------------------------------------------------------------------------- #
class WordTeacher(nn.Module):
    """A frozen word model used as the teacher of the sentence model.

    The teacher is the Phase-6 word classifier: a Transformer + TCNN that
    pools the frames of an isolated word clip and predicts its class.

    Its head was trained on the OLD class list, so the rows are re-mapped BY
    CLASS NAME onto the current vocabulary.  A class the teacher does not know
    gets a probability of zero (and the word is dropped by the filter).
    """

    def __init__(self, checkpoint_path: str, classes: Sequence[str],
                 teacher_classes_path: Optional[str] = None, device="cpu"):
        super().__init__()
        from training.phase6_trainer import Phase6Model

        self.device = device
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw

        # how many classes did the teacher learn?
        head_rows = None
        for key in ("head.4.weight", "head.3.weight", "head.2.weight", "classifier.4.weight"):
            if key in state:
                head_rows = state[key].shape[0]
                break
        if head_rows is None:
            raise RuntimeError(f"Could not find the classifier head in {checkpoint_path}")

        # the teacher's own class list
        if teacher_classes_path is None:
            teacher_classes_path = os.path.join(os.path.dirname(checkpoint_path), "classes.json")
        if os.path.exists(teacher_classes_path):
            with open(teacher_classes_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            teacher_classes = payload["classes"] if isinstance(payload, dict) else payload
        else:
            teacher_classes = None

        self.model = Phase6Model(head_rows)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # the permutation: current class index -> teacher row (-1 = unknown)
        self.row_map: List[int] = []
        if teacher_classes:
            src = {c: i for i, c in enumerate(teacher_classes)}
            self.row_map = [src.get(c, -1) for c in self.classes]
        else:
            self.row_map = list(range(min(head_rows, len(self.classes))))

        self.to(device)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """clips: (B, T, 163) -> (B, C) probabilities over the CURRENT classes."""
        logits = self.model(clips)                      # (B, teacher_rows)
        probs = F.softmax(logits.float(), dim=-1)

        if self.row_map and max(self.row_map) < probs.shape[1]:
            idx = torch.tensor([max(0, r) for r in self.row_map], device=probs.device)
            probs = probs[:, idx]
            # a class the teacher does not know gets zero probability
            unknown = torch.tensor([r < 0 for r in self.row_map], device=probs.device)
            if unknown.any():
                probs = probs.masked_fill(unknown.unsqueeze(0), 0.0)
        return probs

    @torch.no_grad()
    def predict_clips(self, clips: torch.Tensor) -> torch.Tensor:
        return self.forward(clips)


# --------------------------------------------------------------------------- #
# The consistency loss
# --------------------------------------------------------------------------- #
def sequence_distillation_loss(
    student_log_probs: torch.Tensor,
    teacher_probs: torch.Tensor,
    spans: Sequence[Tuple[int, int]],
    labels: Sequence[int],
    teacher_conf_threshold: float = 0.5,
    reduction: str = "mean",
):
    """The student-teacher consistency loss of one sample.

    Parameters
    ----------
    student_log_probs : (T, C+1) log-probabilities of the stitched sentence
    teacher_probs    : (N, C) teacher probabilities, one row per word
    spans            : the frames ``[start, end)`` of every word in the sentence
    labels           : the class index of every word
    teacher_conf_threshold : a word is only used when the teacher is at least
        this confident AND its argmax matches the ground truth.

    Returns ``(loss, n_used)`` - the loss is 0 when no word passed the filter.
    """
    device = student_log_probs.device
    T = student_log_probs.shape[0]
    p_student_all = student_log_probs.exp()          # (T, C+1)

    losses = []
    for i, (start, end) in enumerate(spans):
        if i >= len(labels) or i >= teacher_probs.shape[0]:
            break
        if end <= start or start >= T:
            continue
        end = min(end, T)

        t_probs = teacher_probs[i]
        conf, t_arg = t_probs.max(dim=-1)
        # THE FILTER: the teacher has to agree with the ground truth and to be
        # confident enough.  A word the teacher gets wrong is ambiguous, and
        # its synthetic sentence is noise.
        if t_arg.item() != int(labels[i]):
            continue
        if float(conf) < teacher_conf_threshold:
            continue

        p_s = p_student_all[start:end, :-1]           # the non-blank columns
        p_s = p_s.mean(dim=0)                          # the student's belief in the word
        p_s = p_s / p_s.sum().clamp(min=1e-8)

        t = t_probs / t_probs.sum().clamp(min=1e-8)
        losses.append((t * (torch.log(t.clamp(min=1e-8))
                            - torch.log(p_s.clamp(min=1e-8)))).sum())

    if not losses:
        zero = student_log_probs.sum() * 0.0
        return zero, 0

    stacked = torch.stack(losses)
    if reduction == "sum":
        return stacked.sum(), len(losses)
    return stacked.mean(), len(losses)


def batch_distillation_loss(
    student_log_probs: torch.Tensor,
    teacher_probs: torch.Tensor,
    spans_per_sample: Sequence[Sequence[Tuple[int, int]]],
    labels_per_sample: Sequence[Sequence[int]],
    teacher_conf_threshold: float = 0.5,
):
    """The consistency loss over a whole batch.

    ``student_log_probs`` is (B, T, C+1), ``teacher_probs`` is (B, N, C) with
    the words of every sample padded.
    """
    total, n_used = 0.0, 0
    parts = []
    for b in range(student_log_probs.shape[0]):
        loss, used = sequence_distillation_loss(
            student_log_probs[b], teacher_probs[b],
            spans_per_sample[b], labels_per_sample[b],
            teacher_conf_threshold=teacher_conf_threshold)
        if used:
            parts.append(loss)
            n_used += used
    if not parts:
        return student_log_probs.sum() * 0.0, 0
    return torch.stack(parts).mean(), n_used
