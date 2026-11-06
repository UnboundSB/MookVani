"""
Conformer-CTC model for isolated word classification.

Input:  (B, T, 163) pose/hand/face features
Output: (B, T, 229) log-probabilities over [blank, word_0..word_227]

Reuses MultiStreamEmbedding from Backend/models/isl_conformer.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
from models.isl_conformer import MultiStreamEmbedding  # noqa: E402


class ConformerBlock(nn.Module):
    """Single Conformer block: FFN -> MHSA -> Conv -> FFN."""

    def __init__(self, d_model: int, heads: int, d_ff: int, kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_ff), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.conv_norm = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=kernel, padding=kernel // 2, groups=d_model),
            nn.BatchNorm1d(d_model), nn.SiLU(),
            nn.Conv1d(d_model, d_model * 2, 1), nn.SiLU(),
            nn.Conv1d(d_model * 2, d_model, 1), nn.Dropout(dropout),
        )
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_ff), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask=None):
        x = x + 0.5 * self.ffn1(x)
        h = self.attn_norm(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        h = self.conv_norm(x).transpose(1, 2)
        h = self.conv(h).transpose(1, 2)
        x = x + h
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x)


class ConformerCTC(nn.Module):
    def __init__(self, num_words: int = 228, d_model: int = 128, heads: int = 4,
                 layers: int = 4, d_ff: int = 256, kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        self.num_words = num_words
        self.embed = MultiStreamEmbedding(d_model=d_model)
        self.input_proj = nn.Linear(d_model, d_model)
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, heads, d_ff, kernel, dropout) for _ in range(layers)
        ])
        self.head = nn.Linear(d_model, num_words + 1)  # +1 for CTC blank

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """
        x: (B, T, 163)
        lengths: (B,) valid frame counts
        returns: log_probs (B, T, num_words+1)
        """
        pad_mask = lengths_to_padding_mask(lengths, x.shape[1])  # (B, T) True=padding
        h = self.embed(x)                                       # (B, T, d_model)
        h = self.input_proj(h)
        for blk in self.blocks:
            h = blk(h, key_padding_mask=pad_mask)
        logits = self.head(h)                                   # (B, T, num_words+1)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs


def lengths_to_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    batch_size = lengths.size(0)
    arange = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
    return arange >= lengths.unsqueeze(1)
