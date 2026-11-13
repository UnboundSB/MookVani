"""
Minimal ISL Conformer — Single-layer Transformer (64-d) + TCNN Head

Input:  (B, T, 163)  — 163-dim MediaPipe pose features
Output: (B, num_classes) — Classification logits (word-level)

Architecture:
  1. Multi-stream embedding (hands/arms/face) → 64-d
  2. 1-layer Transformer encoder (64-d, 4 heads)
  3. TCNN (1D Conv) head → Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiStreamEmbedding(nn.Module):
    """Project hands/arms/face streams to shared 64-d."""
    def __init__(self, d_model=64):
        super().__init__()
        # Hands: 126 coords + 2 flags = 128
        self.hand_proj = nn.Sequential(
            nn.LayerNorm(128), nn.Linear(128, d_model // 2))
        # Arms: 12 coords + 1 flag = 13
        self.arm_proj = nn.Sequential(
            nn.LayerNorm(13), nn.Linear(13, d_model // 4))
        # Face: 21 coords + 1 flag = 22
        self.face_proj = nn.Sequential(
            nn.LayerNorm(22), nn.Linear(22, d_model // 4))

    def forward(self, x):
        # x: (B, T, 163)
        hands = x[:, :, 0:126]
        arms  = x[:, :, 126:138]
        face  = x[:, :, 138:159]
        flags = x[:, :, 159:163]

        hand_stream = torch.cat([hands, flags[:, :, 0:2]], dim=-1)  # 128
        arm_stream  = torch.cat([arms, flags[:, :, 2:3]], dim=-1)   # 13
        face_stream = torch.cat([face, flags[:, :, 3:4]], dim=-1)   # 22

        h = self.hand_proj(hand_stream)   # (B, T, 32)
        a = self.arm_proj(arm_stream)     # (B, T, 16)
        f = self.face_proj(face_stream)   # (B, T, 16)

        return torch.cat([h, a, f], dim=-1)  # (B, T, 64)


class FeedForward(nn.Module):
    def __init__(self, d_model, mult=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerLayer(nn.Module):
    """Single Transformer encoder layer (Pre-LN)."""
    def __init__(self, d_model=64, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout=dropout)

    def forward(self, x, key_padding_mask=None):
        # Self-attention
        res = x
        x = self.attn_norm(x)
        x, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = res + x

        # Feed-forward
        res = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = res + x
        return x


class TCNNHead(nn.Module):
    """Temporal CNN classifier head."""
    def __init__(self, d_model=64, num_classes=100, kernel=3, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, kernel, padding=kernel // 2, groups=d_model),
            nn.BatchNorm1d(d_model * 2),
            nn.GELU(),
            nn.Conv1d(d_model * 2, d_model, 1),
            nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, lengths=None):
        # x: (B, T, D) -> (B, D, T)
        x = x.transpose(1, 2)
        x = self.conv(x)
        if lengths is not None:
            # Mask padded positions
            mask = torch.arange(x.size(-1), device=x.device).expand(x.size(0), -1) >= lengths.unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(1), 0)
        x = self.pool(x).squeeze(-1)  # (B, D)
        return self.classifier(x)     # (B, num_classes)


class MinimalISLConformer(nn.Module):
    """Minimal ISL Conformer: Embedding → 1-layer Transformer → TCNN Head."""
    def __init__(
        self,
        input_dim=163,
        num_classes=100,
        d_model=64,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # 1. Multi-stream embedding
        self.embedding = MultiStreamEmbedding(d_model)

        # 2. Single Transformer layer
        self.transformer = TransformerLayer(d_model, num_heads, dropout)

        # 3. TCNN classification head
        self.head = TCNNHead(d_model, num_classes, dropout=dropout)

    def forward(self, x, lengths=None):
        """
        x: (B, T, 163) or (B, T, input_dim)
        lengths: (B,) optional, for masking
        Returns: logits (B, num_classes)
        """
        # Embedding
        x = self.embedding(x)  # (B, T, 64)

        # Transformer
        if lengths is not None:
            lengths = lengths.to(x.device)
            max_len = x.size(1)
            pad_mask = torch.arange(max_len, device=x.device).expand(x.size(0), -1) >= lengths.unsqueeze(1)
        else:
            pad_mask = None
        x = self.transformer(x, key_padding_mask=pad_mask)

        # TCNN head
        logits = self.head(x, lengths)
        return logits

    def get_embeddings(self, x, lengths=None):
        """Get 64-d embeddings before classifier (for analysis)."""
        x = self.embedding(x)
        if lengths is not None:
            max_len = x.size(1)
            pad_mask = torch.arange(max_len, device=x.device).expand(x.size(0), -1) >= lengths.unsqueeze(1)
        else:
            pad_mask = None
        x = self.transformer(x, key_padding_mask=pad_mask)
        return x


# Quick test
if __name__ == "__main__":
    model = MinimalISLConformer(num_classes=228)
    x = torch.randn(2, 60, 163)
    lengths = torch.tensor([60, 45])
    logits = model(x, lengths)
    print(f"Input: {x.shape} -> Logits: {logits.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")