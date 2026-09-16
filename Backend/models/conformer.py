import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("isl_conformer")

# ============================================================================
# 1. UTILS & MASKS
# ============================================================================
def lengths_to_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Returns a (B, T) bool mask where True == PADDING."""
    batch_size = lengths.size(0)
    arange = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
    return arange >= lengths.unsqueeze(1)

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0, self.dim1 = dim0, dim1
    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)

# ============================================================================
# 2. MULTI-STREAM EMBEDDING
# ============================================================================
class MultiStreamEmbedding(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        # Hands (126 + 2 flags = 128)
        self.hand_proj = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, d_model // 2))
        # Arms (12 + 1 flag = 13)
        self.arm_proj = nn.Sequential(nn.LayerNorm(13), nn.Linear(13, d_model // 4))
        # Face (21 + 1 flag = 22)
        self.face_proj = nn.Sequential(nn.LayerNorm(22), nn.Linear(22, d_model // 4))

    def forward(self, x):
        hands = x[:, :, 0:126]
        arms  = x[:, :, 126:138]
        face  = x[:, :, 138:159]
        flags = x[:, :, 159:163]

        hand_stream = torch.cat([hands, flags[:, :, 0:2]], dim=-1)
        arm_stream  = torch.cat([arms, flags[:, :, 2:3]], dim=-1)
        face_stream = torch.cat([face, flags[:, :, 3:4]], dim=-1)

        h_emb = self.hand_proj(hand_stream)
        a_emb = self.arm_proj(arm_stream)
        f_emb = self.face_proj(face_stream)

        return torch.cat([h_emb, a_emb, f_emb], dim=-1)  # (B, T, 256)

# ============================================================================
# 3. MBCONV (LOCAL SPATIAL)
# ============================================================================
class SqueezeExcitation1D(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // reduction, 1),
            nn.SiLU(),
            nn.Conv1d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.se(x)

class EfficientNet1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=2):
        super().__init__()
        mid_channels = in_channels * expansion
        self.expand = nn.Conv1d(in_channels, mid_channels, kernel_size=1)
        self.bn_expand = nn.BatchNorm1d(mid_channels)

        self.depthwise = nn.Conv1d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels)
        self.bn_depthwise = nn.BatchNorm1d(mid_channels)

        self.se = SqueezeExcitation1D(mid_channels)
        self.project = nn.Conv1d(mid_channels, out_channels, kernel_size=1)
        self.bn_project = nn.BatchNorm1d(out_channels)
        self.silu = nn.SiLU()

        self.needs_proj = in_channels != out_channels
        self.res_proj = nn.Conv1d(in_channels, out_channels, 1) if self.needs_proj else nn.Identity()

    def forward(self, x, pad_mask_ct=None):
        if pad_mask_ct is not None: x = x.masked_fill(pad_mask_ct, 0.0)
        res = self.res_proj(x)

        x = self.silu(self.bn_expand(self.expand(x)))
        if pad_mask_ct is not None: x = x.masked_fill(pad_mask_ct, 0.0)

        x = self.silu(self.bn_depthwise(self.depthwise(x)))
        x = self.se(x)
        x = self.bn_project(self.project(x))

        x = res + x
        if pad_mask_ct is not None: x = x.masked_fill(pad_mask_ct, 0.0)
        return x

# ============================================================================
# 4. CUSTOM CONFORMER BLOCK
# ============================================================================
class FeedForwardModule(nn.Module):
    def __init__(self, d_model, expansion=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class ConformerConvModule(nn.Module):
    def __init__(self, d_model, kernel_size=15, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, d_model * 2, 1)  # *2 for GLU
        self.dw = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.silu = nn.SiLU()
        self.pw2 = nn.Conv1d(d_model, d_model, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask_ct=None):
        # x is (B, T, D) -> needs (B, D, T) for Conv1d
        x = self.ln(x).transpose(1, 2)
        if pad_mask_ct is not None: x = x.masked_fill(pad_mask_ct, 0.0)

        x = self.pw1(x)
        x = F.glu(x, dim=1)

        x = self.dw(x)
        x = self.bn(x)
        x = self.silu(x)

        x = self.pw2(x)
        x = self.drop(x)

        x = x.transpose(1, 2)  # back to (B, T, D)
        if pad_mask_ct is not None:
            x = x.masked_fill(pad_mask_ct.transpose(1, 2), 0.0)
        return x

class ConformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, conv_kernel_size=15, dropout=0.1):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, dropout=dropout)
        self.attn_ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.conv = ConformerConvModule(d_model, conv_kernel_size, dropout)
        self.ffn2 = FeedForwardModule(d_model, dropout=dropout)
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, x, pad_mask_bt, pad_mask_ct):
        x = x + 0.5 * self.ffn1(x)

        res = x
        x = self.attn_ln(x)
        x, _ = self.attn(x, x, x, key_padding_mask=pad_mask_bt)
        x = res + x

        x = x + self.conv(x, pad_mask_ct)
        x = x + 0.5 * self.ffn2(x)

        return self.final_ln(x)

# ============================================================================
# 5. THE MAIN ARCHITECTURE
# ============================================================================
class ISL_Conformer(nn.Module):
    def __init__(
        self,
        input_dim=163,
        num_classes=100,
        d_model=256,
        num_heads=4,
        num_layers=4,
        conv_kernel_size=15,
        dropout=0.1
    ):
        super().__init__()
        self.input_stage = MultiStreamEmbedding(d_model=d_model)

        self.transpose_to_ct = Transpose(1, 2)
        self.mbconv = EfficientNet1DBlock(in_channels=d_model, out_channels=d_model)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.transpose_to_tc = Transpose(1, 2)

        self.conformer_layers = nn.ModuleList([
            ConformerBlock(d_model, num_heads, conv_kernel_size, dropout)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Linear(d_model, num_classes + 1)  # +1 for CTC blank

    def forward(self, x, lengths):
        B, T, _ = x.shape
        pad_mask_bt = lengths_to_padding_mask(lengths, T)
        pad_mask_ct = pad_mask_bt.unsqueeze(1)

        x = self.input_stage(x)

        x = self.transpose_to_ct(x)
        x = self.mbconv(x, pad_mask_ct)
        x = self.pool(x)

        lengths_pooled = lengths // 2
        T_pooled = x.size(-1)
        pad_mask_bt_pooled = lengths_to_padding_mask(lengths_pooled, T_pooled)
        pad_mask_ct_pooled = pad_mask_bt_pooled.unsqueeze(1)

        mbconv_features = self.transpose_to_tc(x)  # (B, T_pooled, D)

        x_conf = mbconv_features
        for layer in self.conformer_layers:
            x_conf = layer(x_conf, pad_mask_bt_pooled, pad_mask_ct_pooled)

        x_fused = x_conf + mbconv_features

        out = self.classifier(x_fused)
        out = out.masked_fill(pad_mask_bt_pooled.unsqueeze(-1), 0.0)

        return F.log_softmax(out, dim=-1), lengths_pooled

if __name__ == "__main__":
    # Smoke test
    _model = ISL_Conformer(num_classes=100)
    _dummy_x = torch.randn(2, 60, 163)
    _dummy_lengths = torch.tensor([60, 45])
    _log_probs, _pooled_lengths = _model(_dummy_x, _dummy_lengths)
    print("✅ Model smoke test passed.")
    print(f"Input Shape: {_dummy_x.shape} -> Output Shape: {_log_probs.shape}, Pooled Lengths: {_pooled_lengths.tolist()}")
