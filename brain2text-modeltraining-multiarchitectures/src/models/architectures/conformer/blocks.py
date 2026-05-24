import torch
import torch.nn as nn
import torch.nn.functional as F

from .rmsnorm import RMSNorm


class FFNModule(nn.Module):
    def __init__(self, dim=384, expansion=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConvModule(nn.Module):
    """Conformer conv module: pointwise → GLU → depthwise → BN → Swish → pointwise."""

    def __init__(self, dim=384, kernel_size=31, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.pw1 = nn.Conv1d(dim, 2 * dim, kernel_size=1)       # pointwise expand for GLU
        self.dw = nn.Conv1d(
            dim, dim, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=dim,
        )
        # GroupNorm(8, dim) is more stable than BN at batch_size=8-16 (iPhoneme pitfall note)
        self.norm = nn.GroupNorm(8, dim)
        self.pw2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, dim)
        x = self.ln(x)
        xt = x.transpose(1, 2)                # (B, dim, T)
        xt = F.glu(self.pw1(xt), dim=1)       # (B, dim, T) after GLU
        xt = self.dw(xt)
        xt = self.norm(xt)
        xt = F.silu(xt)
        xt = self.pw2(xt)
        return self.drop(xt.transpose(1, 2))  # (B, T, dim)


class ConformerBlock(nn.Module):
    """Macaron-style Conformer block (iPhoneme / ConformerXL).

    Structure: FFN½ → MHSA → Conv → FFN½ → RMSNorm
    """

    def __init__(self, dim=384, num_heads=6, dropout=0.1, attn_dropout=0.1):
        super().__init__()
        self.ln_ffn1 = nn.LayerNorm(dim)
        self.ffn1 = FFNModule(dim, dropout=dropout)

        self.ln_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=attn_dropout, batch_first=True
        )

        self.conv = ConvModule(dim, dropout=dropout)

        self.ln_ffn2 = nn.LayerNorm(dim)
        self.ffn2 = FFNModule(dim, dropout=dropout)

        self.post_norm = RMSNorm(dim)

    def forward(self, x, key_padding_mask=None):
        # Macaron FFN ×0.5
        x = x + 0.5 * self.ffn1(self.ln_ffn1(x))

        # Multi-head self-attention (no RoPE — conv inductive bias replaces it)
        attn_in = self.ln_attn(x)
        attn_out, _ = self.attn(
            attn_in, attn_in, attn_in,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out

        # Convolution module (LN is inside ConvModule)
        x = x + self.conv(x)

        # Macaron FFN ×0.5
        x = x + 0.5 * self.ffn2(self.ln_ffn2(x))

        # Block-end RMSNorm
        return self.post_norm(x)
