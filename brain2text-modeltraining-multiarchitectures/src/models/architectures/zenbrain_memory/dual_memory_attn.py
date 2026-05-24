import torch
import torch.nn as nn

# RoPEAttention reused from the baseline encoder (§4.3 ZenBrain spec)
from src.models.encoder import RoPEAttention


class DualMemoryBlock(nn.Module):
    """BIT TransformerBlock augmented with cross-attention over episodic buffer.

    Training: buffer=None → cross-attn skipped (no-op, weights still initialized).
    Inference (zero-cal): buffer filled → cross-attn activates without backprop.
    """

    def __init__(self, dim: int = 384, n_heads: int = 6, dropout: float = 0.2):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.self_attn = RoPEAttention(dim, n_heads, attn_dropout=dropout)

        self.ln_mem = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )

        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, cos, sin, buffer=None, buf_pad_mask=None, key_padding_mask=None):
        # 1. Self-attention over current trial tokens
        x = x + self.self_attn(self.ln1(x), cos, sin, key_padding_mask=key_padding_mask)

        # 2. Cross-attention over episodic buffer (skipped when buffer is None)
        if buffer is not None:
            B = x.size(0)
            # Expand buffer to batch: (B, M, dim)
            buf_kv = buffer.unsqueeze(0).expand(B, -1, -1)
            q = self.ln_mem(x)
            # buf_pad_mask: (M,) bool, True=pad; expand to (B, M)
            bpm = buf_pad_mask.unsqueeze(0).expand(B, -1) if buf_pad_mask is not None else None
            mem_out, _ = self.cross_attn(q, buf_kv, buf_kv, key_padding_mask=bpm, need_weights=False)
            x = x + mem_out

        # 3. FFN
        x = x + self.mlp(self.ln2(x))
        return x
