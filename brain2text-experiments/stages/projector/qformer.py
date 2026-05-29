"""
stages/projector/qformer.py
---------------------------
Q-Former cross-attention projector (Track E2).

N_QUERIES learned tokens attend to the full neural sequence via cross-attention,
then are projected to LLM embedding dim. Output sequence length is always
N_QUERIES regardless of neural input length.

This is the highest-impact projector change per EXPERIMENT_DESIGN.md E2.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class QFormerProjector(nn.Module):
    """
    Args:
        input_dim  : encoder output dim (384)
        output_dim : LLM embedding dim  (1536)
        n_queries  : number of learned query tokens (16, 32, or 64)
        n_heads    : attention heads (must divide input_dim)
    """
    def __init__(
        self,
        input_dim:  int = 384,
        output_dim: int = 1536,
        n_queries:  int = 32,
        n_heads:    int = 6,
    ):
        super().__init__()
        self.n_queries = n_queries
        # Learned query tokens
        self.queries    = nn.Parameter(torch.randn(n_queries, input_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(input_dim, n_heads, batch_first=True)
        self.ln_q       = nn.LayerNorm(input_dim)
        self.ln_kv      = nn.LayerNorm(input_dim)
        self.proj       = nn.Linear(input_dim, output_dim)
        self.ln_out     = nn.LayerNorm(output_dim)

    def forward(self, neural_tokens, key_padding_mask=None):
        """
        neural_tokens: (B, T_patch, input_dim)
        key_padding_mask: (B, T_patch) bool — True = padded position (ignored)
        Returns: (B, N_QUERIES, output_dim)
        """
        B = neural_tokens.size(0)
        q  = self.queries.unsqueeze(0).expand(B, -1, -1)   # (B, N_Q, input_dim)
        kv = self.ln_kv(neural_tokens)
        out, _ = self.cross_attn(
            self.ln_q(q), kv, kv,
            key_padding_mask = key_padding_mask,
            need_weights     = False,
        )
        return self.ln_out(self.proj(out))   # (B, N_Q, output_dim)


def build(spec: dict, prev_shape: tuple) -> tuple:
    """
    spec keys:
        input_dim  : int = 384
        output_dim : int = 1536
        n_queries  : int = 32    # 16 | 32 | 64
        n_heads    : int = 6
    """
    input_dim  = spec.get("input_dim",  prev_shape[-1] if prev_shape else 384)
    output_dim = spec.get("output_dim", 1536)
    n_queries  = spec.get("n_queries",  32)
    n_heads    = spec.get("n_heads",    6)

    assert input_dim % n_heads == 0, (
        f"QFormer: input_dim={input_dim} must be divisible by n_heads={n_heads}"
    )

    projector = QFormerProjector(input_dim, output_dim, n_queries, n_heads)
    # Output sequence is always n_queries long (key Q-Former property)
    out_shape  = (n_queries, output_dim)
    return projector, out_shape
