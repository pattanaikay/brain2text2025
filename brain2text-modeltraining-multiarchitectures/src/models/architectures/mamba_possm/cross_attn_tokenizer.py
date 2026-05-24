import torch
import torch.nn as nn


class IndividualSpikeTokenizer(nn.Module):
    """Replace linear patch embedding with cross-attention tokenisation.

    Each patch of patch_size bins is encoded by cross-attending n_queries learned
    query tokens against the per-bin KV features, then sum-pooling the query
    dimension to produce one token per patch.

    POSSM §3.1: cross-attention over per-spike features improves spike alignment.
    """

    def __init__(
        self,
        in_dim: int = 512,
        patch_size: int = 4,
        n_queries: int = 64,
        embed_dim: int = 384,
        n_heads: int = 6,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.n_queries = n_queries
        self.queries = nn.Parameter(torch.randn(n_queries, embed_dim) * 0.02)
        self.kv_proj = nn.Linear(in_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.out_ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, T_bins_padded, in_dim)  — already padded to multiple of patch_size
        B, T_bins, _ = x.shape
        T_patch = T_bins // self.patch_size

        # Reshape to (B*T_patch, patch_size, in_dim)
        x = x.view(B * T_patch, self.patch_size, -1)

        # KV projection: (B*T_patch, patch_size, embed_dim)
        kv = self.kv_proj(x)

        # Queries: (B*T_patch, n_queries, embed_dim)
        q = self.queries.unsqueeze(0).expand(B * T_patch, -1, -1)

        # Cross-attention: Q attends to K,V
        out, _ = self.attn(q, kv, kv, need_weights=False)  # (B*T_patch, n_queries, embed_dim)

        # Sum-pool over query tokens → one token per patch
        out = out.sum(dim=1)                               # (B*T_patch, embed_dim)
        out = self.out_ln(out)
        return out.view(B, T_patch, -1)                    # (B, T_patch, embed_dim)
