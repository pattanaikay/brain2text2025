import torch
import torch.nn as nn

from .router import TopKRouter


class SpecificExpert(nn.Module):
    def __init__(self, dim=384, hidden=1024, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SharedExpert(SpecificExpert):
    pass  # identical architecture, always active (soft-weighted)


class SSMoEBlock(nn.Module):
    """Sparse-Shared Mixture-of-Experts FFN replacing the TransformerBlock MLP.

    EEGMoE §3: n_specific experts via top-k routing + n_shared experts applied
    uniformly. Outputs (result, aux_load_balance_loss).
    """

    def __init__(
        self,
        dim: int = 384,
        n_specific: int = 6,
        n_shared: int = 2,
        top_k: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_specific = n_specific
        self.n_shared = n_shared
        self.top_k = top_k
        self.specific = nn.ModuleList([SpecificExpert(dim, dropout=dropout) for _ in range(n_specific)])
        self.shared = nn.ModuleList([SharedExpert(dim, dropout=dropout) for _ in range(n_shared)])
        self.router = TopKRouter(dim, n_experts=n_specific, top_k=top_k)

    def forward(self, x):
        # x: (B, T, dim)
        routed, aux_loss = self._dispatch_specific(x)
        shared_out = sum(e(x) for e in self.shared) / self.n_shared
        return routed + shared_out, aux_loss

    def _dispatch_specific(self, x):
        gates, indices, aux_loss = self.router(x)  # (B,T,K), (B,T,K), scalar
        B, T, D = x.shape
        out = torch.zeros_like(x)
        for k in range(self.top_k):
            for e_idx in range(self.n_specific):
                mask = (indices[..., k] == e_idx)  # (B, T) bool
                if not mask.any():
                    continue
                expert_in = x[mask]                # (N_sel, D)
                expert_out = self.specific[e_idx](expert_in)
                weight = gates[..., k][mask].unsqueeze(-1)
                out[mask] = out[mask] + weight * expert_out
        return out, aux_loss
