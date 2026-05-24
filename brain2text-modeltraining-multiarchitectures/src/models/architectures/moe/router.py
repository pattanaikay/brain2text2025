import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKRouter(nn.Module):
    """Switch-Transformer-style Top-K router with load-balance auxiliary loss."""

    def __init__(self, dim: int, n_experts: int, top_k: int = 2):
        super().__init__()
        self.gate = nn.Linear(dim, n_experts, bias=False)
        self.top_k = top_k
        self.n_experts = n_experts

    def forward(self, x):
        # x: (B, T, dim)
        logits = self.gate(x)                         # (B, T, E)
        probs = F.softmax(logits, dim=-1)
        top_p, top_i = probs.topk(self.top_k, dim=-1) # (B, T, K)
        top_p = top_p / top_p.sum(dim=-1, keepdim=True)  # renormalize selected

        # Switch-Transformer load-balance aux loss
        importance = probs.sum(dim=(0, 1))             # (E,)
        load = (probs > 0).float().sum(dim=(0, 1))     # (E,)
        importance = importance / (importance.sum() + 1e-8)
        load = load / (load.sum() + 1e-8)
        aux_loss = (importance * load).sum() * self.n_experts

        return top_p, top_i, aux_loss
