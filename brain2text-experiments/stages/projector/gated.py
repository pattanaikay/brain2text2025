"""
stages/projector/gated.py
-------------------------
Gated MLP projector (Track E1b).
Multiplicative interaction: h = relu(fc1(x)) * sigmoid(gate(x))
"""

from __future__ import annotations
import torch
import torch.nn as nn


class GatedMLPProjector(nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 1024, output_dim: int = 1536):
        super().__init__()
        self.fc1  = nn.Linear(input_dim, hidden_dim)
        self.gate = nn.Linear(input_dim, hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, output_dim)
        self.ln   = nn.LayerNorm(output_dim)

    def forward(self, x):
        h = torch.relu(self.fc1(x)) * torch.sigmoid(self.gate(x))
        return self.ln(self.fc2(h))


def build(spec: dict, prev_shape: tuple) -> tuple:
    """
    spec keys:
        input_dim  : int = 384
        hidden_dim : int = 1024
        output_dim : int = 1536
    """
    input_dim  = spec.get("input_dim",  prev_shape[-1] if prev_shape else 384)
    hidden_dim = spec.get("hidden_dim", 1024)
    output_dim = spec.get("output_dim", 1536)
    T = prev_shape[0] if prev_shape and len(prev_shape) >= 2 else None
    projector = GatedMLPProjector(input_dim, hidden_dim, output_dim)
    out_shape  = (T, output_dim) if T else (output_dim,)
    return projector, out_shape
