"""
stages/projector/deep_mlp.py
----------------------------
5-layer MLP with bottleneck at 2048 (Track E1a).
"""

from __future__ import annotations
import torch.nn as nn


class DeepMLPProjector(nn.Module):
    def __init__(self, input_dim: int = 384, output_dim: int = 1536):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 1024),      nn.ReLU(),
            nn.Linear(1024, 2048),      nn.ReLU(),
            nn.Linear(2048, 1024),      nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return self.mlp(x)


def build(spec: dict, prev_shape: tuple) -> tuple:
    """
    spec keys:
        input_dim  : int = 384
        output_dim : int = 1536
    """
    input_dim  = spec.get("input_dim",  prev_shape[-1] if prev_shape else 384)
    output_dim = spec.get("output_dim", 1536)
    T = prev_shape[0] if prev_shape and len(prev_shape) >= 2 else None
    projector = DeepMLPProjector(input_dim=input_dim, output_dim=output_dim)
    out_shape  = (T, output_dim) if T else (output_dim,)
    return projector, out_shape
