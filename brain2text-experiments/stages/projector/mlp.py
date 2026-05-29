"""
stages/projector/mlp.py
-----------------------
3-layer MLP projector — the current BIT baseline projector (Track E baseline).
"""

from __future__ import annotations
import torch.nn as nn
from docks.bit_dock import MLPProjector


def build(spec: dict, prev_shape: tuple) -> tuple:
    """
    spec keys:
        input_dim  : int = 384
        hidden_dim : int = 1024
        output_dim : int = 1536    # must match decoder llm_dim
    """
    input_dim  = spec.get("input_dim",  prev_shape[-1] if prev_shape else 384)
    hidden_dim = spec.get("hidden_dim", 1024)
    output_dim = spec.get("output_dim", 1536)

    projector = MLPProjector(
        input_dim  = input_dim,
        hidden_dim = hidden_dim,
        output_dim = output_dim,
    )

    # prev_shape = (T_patch, input_dim) → output = (T_patch, output_dim)
    T = prev_shape[0] if prev_shape and len(prev_shape) >= 1 else None
    out_shape = (T, output_dim) if T else (output_dim,)
    return projector, out_shape
