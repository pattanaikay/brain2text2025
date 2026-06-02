"""
stages/projector/dietcorp_recal.py
----------------------------------
Track G: the reusable novelty from transformers_with_dietcorp, re-expressed as
a native projector stage on our own (B,T,512) data.

DietCorp's two load-bearing tricks:
  1. Corrupted-input regularization  — Gaussian/dropout corruption of the input
     during training acts as a strong regulariser on tiny BCI datasets.
  2. Per-day (per-session) affine recalibration — a learned per-day scale+shift
     absorbs day-to-day electrode drift, so the shared transformer sees a
     recalibrated input. THIS is the index that must actually route.

This is real, runnable code (not a stub) — it's the seam the ADHD pass said to
ship first, because the vendored upstream loop is only needed to generate the
oracle metrics. forward(x, day_index=...) proves the day index routes.

build(spec, prev_shape) -> (DietCorpRecalProjector, (T, output_dim))
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DietCorpRecalProjector(nn.Module):
    def __init__(self, input_dim: int = 512, output_dim: int = 384,
                 n_days: int = 8, corrupt_std: float = 0.1,
                 corrupt_drop: float = 0.1):
        super().__init__()
        self.n_days       = n_days
        self.corrupt_std  = corrupt_std
        self.corrupt_drop = corrupt_drop

        # Per-day affine recalibration: learned scale + shift per session.
        self.day_scale = nn.Embedding(n_days, input_dim)
        self.day_shift = nn.Embedding(n_days, input_dim)
        nn.init.ones_(self.day_scale.weight)
        nn.init.zeros_(self.day_shift.weight)

        self.dropout = nn.Dropout(corrupt_drop)
        self.proj    = nn.Linear(input_dim, output_dim)

    def forward(self, x, day_index=None, **_):
        """x: (B, T, input_dim). day_index: (B,) long tensor of session ids."""
        B = x.size(0)
        if day_index is None:
            day_index = torch.zeros(B, dtype=torch.long, device=x.device)
        day_index = day_index.clamp(0, self.n_days - 1)

        # Per-day affine recalibration (broadcast over time).
        scale = self.day_scale(day_index).unsqueeze(1)     # (B, 1, C)
        shift = self.day_shift(day_index).unsqueeze(1)
        x = x * scale + shift

        # Corrupted-input regularization (train only).
        if self.training:
            x = x + torch.randn_like(x) * self.corrupt_std
            x = self.dropout(x)

        return self.proj(x)                                # (B, T, output_dim)


def build(spec: dict, prev_shape: tuple) -> tuple:
    """
    spec keys:
        input_dim    : int   = 512
        output_dim   : int   = 384
        n_days       : int   = 8
        corrupt_std  : float = 0.1
        corrupt_drop : float = 0.1
    """
    input_dim  = spec.get("input_dim",  prev_shape[-1] if prev_shape else 512)
    output_dim = spec.get("output_dim", 384)

    proj = DietCorpRecalProjector(
        input_dim    = input_dim,
        output_dim   = output_dim,
        n_days       = spec.get("n_days", 8),
        corrupt_std  = spec.get("corrupt_std", 0.1),
        corrupt_drop = spec.get("corrupt_drop", 0.1),
    )
    T = prev_shape[0] if prev_shape and len(prev_shape) >= 1 else None
    out_shape = (T, output_dim) if T else (output_dim,)
    return proj, out_shape
