"""
stages/encoder/zenbrain.py
--------------------------
Stage adapter for ZenBrainEncoder (Track B5).

Training:  use_memory=False  (cross-attention to buffer is skipped)
Inference: use_memory=True   (buffer populated from high-confidence trials)

Two-pass eval protocol (from EXPERIMENT_DESIGN.md B5):
    Pass 1: use_memory=False  → cold WER
    Pass 2: use_memory=True, buffer filled from pass 1 → warm WER
    Delta WER = benefit of episodic memory

run.py handles the two-pass automatically when spec.two_pass_eval=True.
"""

from __future__ import annotations
import math
from docks.multiarch_dock import ZenBrainEncoder


def build(spec: dict, prev_shape: tuple) -> tuple:
    """
    spec keys:
        input_dim    : int   = 512
        embed_dim    : int   = 384
        num_layers   : int   = 7
        num_heads    : int   = 6
        patch_size   : int   = 4
        buffer_size  : int   = 3000
        ctc_thresh   : float = 0.7   # min CTC confidence to add to buffer
        dropout      : float = 0.2
        session_ids  : list  = None
        two_pass_eval: bool  = True
    """
    input_dim    = spec.get("input_dim",    512)
    embed_dim    = spec.get("embed_dim",    384)
    num_layers   = spec.get("num_layers",   7)
    num_heads    = spec.get("num_heads",    6)
    patch_size   = spec.get("patch_size",   4)
    buffer_size  = spec.get("buffer_size",  3000)
    dropout      = spec.get("dropout",      0.2)
    session_ids  = spec.get("session_ids",  None)

    encoder = ZenBrainEncoder(
        input_dim   = input_dim,
        embed_dim   = embed_dim,
        num_layers  = num_layers,
        num_heads   = num_heads,
        patch_size  = patch_size,
        buffer_size = buffer_size,
        dropout     = dropout,
        session_ids = session_ids,
    )

    T_bins  = prev_shape[0] if prev_shape else 240
    T_patch = math.ceil(T_bins / patch_size)
    return encoder, (T_patch, embed_dim)
