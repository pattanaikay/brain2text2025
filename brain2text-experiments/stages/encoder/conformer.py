"""
stages/encoder/conformer.py
---------------------------
Stage adapter for ConformerEncoder (Track B1).

Wraps multiarch_dock.ConformerEncoder behind the standard
build(spec, prev_shape) → (Module, out_shape) interface.
"""

from __future__ import annotations
import math
from docks.multiarch_dock import ConformerEncoder


def build(spec: dict, prev_shape: tuple) -> tuple:
    """
    spec keys:
        input_dim    : int   = 512
        embed_dim    : int   = 384
        num_layers   : int   = 12
        patch_size   : int   = 4
        dropout      : float = 0.1
        attn_dropout : float = 0.1

    Note: ConformerEncoder uses universal read-in (no session_ids).
    No pretrained checkpoint exists — always starts from scratch.
    Per EXPERIMENT_DESIGN.md B1: compare against B0_baseline (BIT from scratch).
    """
    input_dim    = spec.get("input_dim",    512)
    embed_dim    = spec.get("embed_dim",    384)
    num_layers   = spec.get("num_layers",   12)
    patch_size   = spec.get("patch_size",   4)
    dropout      = spec.get("dropout",      0.1)
    attn_dropout = spec.get("attn_dropout", 0.1)

    encoder = ConformerEncoder(
        input_dim    = input_dim,
        embed_dim    = embed_dim,
        num_layers   = num_layers,
        patch_size   = patch_size,
        dropout      = dropout,
        attn_dropout = attn_dropout,
    )

    T_bins  = prev_shape[0] if prev_shape else 240
    T_patch = math.ceil(T_bins / patch_size)
    return encoder, (T_patch, embed_dim)
