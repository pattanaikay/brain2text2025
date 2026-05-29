"""
stages/encoder/moe.py
---------------------
Stage adapter for MoEEncoder (Track B4).

IMPORTANT: MoEEncoder exposes last_aux_loss after each forward pass.
The training loop MUST add it to the total loss to prevent expert collapse:

    if hasattr(stack.encoder, 'last_aux_loss'):
        total_loss = total_loss + aux_weight * stack.encoder.last_aux_loss
"""

from __future__ import annotations
import math
from docks.multiarch_dock import MoEEncoder


def build(spec: dict, prev_shape: tuple) -> tuple:
    """
    spec keys:
        input_dim    : int   = 512
        embed_dim    : int   = 384
        num_layers   : int   = 7
        num_heads    : int   = 6
        patch_size   : int   = 4
        n_specific   : int   = 6    # number of routed experts
        n_shared     : int   = 2    # always-active shared experts
        top_k        : int   = 2    # experts activated per token
        dropout      : float = 0.2
        session_ids  : list  = None
        aux_loss_weight: float = 0.01  # stored in spec; training loop reads it
    """
    input_dim  = spec.get("input_dim",  512)
    embed_dim  = spec.get("embed_dim",  384)
    num_layers = spec.get("num_layers", 7)
    num_heads  = spec.get("num_heads",  6)
    patch_size = spec.get("patch_size", 4)
    n_specific = spec.get("n_specific", 6)
    n_shared   = spec.get("n_shared",   2)
    top_k      = spec.get("top_k",      2)
    dropout    = spec.get("dropout",    0.2)
    session_ids = spec.get("session_ids", None)

    encoder = MoEEncoder(
        input_dim   = input_dim,
        embed_dim   = embed_dim,
        num_layers  = num_layers,
        num_heads   = num_heads,
        patch_size  = patch_size,
        n_specific  = n_specific,
        n_shared    = n_shared,
        top_k       = top_k,
        dropout     = dropout,
        session_ids = session_ids,
    )

    T_bins  = prev_shape[0] if prev_shape else 240
    T_patch = math.ceil(T_bins / patch_size)
    return encoder, (T_patch, embed_dim)
