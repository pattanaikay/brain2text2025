"""
stages/encoder/mamba.py
-----------------------
Stage adapter for MambaPOSSMEncoder (Track B3).

ssm_backbone = "gru"   → no extra install needed (always works)
ssm_backbone = "mamba" → requires: pip install mamba-ssm causal-conv1d
"""

from __future__ import annotations
import math
from docks.multiarch_dock import MambaPOSSMEncoder


def build(spec: dict, prev_shape: tuple) -> tuple:
    """
    spec keys:
        patch_size   : int   = 4
        n_layers     : int   = 7
        ssm_backbone : str   = "gru"   # "gru" | "mamba"
        dropout      : float = 0.1
        drop_path    : float = 0.1
        embed_dim    : int   = 384
    """
    patch_size   = spec.get("patch_size",   4)
    n_layers     = spec.get("n_layers",     7)
    ssm_backbone = spec.get("ssm_backbone", "gru")
    dropout      = spec.get("dropout",      0.1)
    drop_path    = spec.get("drop_path",    0.1)
    embed_dim    = spec.get("embed_dim",    384)

    if ssm_backbone == "mamba":
        try:
            import mamba_ssm  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "ssm_backbone='mamba' requires mamba-ssm. "
                "Install: pip install mamba-ssm causal-conv1d\n"
                "Or use ssm_backbone='gru' to run without it."
            )

    encoder = MambaPOSSMEncoder(
        patch_size   = patch_size,
        n_layers     = n_layers,
        ssm_backbone = ssm_backbone,
        dropout      = dropout,
        drop_path    = drop_path,
    )

    T_bins  = prev_shape[0] if prev_shape else 240
    T_patch = math.ceil(T_bins / patch_size)
    return encoder, (T_patch, embed_dim)
