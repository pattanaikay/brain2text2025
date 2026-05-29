"""
stages/encoder/hrm.py
---------------------
Stage adapter for HRMEncoder (Track B2).
"""

from __future__ import annotations
import math
from docks.multiarch_dock import HRMEncoder


def build(spec: dict, prev_shape: tuple) -> tuple:
    """
    spec keys:
        patch_size : int = 4
        l_hidden   : int = 384   (local GRU hidden = embed_dim)

    HRM has no session read-in and no pretrained checkpoint.
    Profile DEQ iteration time before committing to full run:
        import time; enc = HRMEncoder(); x = torch.randn(8, 240, 512)
        t = time.time(); enc(x); print(f"HRM: {time.time()-t:.2f}s")
    """
    patch_size = spec.get("patch_size", 4)
    l_hidden   = spec.get("l_hidden",   384)

    encoder = HRMEncoder(patch_size=patch_size, l_hidden=l_hidden)

    T_bins  = prev_shape[0] if prev_shape else 240
    T_patch = math.ceil(T_bins / patch_size)
    embed_dim = l_hidden
    return encoder, (T_patch, embed_dim)
