"""
stages/encoder/bit.py
---------------------
Stage adapter for BIT_Transformer (the baseline encoder).

build(spec, prev_shape) → (BIT_Transformer, out_shape)

out_shape = (T_patch, embed_dim)  where T_patch = ceil(T_bins / patch_size)
"""

from __future__ import annotations
import math
from docks.bit_dock import BIT_Transformer


def build(spec: dict, prev_shape: tuple) -> tuple:
    """
    spec keys (all optional, defaults match EXPERIMENT_DESIGN.md baseline):
        input_dim    : int   = 512
        embed_dim    : int   = 384
        num_layers   : int   = 7
        num_heads    : int   = 6
        dropout      : float = 0.2
        attn_dropout : float = 0.4
        patch_size   : int   = 4
        session_ids  : list  = None
        pretrained_ckpt: str = None  # path to CTC/SSL checkpoint
    """
    input_dim    = spec.get("input_dim",    512)
    embed_dim    = spec.get("embed_dim",    384)
    num_layers   = spec.get("num_layers",   7)
    num_heads    = spec.get("num_heads",    6)
    dropout      = spec.get("dropout",      0.2)
    attn_dropout = spec.get("attn_dropout", 0.4)
    patch_size   = spec.get("patch_size",   4)
    session_ids  = spec.get("session_ids",  None)

    encoder = BIT_Transformer(
        input_dim    = input_dim,
        embed_dim    = embed_dim,
        num_layers   = num_layers,
        num_heads    = num_heads,
        dropout      = dropout,
        attn_dropout = attn_dropout,
        patch_size   = patch_size,
        session_ids  = session_ids,
    )

    # Optional: load CTC/SSL pretrained weights
    pretrained = spec.get("pretrained_ckpt", None)
    if pretrained:
        import torch, os
        if os.path.exists(pretrained):
            ckpt = torch.load(pretrained, map_location="cpu")
            sd = ckpt.get("model_state_dict", ckpt)
            # Strip 'encoder.' prefix if present (CTC training saves as encoder.*)
            enc_sd = {k.replace("encoder.", "", 1): v
                      for k, v in sd.items() if not k.startswith("head.")}
            result = encoder.load_state_dict(enc_sd, strict=False)
            print(f"[BIT stage] Loaded pretrained encoder from {pretrained} "
                  f"(missing={len(result.missing_keys)}, "
                  f"unexpected={len(result.unexpected_keys)})")
        else:
            print(f"[BIT stage] Warning: pretrained_ckpt not found: {pretrained}")

    # Compute output shape  (T_bins from prev_shape, embed_dim from spec)
    T_bins = prev_shape[0] if prev_shape else 240
    T_patch = math.ceil(T_bins / patch_size)
    out_shape = (T_patch, embed_dim)

    return encoder, out_shape
