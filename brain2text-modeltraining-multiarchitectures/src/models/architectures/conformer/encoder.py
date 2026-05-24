import torch
import torch.nn as nn
import torch.nn.functional as F

from .prenet import JitterCorrectionPrenet
from .blocks import ConformerBlock


class ConformerEncoder(nn.Module):
    """ConformerXL neural encoder (iPhoneme §3).

    Universal read-in (no per-session ModuleDict). Relies on internal
    regularisation (macaron FFN, conv module) instead of session-specific
    calibration layers.
    """

    embed_dim: int = 384
    input_dim: int = 512

    def __init__(
        self,
        input_dim: int = 512,
        embed_dim: int = 384,
        num_layers: int = 12,
        num_heads: int = 6,
        patch_size: int = 4,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        **kwargs,  # absorbs session_ids and other unused kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self.read_in = nn.Linear(input_dim, input_dim)
        self.prenet = JitterCorrectionPrenet(input_dim)

        # LN sandwich patch embedding (same as BIT baseline)
        self.patch_ln1 = nn.LayerNorm(input_dim * patch_size)
        self.patch_embedding = nn.Linear(input_dim * patch_size, embed_dim)
        self.patch_ln2 = nn.LayerNorm(embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.layers = nn.ModuleList([
            ConformerBlock(embed_dim, num_heads, dropout, attn_dropout)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, session_id=None, mask_patches=None, neural_lengths=None):
        # session_id intentionally ignored — universal read-in
        B, T, C = x.shape

        x = self.read_in(x)
        x = self.prenet(x)

        # Pad T to multiple of patch_size
        pad_len = (self.patch_size - T % self.patch_size) % self.patch_size
        if pad_len:
            x = F.pad(x, (0, 0, 0, pad_len))

        B, T_pad, _ = x.shape
        x = x.view(B, T_pad // self.patch_size, self.patch_size * C)

        # Patch embedding
        x = self.patch_ln1(x)
        x = self.patch_embedding(x)
        x = self.patch_ln2(x)

        # SSL mask token substitution
        if mask_patches is not None:
            mask_tok = self.mask_token.expand_as(x).to(x.dtype)
            x = torch.where(mask_patches.unsqueeze(-1), mask_tok, x)

        # Build key padding mask in patch space from raw bin lengths
        key_padding_mask = None
        if neural_lengths is not None:
            patched_lengths = (neural_lengths + self.patch_size - 1) // self.patch_size
            T_patch = x.size(1)
            arange = torch.arange(T_patch, device=x.device).unsqueeze(0)
            key_padding_mask = arange >= patched_lengths.unsqueeze(1)  # True = pad

        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        return self.layer_norm(x)  # (B, T_patch, 384)
