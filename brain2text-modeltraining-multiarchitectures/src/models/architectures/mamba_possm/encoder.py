import torch
import torch.nn as nn
import torch.nn.functional as F

from .cross_attn_tokenizer import IndividualSpikeTokenizer
from .ssm_block import SSMBlock
from src.models.architectures.conformer.rmsnorm import RMSNorm


class MambaPOSSMEncoder(nn.Module):
    """Hybrid SSM encoder (POSSM §3).

    Cross-attention spike tokenizer replaces linear patch embedding.
    7 SSM blocks with optional Mamba/S4/GRU backbone (default: mamba with GRU
    fallback if mamba-ssm is unavailable).

    session_id is accepted and ignored — universal read-in.
    """

    embed_dim: int = 384
    input_dim: int = 512

    def __init__(
        self,
        patch_size: int = 4,
        n_layers: int = 7,
        ssm_backbone: str = "mamba",
        dropout: float = 0.1,
        drop_path: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = 512
        self.embed_dim = 384
        self.patch_size = patch_size

        self.read_in = nn.Linear(512, 512)

        self.tokenizer = IndividualSpikeTokenizer(
            in_dim=512, patch_size=patch_size, embed_dim=384
        )

        # 3-layer input compression (POSSM §3.4 regularisation)
        self.compress = nn.Sequential(
            nn.Linear(384, 384), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(384, 384), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(384, 384), nn.GELU(), nn.Dropout(0.4),
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 384))
        nn.init.normal_(self.mask_token, std=0.02)

        self.layers = nn.ModuleList([
            SSMBlock(dim=384, drop_path=drop_path, dropout=dropout, backbone=ssm_backbone)
            for _ in range(n_layers)
        ])

        self.out_proj = nn.Linear(384, 384)
        self.out_ln = nn.LayerNorm(384)

    def forward(self, x, session_id=None, mask_patches=None, neural_lengths=None):
        # session_id ignored
        B, T, C = x.shape

        x = self.read_in(x)

        # Pad T to multiple of patch_size
        pad_len = (self.patch_size - T % self.patch_size) % self.patch_size
        if pad_len:
            x = F.pad(x, (0, 0, 0, pad_len))

        # Cross-attention tokenizer → (B, T_patch, 384)
        x = self.tokenizer(x)

        # SSL mask token substitution
        if mask_patches is not None:
            mask_tok = self.mask_token.expand_as(x).to(x.dtype)
            x = torch.where(mask_patches.unsqueeze(-1), mask_tok, x)

        x = self.compress(x)

        # Build float mask for SSM (no key_padding_mask in Mamba/GRU)
        float_mask = None
        if neural_lengths is not None:
            patched_lengths = (neural_lengths + self.patch_size - 1) // self.patch_size
            T_patch = x.size(1)
            arange = torch.arange(T_patch, device=x.device).unsqueeze(0)  # (1, T_patch)
            # True for valid positions
            valid = arange < patched_lengths.unsqueeze(1)
            float_mask = valid.float().unsqueeze(-1)  # (B, T_patch, 1)

        for layer in self.layers:
            x = layer(x)
            # Zero out padded positions to prevent state pollution (§Arch-2 pitfalls)
            if float_mask is not None:
                x = x * float_mask

        return self.out_ln(self.out_proj(x))  # (B, T_patch, 384)
