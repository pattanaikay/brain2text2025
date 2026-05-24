import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoder import RoPE, RoPEAttention
from .ssmoe_block import SSMoEBlock


class MoETransformerBlock(nn.Module):
    """BIT TransformerBlock with MLP replaced by SSMoEBlock.

    Returns (x, aux_loss) so caller can accumulate load-balance loss.
    """

    def __init__(self, embed_dim=384, num_heads=6, dropout=0.2, attn_dropout=0.4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = RoPEAttention(embed_dim, num_heads, attn_dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = SSMoEBlock(embed_dim)

    def forward(self, x, cos, sin, key_padding_mask=None):
        x = x + self.attn(self.ln1(x), cos, sin, key_padding_mask=key_padding_mask)
        moe_out, aux_loss = self.mlp(self.ln2(x))
        x = x + moe_out
        return x, aux_loss


class MoEEncoder(nn.Module):
    """MoE-augmented BIT encoder (EEGMoE / NeuroMoE §3).

    Replaces per-session ModuleDict with universal read-in and swaps FFN
    inside each TransformerBlock with SSMoEBlock for dynamic domain routing.

    Exposes last_aux_loss (sum of per-block load-balance losses) for
    use in train_e2e.py / train_moe.py.

    session_id accepted and ignored.
    """

    embed_dim: int = 384
    input_dim: int = 512

    def __init__(
        self,
        input_dim: int = 512,
        embed_dim: int = 384,
        num_layers: int = 7,
        num_heads: int = 6,
        dropout: float = 0.2,
        attn_dropout: float = 0.4,
        patch_size: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.last_aux_loss = torch.tensor(0.0)

        self.read_in = nn.Linear(input_dim, input_dim)

        self.patch_ln1 = nn.LayerNorm(input_dim * patch_size)
        self.patch_embedding = nn.Linear(input_dim * patch_size, embed_dim)
        self.patch_ln2 = nn.LayerNorm(embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.rope = RoPE(embed_dim // num_heads)

        self.layers = nn.ModuleList([
            MoETransformerBlock(embed_dim, num_heads, dropout, attn_dropout)
            for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, session_id=None, mask_patches=None, neural_lengths=None):
        # session_id ignored
        B, T, C = x.shape

        x = self.read_in(x)

        pad_len = (self.patch_size - T % self.patch_size) % self.patch_size
        if pad_len:
            x = F.pad(x, (0, 0, 0, pad_len))

        B, T_pad, _ = x.shape
        x = x.view(B, T_pad // self.patch_size, self.patch_size * C)

        x = self.patch_ln1(x)
        x = self.patch_embedding(x)
        x = self.patch_ln2(x)

        if mask_patches is not None:
            mask_tok = self.mask_token.expand_as(x).to(x.dtype)
            x = torch.where(mask_patches.unsqueeze(-1), mask_tok, x)

        key_padding_mask = None
        if neural_lengths is not None:
            patched_lengths = (neural_lengths + self.patch_size - 1) // self.patch_size
            T_patch = x.size(1)
            arange = torch.arange(T_patch, device=x.device).unsqueeze(0)
            key_padding_mask = arange >= patched_lengths.unsqueeze(1)

        cos, sin = self.rope(x.size(1))

        aux_total = torch.tensor(0.0, device=x.device)
        for layer in self.layers:
            x, aux = layer(x, cos, sin, key_padding_mask=key_padding_mask)
            aux_total = aux_total + aux

        self.last_aux_loss = aux_total
        return self.layer_norm(x)  # (B, T_patch, 384)
