import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoder import RoPE
from .episodic_buffer import EpisodicBuffer
from .dual_memory_attn import DualMemoryBlock


class ZenBrainEncoder(nn.Module):
    """Zero-calibration memory-augmented BIT encoder (ZenBrain / DietCORP).

    Pretraining: use_memory=False (buffer=None guard skips cross-attn).
    Inference: use_memory=True; buffer fills via update_buffer_from_outputs().

    session_id is accepted and ignored — universal read-in.
    """

    embed_dim: int = 384
    input_dim: int = 512

    def __init__(
        self,
        patch_size: int = 4,
        num_layers: int = 7,
        num_heads: int = 6,
        dropout: float = 0.2,
        buffer_size: int = 3000,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = 512
        self.embed_dim = 384
        self.patch_size = patch_size
        self.use_memory = False  # toggled by train_zerocal.py

        self.read_in = nn.Linear(512, 512)

        # LN sandwich patch embedding
        self.patch_ln1 = nn.LayerNorm(512 * patch_size)
        self.patch_embedding = nn.Linear(512 * patch_size, 384)
        self.patch_ln2 = nn.LayerNorm(384)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 384))
        nn.init.normal_(self.mask_token, std=0.02)

        self.rope = RoPE(384 // num_heads)

        self.layers = nn.ModuleList([
            DualMemoryBlock(384, num_heads, dropout) for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(384)
        self.episodic_buffer = EpisodicBuffer(M=buffer_size, dim=384)

    def forward(self, x, session_id=None, mask_patches=None, neural_lengths=None):
        # session_id ignored
        B, T, C = x.shape

        x = self.read_in(x)

        # Pad to multiple of patch_size
        pad_len = (self.patch_size - T % self.patch_size) % self.patch_size
        if pad_len:
            x = F.pad(x, (0, 0, 0, pad_len))

        B, T_pad, _ = x.shape
        x = x.view(B, T_pad // self.patch_size, self.patch_size * C)

        # Patch embedding
        x = self.patch_ln1(x)
        x = self.patch_embedding(x)
        x = self.patch_ln2(x)

        # SSL mask token
        if mask_patches is not None:
            mask_tok = self.mask_token.expand_as(x).to(x.dtype)
            x = torch.where(mask_patches.unsqueeze(-1), mask_tok, x)

        # Key padding mask
        key_padding_mask = None
        if neural_lengths is not None:
            patched_lengths = (neural_lengths + self.patch_size - 1) // self.patch_size
            T_patch = x.size(1)
            arange = torch.arange(T_patch, device=x.device).unsqueeze(0)
            key_padding_mask = arange >= patched_lengths.unsqueeze(1)

        cos, sin = self.rope(x.size(1))

        # Episodic buffer state (shared across blocks within a forward pass)
        buf, buf_pad = self.episodic_buffer.get() if self.use_memory else (None, None)

        for layer in self.layers:
            x = layer(x, cos, sin, buffer=buf, buf_pad_mask=buf_pad,
                      key_padding_mask=key_padding_mask)

        return self.layer_norm(x)  # (B, T_patch, 384)

    @torch.no_grad()
    def update_buffer_from_outputs(self, neural_tokens, ctc_logits):
        """Push high-confidence patch tokens into the episodic buffer.

        Called from inference scripts after each batch.
        ctc_logits: (B, T_patch, 42)
        """
        conf = F.softmax(ctc_logits.float(), dim=-1).max(dim=-1).values  # (B, T_patch)
        keep = conf > 0.7
        tokens_to_push = neural_tokens[keep]
        if tokens_to_push.numel() > 0:
            self.episodic_buffer.push(tokens_to_push.detach())
