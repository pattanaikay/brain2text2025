"""
core/bit_encoder.py
-------------------
Verbatim copy of brain2text-modeltraining/src/models/encoder.py (the BIT_Transformer
that produced best_model_per.pth). Kept standalone so this study has zero dependency
on the sibling repos. Only torch is required.
"""
import torch
import torch.nn as nn
import math


def apply_rotary_pos_emb(x, cos, sin):
    # x: (batch, seq_len, num_heads, head_dim)
    d = x.shape[-1]
    x1 = x[..., :d // 2]
    x2 = x[..., d // 2:]
    x_rot = torch.cat([-x2, x1], dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    return x * cos + x_rot * sin


class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len=4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def forward(self, seq_len):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


class RoPEAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout=0.4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x, cos, sin, key_padding_mask=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(mask, float('-inf'))
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.2, attn_dropout=0.4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = RoPEAttention(embed_dim, num_heads, attn_dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, cos, sin, key_padding_mask=None):
        x = x + self.attn(self.ln1(x), cos, sin, key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class BIT_Transformer(nn.Module):
    def __init__(self, input_dim=512, embed_dim=384, num_layers=7, num_heads=6,
                 dropout=0.2, attn_dropout=0.4, patch_size=4, session_ids=None):
        super().__init__()
        self.patch_size = patch_size
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.read_in = nn.ModuleDict()
        if session_ids:
            for sid in session_ids:
                safe_sid = str(sid).replace('.', '_')
                self.read_in[safe_sid] = nn.Linear(input_dim, input_dim)
            self.read_in["default"] = nn.Linear(input_dim, input_dim)
        else:
            self.read_in["default"] = nn.Identity()

        self.patch_ln1 = nn.LayerNorm(input_dim * patch_size)
        self.patch_embedding = nn.Linear(input_dim * patch_size, embed_dim)
        self.patch_ln2 = nn.LayerNorm(embed_dim)

        self.rope = RoPE(embed_dim // num_heads)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout, attn_dropout=attn_dropout)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, x, session_id=None, mask_patches=None, neural_lengths=None):
        batch_size, time_steps, channels = x.shape

        if session_id is not None:
            if isinstance(session_id, (list, tuple)):
                if len(set(session_id)) == 1:
                    sid = str(session_id[0]).replace('.', '_')
                    layer = self.read_in[sid] if sid in self.read_in else self.read_in["default"]
                    x = layer(x)
                else:
                    new_x = []
                    for i in range(batch_size):
                        sid = str(session_id[i]).replace('.', '_')
                        layer = self.read_in[sid] if sid in self.read_in else self.read_in["default"]
                        new_x.append(layer(x[i:i + 1]))
                    x = torch.cat(new_x, dim=0)
            else:
                sid = str(session_id).replace('.', '_')
                layer = self.read_in[sid] if sid in self.read_in else self.read_in["default"]
                x = layer(x)
        else:
            x = self.read_in["default"](x)

        pad_len = (self.patch_size - (time_steps % self.patch_size)) % self.patch_size
        if pad_len > 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_len))

        batch_size, new_time_steps, _ = x.shape
        x = x.view(batch_size, new_time_steps // self.patch_size, self.patch_size * channels)

        x = self.patch_ln1(x)
        x = self.patch_embedding(x)
        x = self.patch_ln2(x)

        if mask_patches is not None:
            mask_token = self.mask_token.expand_as(x).to(x.dtype)
            x = torch.where(mask_patches.unsqueeze(-1), mask_token, x)

        key_padding_mask = None
        if neural_lengths is not None:
            patched_lengths = (neural_lengths + self.patch_size - 1) // self.patch_size
            T_patch = x.size(1)
            arange = torch.arange(T_patch, device=x.device).unsqueeze(0)
            key_padding_mask = arange >= patched_lengths.unsqueeze(1)

        seq_len = x.size(1)
        cos, sin = self.rope(seq_len)
        for layer in self.layers:
            x = layer(x, cos, sin, key_padding_mask=key_padding_mask)
        x = self.layer_norm(x)
        return x
