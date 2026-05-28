import torch
import torch.nn as nn
import math

def apply_rotary_pos_emb(x, cos, sin):
    # x: (batch, seq_len, num_heads, head_dim)
    # cos, sin: (seq_len, head_dim)
    d = x.shape[-1]
    x1 = x[..., :d//2]
    x2 = x[..., d//2:]
    x_rot = torch.cat([-x2, x1], dim=-1)
    
    # Expand cos/sin to match x shape: (1, seq_len, 1, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    return x * cos + x_rot * sin

class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len=4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq)
        # Concatenate freqs to match head_dim
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

        # Apply RoPE
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # (B, num_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if key_padding_mask is not None:
            # (B, T) → (B, 1, 1, T) broadcastable to (B, H, T, T)
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(mask, float('-inf'))

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v) # (B, num_heads, T, head_dim)
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
            nn.Dropout(dropout)
        )
        
    def forward(self, x, cos, sin, key_padding_mask=None):
        x = x + self.attn(self.ln1(x), cos, sin, key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.ln2(x))
        return x

class BIT_Transformer(nn.Module):
    def __init__(self, input_dim=512, embed_dim=384, num_layers=7, num_heads=6, dropout=0.2, attn_dropout=0.4, patch_size=4, session_ids=None):
        """
        Neural Encoder for BCI with RoPE.
        Args:
            input_dim: Number of neural channels (512).
            embed_dim: Transformer embedding dimension (384).
            num_layers: Number of Transformer layers (7).
            num_heads: Number of attention heads (6).
            dropout: Dropout rate (0.2).
            attn_dropout: Attention dropout rate (0.4).
            patch_size: Number of bins to group (5 bins of 20ms = 100ms).
            session_ids: List of unique session identifiers for subject-specific read-in layers.
        """
        super().__init__()
        self.patch_size = patch_size
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # Subject-Specific Read-in Layers
        self.read_in = nn.ModuleDict()
        if session_ids:
            for sid in session_ids:
                # Sanitize key: PyTorch doesn't allow dots in module names
                safe_sid = str(sid).replace('.', '_')
                self.read_in[safe_sid] = nn.Linear(input_dim, input_dim)
            # Always include a default fallback layer
            self.read_in["default"] = nn.Linear(input_dim, input_dim)
        else:
            self.read_in["default"] = nn.Identity()

        # LayerNorm Sandwich Patch Embedding
        self.patch_ln1 = nn.LayerNorm(input_dim * patch_size)
        self.patch_embedding = nn.Linear(input_dim * patch_size, embed_dim)
        self.patch_ln2 = nn.LayerNorm(embed_dim)

        # Transformer Encoder with RoPE
        self.rope = RoPE(embed_dim // num_heads)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout, attn_dropout=attn_dropout) 
            for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Learnable mask token for MAE-style SSL pretraining.
        # Initialized small (std=0.02) so it doesn't dominate untrained encoder outputs.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, x, session_id=None, mask_patches=None, neural_lengths=None):
        """
        Args:
            x: (Batch, Time, Channels) - Z-score normalized & Gaussian smoothed
            session_id: String or list of strings indicating the session for each batch item.
        """
        batch_size, time_steps, channels = x.shape

        # 1. Subject-Specific Read-in
        if session_id is not None:
            # Handle list of session IDs or a single one
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
                        new_x.append(layer(x[i:i+1]))
                    x = torch.cat(new_x, dim=0)
            else:
                sid = str(session_id).replace('.', '_')
                layer = self.read_in[sid] if sid in self.read_in else self.read_in["default"]
                x = layer(x)
        else:
            layer = self.read_in["default"]
            x = layer(x)

        # 2. Time Patching (20ms -> 100ms)
        pad_len = (self.patch_size - (time_steps % self.patch_size)) % self.patch_size
        if pad_len > 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_len))

        batch_size, new_time_steps, _ = x.shape
        x = x.view(batch_size, new_time_steps // self.patch_size, self.patch_size * channels)

        # 3. Patch Embedding (LN -> Linear -> LN)
        x = self.patch_ln1(x)
        x = self.patch_embedding(x)
        x = self.patch_ln2(x)

        # Apply learnable mask token to masked patch positions (SSL only).
        # mask_patches: bool tensor (B, T_patch). True means "mask this patch".
        if mask_patches is not None:
            mask_token = self.mask_token.expand_as(x).to(x.dtype)
            x = torch.where(mask_patches.unsqueeze(-1), mask_token, x)

        # Build key padding mask in patch space from neural_lengths (raw bin lengths).
        key_padding_mask = None
        if neural_lengths is not None:
            patched_lengths = (neural_lengths + self.patch_size - 1) // self.patch_size
            T_patch = x.size(1)
            arange = torch.arange(T_patch, device=x.device).unsqueeze(0)  # (1, T_patch)
            key_padding_mask = arange >= patched_lengths.unsqueeze(1)      # (B, T_patch), True = pad

        # 4. Transformer Forward Pass with RoPE
        seq_len = x.size(1)
        cos, sin = self.rope(seq_len)

        for layer in self.layers:
            x = layer(x, cos, sin, key_padding_mask=key_padding_mask)
            
        x = self.layer_norm(x)

        return x
