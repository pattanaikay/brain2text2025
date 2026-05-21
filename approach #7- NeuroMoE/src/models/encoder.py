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
        
    def forward(self, x, cos, sin):
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
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v) # (B, num_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

from .moe import SSMoEBlock, RegionalExpert, BrainStackRouter

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.2, attn_dropout=0.4, num_specific=6, num_shared=2, top_k=2):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = RoPEAttention(embed_dim, num_heads, attn_dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        # Replacing standard MLP with SSMoEBlock from EEGMoE
        self.moe = SSMoEBlock(embed_dim, 1024, num_specific, num_shared, top_k)
        
    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        moe_out, aux_loss = self.moe(self.ln2(x))
        x = x + moe_out
        return x, aux_loss

class BIT_Transformer(nn.Module):
    def __init__(self, input_dim=512, embed_dim=384, num_layers=7, num_heads=6, dropout=0.2, attn_dropout=0.4, patch_size=5, session_ids=None, num_regions=8):
        """
        Neural Encoder with Neuro-MoE (EEGMoE + BrainStack integration).
        """
        super().__init__()
        self.patch_size = patch_size
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_regions = num_regions

        # 1. BrainStack: Regional Experts
        self.channels_per_region = input_dim // num_regions
        self.regional_experts = nn.ModuleList([
            RegionalExpert(self.channels_per_region, embed_dim)
            for _ in range(num_regions)
        ])
        self.brainstack_router = BrainStackRouter(embed_dim, num_regions)

        # 2. Subject-Specific Read-in Layers
        self.read_in = nn.ModuleDict()
        if session_ids:
            for sid in session_ids:
                self.read_in[str(sid)] = nn.Linear(input_dim, input_dim)
            self.read_in["default"] = nn.Linear(input_dim, input_dim)
        else:
            self.read_in["default"] = nn.Identity()

        # 3. Patch Embedding
        self.patch_ln1 = nn.LayerNorm(input_dim * patch_size)
        self.patch_embedding = nn.Linear(input_dim * patch_size, embed_dim)
        self.patch_ln2 = nn.LayerNorm(embed_dim)

        # 4. Transformer Encoder with SSMoE
        self.rope = RoPE(embed_dim // num_heads)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout, attn_dropout=attn_dropout) 
            for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Final Fusion: Global Transformer output + Regional Meta-representation
        self.fusion = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x, session_id=None):
        B, T, C = x.shape

        # --- A. BrainStack: Regional Processing ---
        region_features = []
        for i in range(self.num_regions):
            start = i * self.channels_per_region
            end = (i + 1) * self.channels_per_region
            region_features.append(self.regional_experts[i](x[:, :, start:end]))
        
        f_regional_meta, _ = self.brainstack_router(torch.stack(region_features, dim=1))
        # (Batch, Dim) -> expand to match sequence length later or use as global token
        
        # --- B. Global Transformer (EEGMoE Style) ---
        # 1. Subject-Specific Read-in
        if session_id is not None:
            if isinstance(session_id, (list, tuple)):
                if len(set(session_id)) == 1:
                    sid = str(session_id[0])
                    layer = self.read_in[sid] if sid in self.read_in else self.read_in["default"]
                    x = layer(x)
                else:
                    new_x = []
                    for i in range(B):
                        sid = str(session_id[i])
                        layer = self.read_in[sid] if sid in self.read_in else self.read_in["default"]
                        new_x.append(layer(x[i:i+1]))
                    x = torch.cat(new_x, dim=0)
            else:
                sid = str(session_id)
                layer = self.read_in[sid] if sid in self.read_in else self.read_in["default"]
                x = layer(x)
        else:
            x = self.read_in["default"](x)

        # 2. Time Patching
        pad_len = (self.patch_size - (T % self.patch_size)) % self.patch_size
        if pad_len > 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_len))
        
        B, T_new, _ = x.shape
        x = x.view(B, T_new // self.patch_size, self.patch_size * C)

        # 3. Patch Embedding
        x = self.patch_ln1(x)
        x = self.patch_embedding(x)
        x = self.patch_ln2(x)

        # 4. Transformer Forward Pass with SSMoE
        seq_len = x.size(1)
        cos, sin = self.rope(seq_len)
        
        total_aux_loss = 0
        for layer in self.layers:
            x, aux_loss = layer(x, cos, sin)
            total_aux_loss += aux_loss
            
        x_global = self.layer_norm(x) # (B, S, Dim)

        # --- C. Fusion ---
        # Concatenate regional meta-features to every global token
        f_regional_expanded = f_regional_meta.unsqueeze(1).repeat(1, seq_len, 1)
        combined = torch.cat([x_global, f_regional_expanded], dim=-1)
        x_fused = self.fusion(combined)

        return x_fused, total_aux_loss / len(self.layers)

