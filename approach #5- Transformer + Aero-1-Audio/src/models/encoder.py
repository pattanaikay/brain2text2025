import torch
import torch.nn as nn

class BIT_Transformer(nn.Module):
    def __init__(self, input_dim=512, embed_dim=384, num_layers=7, num_heads=6, dropout=0.1, patch_size=5, session_ids=None):
        """
        Neural Encoder for BCI.
        Args:
            input_dim: Number of neural channels (512).
            embed_dim: Transformer embedding dimension (384).
            num_layers: Number of Transformer layers (7).
            num_heads: Number of attention heads (6).
            dropout: Dropout rate.
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
                self.read_in[str(sid)] = nn.Linear(input_dim, input_dim)
            # Always include a default fallback layer
            self.read_in["default"] = nn.Linear(input_dim, input_dim)
        else:
            self.read_in["default"] = nn.Identity()

        # Linear Patch Embedding
        self.patch_embedding = nn.Linear(input_dim * patch_size, embed_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=1024,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, session_id=None):
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
                # This is tricky for vectorized batch processing. 
                # For now, assume a single session_id per batch for simplicity or handle loop
                # If it's a list, we might need to process items individually if they differ
                if len(set(session_id)) == 1:
                    sid = str(session_id[0])
                    layer = self.read_in[sid] if sid in self.read_in else self.read_in["default"]
                    x = layer(x)
                else:
                    # Mixed session batch - less efficient
                    new_x = []
                    for i in range(batch_size):
                        sid = str(session_id[i])
                        layer = self.read_in[sid] if sid in self.read_in else self.read_in["default"]
                        new_x.append(layer(x[i:i+1]))
                    x = torch.cat(new_x, dim=0)
            else:
                sid = str(session_id)
                layer = self.read_in[sid] if sid in self.read_in else self.read_in["default"]
                x = layer(x)
        else:
            # Use default if no session_id provided
            layer = self.read_in["default"]
            x = layer(x)

        # 2. Time Patching (20ms -> 100ms)
        # Pad time_steps to be divisible by patch_size
        pad_len = (self.patch_size - (time_steps % self.patch_size)) % self.patch_size
        if pad_len > 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_len))
        
        batch_size, new_time_steps, _ = x.shape
        x = x.view(batch_size, new_time_steps // self.patch_size, self.patch_size * channels)
        
        # 3. Patch Embedding
        x = self.patch_embedding(x)
        
        # 4. Transformer
        x = self.transformer(x)
        x = self.layer_norm(x)
        
        return x
