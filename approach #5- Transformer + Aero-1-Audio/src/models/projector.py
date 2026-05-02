import torch
import torch.nn as nn

class MLPProjector(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=1024, output_dim=1536):
        """
        Maps neural embeddings to LLM latent space.
        Args:
            input_dim: Neural encoder output dimension (384).
            hidden_dim: Hidden dimension for MLP.
            output_dim: LLM input embedding dimension (1536 for Aero-1-Audio).
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.mlp(x)
