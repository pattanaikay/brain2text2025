import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class SSMoEBlock(nn.Module):
    """
    Specific and Shared Mixture-of-Experts (SSMoE) block.
    Integrates concepts from EEGMoE:
    - Specific Expert Group: Top-K routing for domain-specific features.
    - Shared Expert Group: Soft routing for domain-shared features.
    """
    def __init__(self, embed_dim, hidden_dim, num_specific=6, num_shared=2, top_k=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_specific = num_specific
        self.num_shared = num_shared
        self.top_k = top_k

        # Specific Expert Group (Top-K Routing)
        self.specific_experts = nn.ModuleList([Expert(embed_dim, hidden_dim) for _ in range(num_specific)])
        self.specific_router = nn.Linear(embed_dim, num_specific)

        # Shared Expert Group (Soft Routing)
        self.shared_experts = nn.ModuleList([Expert(embed_dim, hidden_dim) for _ in range(num_shared)])
        self.shared_router = nn.Linear(embed_dim, num_shared)

    def forward(self, x):
        # x shape: (Batch * SeqLen, Dim)
        orig_shape = x.shape
        x_flat = x.view(-1, self.embed_dim)
        
        # 1. Specific MoE (Top-K)
        # gx = We * x
        spec_logits = self.specific_router(x_flat)
        # pi(x) = exp(gxi) / sum(exp(gxj))
        spec_probs = F.softmax(spec_logits, dim=-1)
        
        top_k_probs, top_k_indices = torch.topk(spec_probs, self.top_k, dim=-1)
        # Re-normalize Top-K probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        spec_out = torch.zeros_like(x_flat)
        # We iterate through experts for simplicity in this implementation, 
        # but in large-scale MoE, specialized kernels are used.
        for i, expert in enumerate(self.specific_experts):
            mask = (top_k_indices == i).any(dim=-1)
            if mask.any():
                # Get the specific probability for this expert for each token in the mask
                expert_mask = (top_k_indices == i)
                # Weights are the probabilities where the expert was selected
                weights = (top_k_probs * expert_mask.float()).sum(dim=-1, keepdim=True)
                spec_out[mask] += weights[mask] * expert(x_flat[mask])

        # 2. Shared MoE (Soft Routing)
        shared_logits = self.shared_router(x_flat)
        shared_probs = F.softmax(shared_logits, dim=-1)
        
        shared_out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.shared_experts):
            shared_out += shared_probs[:, i:i+1] * expert(x_flat)

        # 3. Combine: SSMoE(x) = SpecMoE(x) + ShareMoE(x)
        output = spec_out + shared_out
        
        # 4. Load Balancing Loss (Auxiliary)
        # Laux = E * sum(hi * Di)
        E = self.num_specific
        # hi: fraction of tokens allocated to expert i
        h = torch.zeros(E, device=x.device)
        for i in range(E):
            h[i] = (top_k_indices == i).any(dim=-1).float().mean()
        # Di: fraction of router probability
        D = spec_probs.mean(dim=0)
        aux_loss = E * torch.sum(h * D)

        return output.view(orig_shape), aux_loss

class RegionalExpert(nn.Module):
    """
    Functionally Guided Regional Expert (from BrainStack).
    Captures localized neural dynamics for specific cortical regions.
    """
    def __init__(self, num_channels, embed_dim):
        super().__init__()
        # CNet Architecture: Temporal Conv -> Spatial Conv -> Separable Conv
        self.temporal_conv = nn.Conv1d(num_channels, num_channels, kernel_size=31, padding=15, groups=num_channels)
        self.spatial_conv = nn.Conv1d(num_channels, embed_dim, kernel_size=1)
        self.bn = nn.BatchNorm1d(embed_dim)
        self.elu = nn.ELU()
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (Batch, Time, Channels) -> (Batch, Channels, Time)
        x = x.transpose(1, 2)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.bn(x)
        x = self.elu(x)
        return self.pool(x).squeeze(-1) # (Batch, Dim)

class BrainStackRouter(nn.Module):
    """
    Adaptive Expert Routing Gate from BrainStack.
    Aggregates regional features into a meta-representation.
    """
    def __init__(self, embed_dim, num_regions):
        super().__init__()
        self.scoring = nn.Linear(embed_dim, 1)
        
    def forward(self, region_features):
        # region_features: (Batch, NumRegions, Dim)
        # alpha_i = exp(h(Fi)) / sum(exp(h(Fj)))
        logits = self.scoring(region_features) # (Batch, NumRegions, 1)
        weights = F.softmax(logits, dim=1)
        
        # Fmeta = sum(alpha_i * Fi)
        f_meta = torch.sum(weights * region_features, dim=1)
        return f_meta, weights
