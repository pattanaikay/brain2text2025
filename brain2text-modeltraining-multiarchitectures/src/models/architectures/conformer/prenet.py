import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleDilatedConv(nn.Module):
    """Three parallel dilated Conv1d branches fused to dim channels."""

    def __init__(self, dim=512):
        super().__init__()
        # Same-padding: pad = (kernel-1)*dilation//2
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=5, padding=2, dilation=1)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=5, padding=4, dilation=2)
        self.conv4 = nn.Conv1d(dim, dim, kernel_size=5, padding=8, dilation=4)
        self.fuse = nn.Conv1d(3 * dim, dim, kernel_size=1)

    def forward(self, x):
        # x: (B, T, dim)
        xt = x.transpose(1, 2)  # (B, dim, T)
        o1 = F.gelu(self.conv1(xt))
        o2 = F.gelu(self.conv2(xt))
        o4 = F.gelu(self.conv4(xt))
        out = self.fuse(torch.cat([o1, o2, o4], dim=1))  # (B, dim, T)
        return out.transpose(1, 2)  # (B, T, dim)


class JitterCorrectionPrenet(nn.Module):
    """Multi-scale dilated conv + BiGRU to correct neural spike-timing jitter.

    iPhoneme §3.1: local temporal alignment before patching improves CTC convergence.
    """

    def __init__(self, dim=512, gru_hidden=256):
        super().__init__()
        self.dilated = MultiScaleDilatedConv(dim)
        self.ln = nn.LayerNorm(dim)
        self.gru = nn.GRU(
            dim, gru_hidden, num_layers=1, batch_first=True, bidirectional=True
        )
        self.proj = nn.Linear(2 * gru_hidden, dim)

    def forward(self, x):
        # x: (B, T, dim)
        residual = x
        x = self.dilated(x)
        x = self.ln(x)
        x, _ = self.gru(x)
        x = self.proj(x)
        return x + residual
