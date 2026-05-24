import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopoLoss(nn.Module):
    """Topographic organization loss (TopoNets §3).

    Forces neurons in target FFN layers to become spatially similar to their
    grid neighbours — mimicking cortical topographic maps.
    d_hidden must be a perfect square; pass target linears whose d_hidden satisfies this.
    """

    def __init__(self, target_modules: list, sigma: float = 1.0):
        super().__init__()
        self.targets = target_modules
        self.sigma = sigma
        kernel = self._build_gaussian_kernel(sigma)
        self.register_buffer("kernel", kernel)

    @staticmethod
    def _build_gaussian_kernel(sigma: float) -> torch.Tensor:
        # 5×5 Gaussian kernel (7 if sigma > 2)
        ksize = 5 if sigma <= 2 else 7
        half = ksize // 2
        coords = torch.arange(ksize, dtype=torch.float32) - half
        g1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        g2d = g1d.unsqueeze(0) * g1d.unsqueeze(1)   # (ksize, ksize)
        g2d = g2d / g2d.sum()
        return g2d.unsqueeze(0).unsqueeze(0)          # (1, 1, ksize, ksize)

    def forward(self) -> torch.Tensor:
        total = torch.tensor(0.0, device=self.kernel.device)
        for lin in self.targets:
            W = lin.weight                            # (d_hidden, d_in)
            d_hidden, d_in = W.shape
            grid = int(round(math.sqrt(d_hidden)))
            if grid * grid != d_hidden:
                # Pad to next perfect square
                next_sq = grid + 1
                pad_needed = next_sq * next_sq - d_hidden
                W = F.pad(W, (0, 0, 0, pad_needed))
                grid = next_sq
            M = W.t().reshape(d_in, grid, grid)       # (d_in, H, W)
            # depthwise conv: kernel expanded to (d_in, 1, ksize, ksize)
            k = self.kernel.expand(d_in, 1, *self.kernel.shape[-2:]).to(W.dtype)
            blurred = F.conv2d(
                M.unsqueeze(0), k,
                padding=k.shape[-1] // 2, groups=d_in,
            ).squeeze(0)                              # (d_in, H, W)
            sim = F.cosine_similarity(
                M.reshape(d_in, -1), blurred.reshape(d_in, -1), dim=-1
            )
            total = total - sim.mean()
        return total / max(1, len(self.targets))
