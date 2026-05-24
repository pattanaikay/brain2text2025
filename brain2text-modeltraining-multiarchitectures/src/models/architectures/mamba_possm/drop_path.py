import torch
import torch.nn as nn


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        keep = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(1 - self.drop_prob)
        return x * keep / (1 - self.drop_prob)
