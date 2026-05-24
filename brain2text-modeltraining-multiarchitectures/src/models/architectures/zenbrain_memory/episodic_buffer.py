import torch
import torch.nn as nn


class EpisodicBuffer(nn.Module):
    """FIFO buffer of M token vectors representing ~5 min of neural context.

    The buffer is a registered non-trainable buffer (not a Parameter), so it
    travels with the model on .to(device) but is never updated by the optimizer.
    """

    def __init__(self, M: int = 3000, dim: int = 384):
        super().__init__()
        self.M = M
        self.register_buffer("buf", torch.zeros(M, dim))
        self.register_buffer("filled", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def push(self, tokens: torch.Tensor):
        # tokens: (N, dim)
        N = tokens.shape[0]
        if N == 0:
            return
        if N >= self.M:
            self.buf.copy_(tokens[-self.M:])
            self.filled.fill_(self.M)
        else:
            remaining = self.buf[N:].clone()
            self.buf.copy_(torch.cat([remaining, tokens], dim=0))
            self.filled.copy_(torch.clamp(self.filled + N, max=self.M))

    def get(self):
        """Returns (buf_tokens, key_padding_mask) where mask True = pad slot."""
        valid = torch.arange(self.M, device=self.buf.device) >= (self.M - self.filled)
        return self.buf, ~valid   # ~valid: True = unfilled (pad)
