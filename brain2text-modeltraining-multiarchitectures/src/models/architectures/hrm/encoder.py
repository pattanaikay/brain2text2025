import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import LowLevelRecurrent, HighLevelRecurrent
from .deq import fixed_point_solver, OneStepGrad


class HRMEncoder(nn.Module):
    """Hierarchical Reasoning Model encoder (HRM §3).

    Dual-timescale recurrence: L-module (20 ms bins) + H-module (100 ms patches).
    Fixed-point DEQ solver + 1-step gradient gives O(1) memory w.r.t. seq length.

    session_id accepted and ignored.
    """

    embed_dim: int = 384
    input_dim: int = 512

    def __init__(
        self,
        patch_size: int = 4,
        l_hidden: int = 384,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = 512
        self.embed_dim = 384
        self.patch_size = patch_size
        self.l_hidden = l_hidden

        self.read_in = nn.Linear(512, 512)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 384))
        nn.init.normal_(self.mask_token, std=0.02)

        self.L = LowLevelRecurrent(dim=512, hidden=l_hidden)
        self.H = HighLevelRecurrent(hidden=384)

        self.out_proj = nn.Linear(l_hidden, 384)
        self.out_ln = nn.LayerNorm(384)

    def forward(self, x, session_id=None, mask_patches=None, neural_lengths=None):
        # session_id ignored
        B, T_bins, C = x.shape

        x = self.read_in(x)

        pad_len = (self.patch_size - T_bins % self.patch_size) % self.patch_size
        if pad_len:
            x = F.pad(x, (0, 0, 0, pad_len))
        T_padded = x.size(1)
        T_patch = T_padded // self.patch_size

        h_h = torch.zeros(B, 384, device=x.device, dtype=x.dtype)
        h_l = torch.zeros(B, self.l_hidden, device=x.device, dtype=x.dtype)

        patch_outputs = []
        for p in range(T_patch):
            patch = x[:, p * self.patch_size:(p + 1) * self.patch_size, :]  # (B, ps, 512)

            # Capture loop variables by value for the closure
            _patch = patch
            _h_h = h_h.detach()
            _L = self.L

            def step(h, _p=_patch, _hh=_h_h, _l=_L):
                h_iter = h
                for t in range(self.patch_size):
                    h_iter = _l(_p[:, t, :], h_iter, _hh)
                return h_iter

            h_star = fixed_point_solver(step, h_l, tol=1e-3, max_iter=10)
            # 1-step gradient via custom autograd (keeps O(1) memory per patch)
            h_star = OneStepGrad.apply(h_star, self.L, patch, h_h.detach(), self.patch_size)

            h_h = self.H(h_star, h_h)
            patch_outputs.append(self.out_proj(h_star))

            # Detach across patches — O(1) memory (no BPTT between patches)
            h_l = h_star.detach()

        out = torch.stack(patch_outputs, dim=1)  # (B, T_patch, l_hidden) → projected to 384

        # Apply mask token AFTER building outputs (SSL: replace masked positions)
        if mask_patches is not None:
            mask_tok = self.mask_token.expand_as(out).to(out.dtype)
            out = torch.where(mask_patches.unsqueeze(-1), mask_tok, out)

        return self.out_ln(out)  # (B, T_patch, 384)
