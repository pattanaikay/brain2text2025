import torch
import torch.nn as nn


class LowLevelRecurrent(nn.Module):
    """L-module: fast 20 ms recurrence gated by H-level context.

    HRM §3.2: L iterates at the raw-bin timescale until convergence.
    """

    def __init__(self, dim=512, hidden=384):
        super().__init__()
        # GRUCell input = x_t (dim) concat h_context (hidden)
        self.cell = nn.GRUCell(dim + hidden, hidden)
        self.proj_h_in = nn.Linear(hidden, hidden)  # H-state → L input gate

    def forward(self, x_t, h_state, h_context):
        # x_t: (B, dim); h_state: (B, hidden); h_context: (B, hidden)
        ctx = self.proj_h_in(h_context)
        return self.cell(torch.cat([x_t, ctx], dim=-1), h_state)


class HighLevelRecurrent(nn.Module):
    """H-module: slow 100 ms recurrence over patch summaries.

    HRM §3.3: one H-step per patch_size L-steps.
    """

    def __init__(self, hidden=384):
        super().__init__()
        self.cell = nn.GRUCell(hidden, hidden)

    def forward(self, h_summary, h_state):
        # h_summary = last L-state at end of a patch
        return self.cell(h_summary, h_state)
