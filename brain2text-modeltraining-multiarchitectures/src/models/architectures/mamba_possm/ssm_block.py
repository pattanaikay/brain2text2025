import torch
import torch.nn as nn

from .drop_path import DropPath

try:
    from mamba_ssm import Mamba as _Mamba
    _MAMBA_AVAILABLE = True
except ImportError:
    _MAMBA_AVAILABLE = False


class _GRUSSMWrapper(nn.Module):
    """Pure-PyTorch GRU fallback when mamba-ssm is unavailable (e.g. CPU smoke tests)."""

    def __init__(self, d_model=384, **kwargs):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, batch_first=True)

    def forward(self, x):
        out, _ = self.gru(x)
        return out


def _build_ssm(backbone: str, dim: int, d_state: int, expand: int):
    if backbone == "mamba":
        if _MAMBA_AVAILABLE:
            return _Mamba(
                d_model=dim, d_state=d_state, d_conv=4, expand=expand,
                dt_min=0.01, dt_max=0.5,  # POSSM §3.2 soft-window bias
            )
        # Silent fallback — warn once
        import warnings
        warnings.warn(
            "mamba-ssm not installed; falling back to GRU backbone. "
            "Install with: pip install mamba-ssm causal-conv1d",
            stacklevel=3,
        )
        return _GRUSSMWrapper(dim)
    if backbone == "gru":
        return _GRUSSMWrapper(dim)
    # s4: use GRU as placeholder; swap with S4 when available
    return _GRUSSMWrapper(dim)


class SSMBlock(nn.Module):
    """SSM block with residual FFN (POSSM backbone layer)."""

    def __init__(
        self,
        dim: int = 384,
        expand: int = 2,
        d_state: int = 16,
        drop_path: float = 0.1,
        dropout: float = 0.1,
        backbone: str = "mamba",
    ):
        super().__init__()
        # Import here to avoid circular dependency on RMSNorm
        from src.models.architectures.conformer.rmsnorm import RMSNorm
        self.norm = RMSNorm(dim)
        self.ssm = _build_ssm(backbone, dim, d_state, expand)
        self.drop_path = DropPath(drop_path)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )
        from src.models.architectures.conformer.rmsnorm import RMSNorm as _RMS
        self.ffn_norm = _RMS(dim)

    def forward(self, x):
        x = x + self.drop_path(self.ssm(self.norm(x)))
        x = x + self.drop_path(self.ffn(self.ffn_norm(x)))
        return x
