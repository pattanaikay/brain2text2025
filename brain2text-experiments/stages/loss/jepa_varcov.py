"""
stages/loss/jepa_varcov.py
--------------------------
Track F anti-collapse guard for JEPA (VICReg-style variance + covariance).

THE LOAD-BEARING RISK (named in the ADHD deepen pass): a JEPA predictor can
"win" by driving target representations to a constant — predictor-loss drops,
the smoke test goes green, and the self-supervised signal is hollow. This loss
keeps per-dimension variance above a hinge and decorrelates dimensions, so a
decrease in the JEPA loss has to be real structure, not collapse.

Composed via compose([...]) alongside the JEPA pretrain loss. Reads the
context embeddings the encoder exposes in outputs["jepa_embeds"].

Reference: Bardes, Ponce, LeCun — "VICReg" (2022).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPAVarCovLoss(nn.Module):
    def __init__(self, var_weight: float = 1.0, cov_weight: float = 0.04,
                 var_target: float = 1.0, eps: float = 1e-4):
        super().__init__()
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.var_target = var_target
        self.eps = eps

    def __call__(self, batch, stack, outputs: dict) -> dict:
        z = outputs.get("jepa_embeds")           # (B, T, E) context embeddings
        if z is None:
            return {"loss_jepa_varcov": torch.tensor(0.0)}

        z = z.reshape(-1, z.shape[-1])           # (N, E)
        z = z - z.mean(0, keepdim=True)

        # Variance hinge: push each dim's std toward var_target.
        std = torch.sqrt(z.var(0) + self.eps)
        var_loss = F.relu(self.var_target - std).mean()

        # Covariance: penalise off-diagonal correlations.
        n, d = z.shape
        cov = (z.T @ z) / max(n - 1, 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        cov_loss = off_diag.pow(2).sum() / d

        loss = self.var_weight * var_loss + self.cov_weight * cov_loss
        return {"loss_jepa_varcov": loss}


def build(spec: dict, prev_shape) -> tuple:
    """
    spec keys:
        var_weight : float = 1.0
        cov_weight : float = 0.04
        var_target : float = 1.0
    """
    loss = JEPAVarCovLoss(
        var_weight=spec.get("var_weight", 1.0),
        cov_weight=spec.get("cov_weight", 0.04),
        var_target=spec.get("var_target", 1.0),
    )
    return loss, None
