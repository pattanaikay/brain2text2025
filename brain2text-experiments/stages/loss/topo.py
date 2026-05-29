"""
stages/loss/topo.py
-------------------
Topographic cortical-map regularization on FFN weight matrices (Track D3).

Forces FFN weights to become spatially organized like cortical maps —
nearby neurons become functionally similar after Gaussian blur.

d_hidden=1024 in BIT's FFN → sqrt(1024) = 32 ✓ (perfect square)
d_hidden=384  in Conformer  → pads to 400 = 20² (slight inefficiency)

Variants:
    D3a: weight=0.0   (baseline)
    D3b: weight=0.001 (light)
    D3c: weight=0.01  (standard)
    D3d: weight=0.1   (likely too strong)
"""

from __future__ import annotations
import torch
import torch.nn as nn
from docks.multiarch_dock import TopoLoss, collect_ffn_first_linears


class TopoLossStage(nn.Module):
    def __init__(self, weight: float = 0.01, sigma: float = 1.0):
        super().__init__()
        self.weight = weight
        self.sigma  = sigma
        self._topo_fn = None   # initialised lazily after stack is built

    def attach(self, encoder: nn.Module):
        """Call once after Stack.from_spec to register FFN target modules."""
        if TopoLoss is None or collect_ffn_first_linears is None:
            raise ImportError(
                "TopoLoss not available. Check multiarch_dock imports."
            )
        targets = collect_ffn_first_linears(encoder)
        self._topo_fn = TopoLoss(targets, sigma=self.sigma)
        return self

    def __call__(self, batch, stack, outputs: dict) -> dict:
        if self.weight == 0.0 or self._topo_fn is None:
            return {"loss_topo": torch.tensor(0.0)}
        loss = self._topo_fn() * self.weight
        return {"loss_topo": loss}


def build(spec: dict, prev_shape) -> tuple:
    """
    spec keys:
        weight : float = 0.01
        sigma  : float = 1.0

    After build, caller must call:
        topo_stage.attach(stack.encoder)
    run.py does this automatically.
    """
    loss = TopoLossStage(
        weight = spec.get("weight", 0.01),
        sigma  = spec.get("sigma",  1.0),
    )
    return loss, None
