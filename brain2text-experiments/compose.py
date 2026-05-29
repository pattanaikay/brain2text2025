"""
compose.py
----------
Hook-based multi-loss composition for the Stack.

Each loss function is a callable that returns a dict of named scalar losses:
    loss_fn(batch, stack, outputs) -> {"loss_ce": Tensor, "loss_ctc": Tensor, ...}

compose([loss_a, loss_b]) returns a single callable that sums all named
losses and returns the combined scalar + the full breakdown dict.

Usage
-----
from compose import compose
from stages.loss.ce          import build as build_ce
from stages.loss.ctc_anneal  import build as build_ctc
from stages.loss.contrastive import build as build_contrastive

loss_fns = compose([
    build_ce({}, prev_shape=None)[0],
    build_ctc({"anneal_epochs": 75}, prev_shape=None)[0],
    build_contrastive({"weight": 1.0}, prev_shape=None)[0],
])

total_loss, breakdown = loss_fns(batch, stack, outputs)
# breakdown = {"loss_ce": ..., "loss_ctc": ..., "loss_contrastive": ...}

Design note
-----------
Loss functions are NOT nn.Modules registered on the Stack; they attach via
forward_hooks so the same trained encoder can be re-evaluated under different
loss combinations without retraining (Track D ablations).
"""

from __future__ import annotations

from typing import Callable, Any
import torch


LossFn = Callable[[dict, Any, dict], dict[str, torch.Tensor]]


class ComposedLoss:
    """
    Holds a list of loss callables.  Calling the instance returns
    (total_loss_tensor, breakdown_dict).
    """
    def __init__(self, fns: list[LossFn]):
        self._fns = fns

    def __call__(
        self,
        batch: dict,
        stack,              # Stack instance
        outputs: dict,      # whatever the decoder forward() returned
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        breakdown: dict[str, torch.Tensor] = {}
        device = None

        for fn in self._fns:
            sub = fn(batch, stack, outputs)
            for k, v in sub.items():
                if k in breakdown:
                    breakdown[k] = breakdown[k] + v
                else:
                    breakdown[k] = v
                if device is None:
                    device = v.device

        if not breakdown:
            return torch.tensor(0.0), {}

        total = sum(breakdown.values())
        return total, breakdown

    def __repr__(self):
        names = [type(fn).__name__ for fn in self._fns]
        return f"ComposedLoss([{', '.join(names)}])"


def compose(fns: list[LossFn]) -> ComposedLoss:
    """
    Combine a list of loss functions.  Order does not matter for the value,
    but does affect the order of breakdown keys (for readability).
    """
    return ComposedLoss(fns)


# ── Convenience: build composed loss from spec list ─────────────────────────
import importlib


def compose_from_spec(loss_specs: list[dict]) -> ComposedLoss:
    """
    Build a ComposedLoss from a list of loss spec dicts, each with a 'variant' key.

    Example:
        compose_from_spec([
            {"variant": "ce"},
            {"variant": "ctc_anneal", "anneal_epochs": 75, "start_weight": 0.3},
            {"variant": "contrastive", "weight": 1.0},
        ])
    """
    fns = []
    for spec in loss_specs:
        spec = dict(spec)
        variant = spec.pop("variant")
        mod = importlib.import_module(f"stages.loss.{variant}")
        loss_module, _ = mod.build(spec, prev_shape=None)
        fns.append(loss_module)
    return compose(fns)
