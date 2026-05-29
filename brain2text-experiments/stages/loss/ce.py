"""
stages/loss/ce.py
-----------------
Standard cross-entropy loss via the LLM's built-in loss.
This is always the primary loss; other losses are additive.
The decoder forward() already computes this — ce.py is a thin wrapper
that extracts it from outputs for the compose() interface.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class CELoss(nn.Module):
    """
    Reads 'ce_loss' from the decoder's output dict.
    Weight is always 1.0 (CE is the primary signal).
    """
    def __call__(self, batch, stack, outputs: dict) -> dict:
        return {"loss_ce": outputs["ce_loss"]}


def build(spec: dict, prev_shape) -> tuple:
    return CELoss(), None
