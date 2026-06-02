"""
stages/loss/episodic_consistency.py
-----------------------------------
Track H (now LIVE): ZenBrain episodic-consistency objective.

The objective is "recall-the-past-latent": pull the current encoder latent toward
the episodic memory's cross-attention retrieval of a consistent past
representation of the same/similar trial. This is the consistency regulariser
that makes the memory readout a stable anchor across drifting days.

The memory stage (stages/memory/episodic_buffer.py) emits `memory_query`
(current latent) and `memory_retrieved` (recalled latent) on its `last_read`,
which run.py merges into `outputs`. The loss is MSE(query, retrieved.detach()) —
the retrieval is the target, so gradient flows into the encoder that produced the
query (not into the read head, which CE already trains).

Was a backprop-safe zero (`* 0.0`) during the skeleton phase; the `* 0.0` has
been removed and H1/H2 are live.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EpisodicConsistencyLoss(nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def __call__(self, batch, stack, outputs: dict) -> dict:
        # Intended inputs once the memory stage is real:
        query     = outputs.get("memory_query")      # (B, T, E) current latent
        retrieved = outputs.get("memory_retrieved")  # (B, T, E) recalled latent

        if query is None or retrieved is None:
            # Memory stage not present in this spec — emit a backprop-safe zero so
            # compose() can still sum without breaking the spine (e.g. for specs
            # that register this loss but omit the `memory:` stage).
            return {"loss_episodic": torch.zeros(1, requires_grad=True).squeeze()}

        # LIVE: pull the current latent toward the recalled (detached) memory.
        loss = F.mse_loss(query, retrieved.detach()) * self.weight
        return {"loss_episodic": loss}


def build(spec: dict, prev_shape) -> tuple:
    """
    spec keys:
        weight : float = 1.0   (currently inert — loss is forced to 0.0)
    """
    return EpisodicConsistencyLoss(weight=spec.get("weight", 1.0)), None
