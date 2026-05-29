"""
stages/loss/contrastive.py
--------------------------
Modality alignment contrastive loss (Track D2).

Matches the ModalityAlignmentLoss from bit_e2e.py (symmetric NT-Xent
with learnable temperature). Weight is separately controllable so we
can ablate it without touching architecture.

Variants:
    D2a: weight=0.0   (ablation — does contrastive help?)
    D2b: weight=0.1
    D2c: weight=1.0   (current baseline)
    D2d: weight=2.0
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from docks.bit_dock import ModalityAlignmentLoss


class ContrastiveLossStage(nn.Module):
    def __init__(self, weight: float = 1.0, temperature: float = 0.07):
        super().__init__()
        self.weight   = weight
        self.loss_fn  = ModalityAlignmentLoss(temperature=temperature)

    def __call__(self, batch, stack, outputs: dict) -> dict:
        if self.weight == 0.0:
            device = next(self.loss_fn.parameters()).device
            return {"loss_contrastive": torch.tensor(0.0, device=device)}

        projected_embeds = outputs.get("projected_embeds")  # (B, T, llm_dim)
        neural_lengths   = batch.get("neural_lengths")

        # text embeddings from decoder
        text_embeds = outputs.get("text_embeds")             # (B, T_text, llm_dim)

        if projected_embeds is None or text_embeds is None:
            device = next(self.loss_fn.parameters()).device
            return {"loss_contrastive": torch.tensor(0.0, device=device)}

        B    = projected_embeds.size(0)
        patch_size = stack.encoder.patch_size if hasattr(stack.encoder, "patch_size") else 4

        # Mean-pool neural embeddings (masked)
        if neural_lengths is not None:
            plen = (neural_lengths + patch_size - 1) // patch_size
            neural_pooled = []
            for i in range(B):
                curr_len = max(1, plen[i].item())
                neural_pooled.append(projected_embeds[i, :curr_len].mean(0))
            neural_pooled = torch.stack(neural_pooled)
        else:
            neural_pooled = projected_embeds.mean(1)

        # Mean-pool text embeddings (masked)
        text_mask   = outputs.get("text_mask")               # (B, T_text, 1) or None
        if text_mask is not None:
            text_pooled = (text_embeds * text_mask).sum(1) / text_mask.sum(1).clamp(min=1)
        else:
            text_pooled = text_embeds.mean(1)

        loss = self.loss_fn(neural_pooled, text_pooled) * self.weight
        return {"loss_contrastive": loss}


def build(spec: dict, prev_shape) -> tuple:
    """
    spec keys:
        weight      : float = 1.0
        temperature : float = 0.07
    """
    loss = ContrastiveLossStage(
        weight      = spec.get("weight",      1.0),
        temperature = spec.get("temperature", 0.07),
    )
    return loss, None
