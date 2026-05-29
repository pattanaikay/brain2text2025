"""
stages/loss/label_smooth.py
----------------------------
Label-smoothed cross-entropy (Track D4).

Replaces the LLM's built-in CE with a custom F.cross_entropy call that
adds ε-smoothing. Prevents overconfidence on the small BCI dataset (~1000 samples).

This stage replaces ce.py — use it instead, not in addition to, ce.py.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothCELoss(nn.Module):
    def __init__(self, eps: float = 0.1):
        super().__init__()
        self.eps = eps

    def __call__(self, batch, stack, outputs: dict) -> dict:
        # We need raw logits — the decoder must expose them in outputs["logits"]
        # run.py sets decoder.return_logits=True when this loss stage is active.
        logits      = outputs.get("logits")       # (B, seq_len, vocab_size)
        full_labels = outputs.get("full_labels")  # (B, seq_len) with -100 masked

        if logits is None or full_labels is None:
            # Fall back to LLM-computed CE if logits not available
            return {"loss_ce": outputs.get("ce_loss", torch.tensor(0.0))}

        B, seq_len, vocab = logits.shape
        shift_logits = logits[..., :-1, :].contiguous().view(-1, vocab)
        shift_labels = full_labels[..., 1:].contiguous().view(-1)

        loss = F.cross_entropy(
            shift_logits, shift_labels,
            ignore_index  = -100,
            label_smoothing = self.eps,
        )
        return {"loss_ce": loss}


def build(spec: dict, prev_shape) -> tuple:
    """
    spec keys:
        eps : float = 0.1   (label smoothing epsilon)
    """
    loss = LabelSmoothCELoss(eps=spec.get("eps", 0.1))
    return loss, None
