"""
stages/loss/ctc_anneal.py
--------------------------
CTC auxiliary loss with optional linear annealing (Track D1).

Variants:
    D1a: fixed ctc_weight=0.3          (anneal_epochs=None)
    D1b: linear anneal 0.3→0.0 / 75ep  (anneal_epochs=75)
    D1c: step anneal                    (step_schedule=[(50,0.1),(100,0.0)])
    D1d: ctc_weight=0.0                 (start_weight=0.0)

CTC forward pass: encoder outputs → ctc_head (Linear 384→42) → CTCLoss.
The ctc_head is created here and must be registered on the Stack encoder
(run.py does this after Stack.from_spec).
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional


class CTCAnnealLoss(nn.Module):
    def __init__(
        self,
        embed_dim:     int   = 384,
        n_phonemes:    int   = 42,
        start_weight:  float = 0.3,
        anneal_epochs: Optional[int] = 75,
        step_schedule: Optional[list] = None,
    ):
        super().__init__()
        self.ctc_head     = nn.Linear(embed_dim, n_phonemes)
        self.ctc_loss_fn  = nn.CTCLoss(blank=0, zero_infinity=True)
        self.start_weight = start_weight
        self.anneal_epochs = anneal_epochs
        self.step_schedule = step_schedule  # [(epoch_thresh, new_weight), ...]
        self._current_weight = start_weight

    def set_epoch(self, epoch: int):
        """Call at the start of each epoch to update the annealed weight."""
        if self.start_weight == 0.0:
            self._current_weight = 0.0
        elif self.step_schedule:
            w = self.start_weight
            for thresh, new_w in sorted(self.step_schedule):
                if epoch >= thresh:
                    w = new_w
            self._current_weight = w
        elif self.anneal_epochs:
            self._current_weight = max(0.0, self.start_weight * (1.0 - epoch / self.anneal_epochs))
        else:
            self._current_weight = self.start_weight

    def __call__(self, batch, stack, outputs: dict) -> dict:
        if self._current_weight == 0.0:
            device = next(self.ctc_head.parameters()).device
            return {"loss_ctc": torch.tensor(0.0, device=device)}

        neural_tokens   = outputs.get("neural_tokens")    # (B, T_patch, embed_dim)
        neural_lengths  = batch.get("neural_lengths")
        phonemes        = batch.get("phonemes")
        phoneme_lengths = batch.get("phoneme_lengths")

        if neural_tokens is None or phonemes is None:
            device = next(self.ctc_head.parameters()).device
            return {"loss_ctc": torch.tensor(0.0, device=device)}

        patch_size = stack.encoder.patch_size if hasattr(stack.encoder, "patch_size") else 4

        ctc_logits   = self.ctc_head(neural_tokens.float())
        ctc_log_prob = nn.functional.log_softmax(ctc_logits, dim=-1).permute(1, 0, 2)
        patched_len  = (neural_lengths + patch_size - 1) // patch_size

        ctc_loss = self.ctc_loss_fn(
            ctc_log_prob, phonemes,
            patched_len.cpu(), phoneme_lengths.cpu(),
        )
        return {"loss_ctc": ctc_loss * self._current_weight}


def build(spec: dict, prev_shape) -> tuple:
    """
    spec keys:
        embed_dim     : int   = 384
        n_phonemes    : int   = 42
        start_weight  : float = 0.3
        anneal_epochs : int   = 75    (None = fixed weight)
        step_schedule : list  = null  # [(50, 0.1), (100, 0.0)]
    """
    loss = CTCAnnealLoss(
        embed_dim     = spec.get("embed_dim",     384),
        n_phonemes    = spec.get("n_phonemes",    42),
        start_weight  = spec.get("start_weight",  0.3),
        anneal_epochs = spec.get("anneal_epochs", 75),
        step_schedule = spec.get("step_schedule", None),
    )
    return loss, None
