"""
core/replay.py
--------------
ZenBrain Simulation-Selection replay scheduler.

Not every past trial is equally worth re-consolidating. We keep a capped host-side store of
past trials and, at each sleep, replay the top-K by:

    priority = |surprise| + (1 - confidence) + novelty

  - surprise  : the trial's CTC loss when it was decoded (high = informative about drift)
  - confidence: mean per-frame max softmax prob (low confidence -> more to learn)
  - novelty   : 1 - max cosine similarity of this trial's latent to recently-replayed latents
                (covers phoneme/context patterns not seen lately)

This mirrors hippocampal prioritized replay (surprise + salience + novelty), the ZenBrain
sleep loop's selection rule, here used to choose which trials feed the N-step consolidation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class ReplayItem:
    neural: torch.Tensor            # (T, C) raw trial — re-augmented during replay
    latent: torch.Tensor           # (E,) mean-pooled latent (for novelty)
    confidence: float
    surprise: float
    session: str = ""
    ref: torch.Tensor | None = None  # optional oracle phoneme ids


class ReplayStore:
    def __init__(self, capacity: int = 512,
                 w_surprise: float = 1.0, w_conf: float = 1.0, w_novelty: float = 1.0):
        self.capacity = capacity
        self.w_surprise, self.w_conf, self.w_novelty = w_surprise, w_conf, w_novelty
        self.items: list[ReplayItem] = []
        self._recent: list[torch.Tensor] = []     # latents of recently replayed items

    def __len__(self):
        return len(self.items)

    def add(self, item: ReplayItem):
        self.items.append(item)
        if len(self.items) > self.capacity:
            self.items.pop(0)                      # FIFO eviction of oldest

    def _novelty(self, latent: torch.Tensor) -> float:
        if not self._recent:
            return 1.0
        recent = torch.stack(self._recent)         # (R, E)
        sims = F.cosine_similarity(latent.unsqueeze(0), recent, dim=-1)
        return float(1.0 - sims.max().clamp(-1, 1))

    def priority(self, item: ReplayItem) -> float:
        return (self.w_surprise * abs(item.surprise)
                + self.w_conf * (1.0 - item.confidence)
                + self.w_novelty * self._novelty(item.latent))

    def sample(self, k: int) -> list[ReplayItem]:
        """Return the top-k items by priority and mark their latents as recently replayed."""
        if not self.items:
            return []
        ranked = sorted(self.items, key=self.priority, reverse=True)
        chosen = ranked[:max(1, k)]
        self._recent = [it.latent for it in chosen][-32:]   # bound the novelty window
        return chosen
