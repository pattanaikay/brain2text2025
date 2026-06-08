"""
core/consolidator.py
--------------------
The N-step "sleep" consolidator — DietCorp TTA generalised to N passes, with optional
episodic-memory anchor, LM-refined / oracle pseudo-labels, and replay-batch consolidation.

Per trial (between sentences):
  1. pseudo_label(clean trial)  -- mode = self | lm | oracle
  2. time-mask augmentations (DietCorp)
  3. N AdamW steps on patch-embed only; loss = CTC(aug, pseudo_label)
        [+ episodic_weight * MSE(latent, memory_readout)]   when a memory is attached
  4. (optional) the step batch also includes prioritized replay trials

Wake latency = one clean forward (through memory read when attached); it is independent of N.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.lm_refine import ctc_beam_search


def ctc_greedy_decode(log_probs: torch.Tensor, blank: int = 0) -> list[torch.Tensor]:
    if log_probs.dim() == 2:
        log_probs = log_probs.unsqueeze(0)
    out = []
    for seq in log_probs.argmax(dim=-1):
        collapsed, prev = [], None
        for tok in seq.tolist():
            if tok != prev and tok != blank:
                collapsed.append(tok)
            prev = tok
        out.append(torch.tensor(collapsed, dtype=torch.long))
    return out


def _now_ms(device):
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() * 1000.0


@dataclass
class TTAConfig:
    n_aug: int = 64
    mask_frac: float = 0.53
    mask_span: int = 4
    lr: float = 1e-5
    grad_clip: float = 1.0
    blank: int = 0
    confidence_threshold: float = 0.0
    min_pseudo_len: int = 1
    episodic_weight: float = 1.0
    lm_beam: int = 8
    lm_weight: float = 0.5


class TTAConsolidator:
    def __init__(self, model, target_params, memory=None, lm=None,
                 config: TTAConfig | None = None):
        self.model = model
        self.memory = memory
        self.lm = lm
        self.target_params = [p for p in target_params if p.requires_grad]
        if not self.target_params:
            raise ValueError("no trainable target_params")
        self.cfg = config or TTAConfig()
        self.ctc = nn.CTCLoss(blank=self.cfg.blank, zero_infinity=True)
        self._device = self.target_params[0].device

    # ── forward (optionally through memory) ────────────────────────────────────
    def logits(self, neural, session_id=None, write_memory=False):
        if neural.dim() == 2:
            neural = neural.unsqueeze(0)
        tok = self.model.encode(neural, session_id=session_id)        # (B,T,E)
        if self.memory is not None:
            tok = self.memory(tok, write=write_memory, session_id=session_id)
        return self.model.head(tok)                                   # (B,T,V)

    # ── augmentation ───────────────────────────────────────────────────────────
    def augment(self, neural, generator=None):
        if neural.dim() == 3:
            neural = neural.squeeze(0)
        T, C = neural.shape
        aug = neural.unsqueeze(0).repeat(self.cfg.n_aug, 1, 1).clone()
        span = max(1, self.cfg.mask_span)
        n_spans = max(1, int(round(self.cfg.mask_frac * T / span)))
        for a in range(self.cfg.n_aug):
            starts = torch.randint(0, max(1, T - span + 1), (n_spans,),
                                   generator=generator, device=neural.device)
            for s in starts.tolist():
                aug[a, s:s + span, :] = 0.0
        return aug

    # ── pseudo-label ───────────────────────────────────────────────────────────
    @torch.no_grad()
    def pseudo_label(self, neural, mode="self", true_labels=None, session_id=None):
        """Returns (labels: LongTensor on device, confidence: float)."""
        logits = self.logits(neural, session_id=session_id, write_memory=False)
        log_prob = F.log_softmax(logits, dim=-1)
        confidence = float(log_prob.exp().max(dim=-1).values.mean().item())
        if mode == "oracle":
            labels = true_labels.to(self._device).long()
        elif mode == "lm" and self.lm is not None:
            ids = ctc_beam_search(log_prob[0], lm=self.lm, beam_size=self.cfg.lm_beam,
                                  lm_weight=self.cfg.lm_weight, blank=self.cfg.blank)
            labels = torch.tensor(ids, dtype=torch.long, device=self._device)
        else:  # self
            labels = ctc_greedy_decode(log_prob, blank=self.cfg.blank)[0].to(self._device)
        return labels, confidence

    # ── one CTC term over an augmentation batch with a single target ───────────
    def _ctc_term(self, aug, labels):
        logits = self.logits(aug, write_memory=False)                # (A,T',V)
        log_prob = F.log_softmax(logits, dim=-1).permute(1, 0, 2)    # (T',A,V)
        A = aug.size(0)
        in_len = torch.full((A,), logits.size(1), dtype=torch.long)
        tgt = labels.unsqueeze(0).repeat(A, 1)
        tgt_len = torch.full((A,), labels.numel(), dtype=torch.long)
        loss = self.ctc(log_prob, tgt, in_len, tgt_len)
        if self.memory is not None and self.memory.last_read:
            q = self.memory.last_read["memory_query"]
            r = self.memory.last_read["memory_retrieved"]
            loss = loss + self.cfg.episodic_weight * F.mse_loss(q, r.detach())
        return loss

    # ── the N-step sleep ───────────────────────────────────────────────────────
    def consolidate(self, trials, n_steps=1):
        """
        trials: list of dicts {neural (T,C), labels (L,), session}. The first is the current
        trial; any extras are prioritized replay items. Returns a metrics dict.
        """
        if not trials:
            return {"skipped": True, "n_steps": 0}
        cur = trials[0]
        neural = cur["neural"]
        if neural.dim() == 2:
            neural = neural.unsqueeze(0)
        neural = neural.to(self._device)

        skipped = (cur["labels"].numel() < self.cfg.min_pseudo_len)

        t0 = _now_ms(self._device)                                   # wake latency (N-independent)
        with torch.no_grad():
            _ = self.logits(neural, session_id=cur.get("session"), write_memory=False)
        wake_ms = _now_ms(self._device) - t0

        if skipped:
            return {"skipped": True, "n_steps": 0, "wake_latency_ms": wake_ms,
                    "consolidate_ms": 0.0, "loss_before": None, "loss_after": None,
                    "params_changed": 0}

        # pre-augment each trial once
        batches = [(self.augment(t["neural"].to(self._device)), t["labels"].to(self._device))
                   for t in trials]

        def _total():
            return sum(self._ctc_term(aug, lab) for aug, lab in batches) / len(batches)

        with torch.no_grad():
            loss_before = float(_total().item())

        before = [p.detach().clone() for p in self.target_params]
        opt = torch.optim.AdamW(self.target_params, lr=self.cfg.lr)
        t1 = _now_ms(self._device)
        for _ in range(max(1, n_steps)):
            opt.zero_grad()
            loss = _total()
            loss.backward()
            nn.utils.clip_grad_norm_(self.target_params, self.cfg.grad_clip)
            opt.step()
        consolidate_ms = _now_ms(self._device) - t1

        with torch.no_grad():
            loss_after = float(_total().item())
        params_changed = sum(int(not torch.equal(b, p.detach()))
                             for b, p in zip(before, self.target_params))

        # write the current trial into episodic memory (confident only — gated in policy)
        if self.memory is not None:
            with torch.no_grad():
                tok = self.model.encode(neural, session_id=cur.get("session"))
                self.memory.write_policy.write(self.memory, tok, confidence=cur.get("confidence"),
                                               session_id=None)

        return {"skipped": False, "n_steps": int(max(1, n_steps)),
                "wake_latency_ms": wake_ms, "consolidate_ms": consolidate_ms,
                "loss_before": loss_before, "loss_after": loss_after,
                "params_changed": params_changed}
