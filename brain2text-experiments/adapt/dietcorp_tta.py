"""
adapt/dietcorp_tta.py
---------------------
Track G2/G3: the DietCorp test-time-adaptation (TTA) loop, native + generalised
to N "sleep" consolidation passes.

DietCorp (arXiv:2507.02800) fights day-to-day electrode drift with a per-trial
TTA step: after decoding a sentence it generates 64 time-masked augmentations of
the trial, refines the raw phonetic output into a pseudo-label, and takes
**exactly one** AdamW gradient step on the patch-embedding module (the network's
input), keeping the rest of the model frozen. Cost: ~18 ms/trial.

The thesis variable (from "Do Language Models Need Sleep", arXiv:2605.26099):
DietCorp's single step is the N=1 special case of a *sleep* — N recurrent offline
consolidation passes performed before the model moves on. The sleep paper's
central claim is that increasing N improves performance on deeply *sequential*
problems while keeping **wake-time inference at a single forward pass**. Neural
drift is sequential (today's signal depends on the electrode's prior state), so
H_main: WER-under-drift should decrease with N at constant wake latency.

This module is deliberately LLM-free: it operates on the cheap, DietCorp-faithful
encoder→CTC-head phoneme path. `logits_fn(neural) -> (B, T', P)` is the only model
contract, so the same consolidator drives a tiny test model on CPU or the real BIT
encoder + CTC head on GPU. The expensive downstream Qwen WER check is a separate,
cloud-gated step.

Public surface:
    TTAConsolidator(logits_fn, target_params, ...)
        .augment(neural)             -> (n_aug, T, C) time-masked copies
        .pseudo_label(neural)        -> (labels: LongTensor, confidence: float)
        .consolidate(neural, n_steps)-> metrics dict  (the N-loop sleep)
    ctc_greedy_decode(log_probs, blank=0) -> list[LongTensor]   (module fn, reused by drift_eval)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── CTC greedy decode (shared with drift_eval) ───────────────────────────────

def ctc_greedy_decode(log_probs: torch.Tensor, blank: int = 0) -> list[torch.Tensor]:
    """
    Collapse a (B, T, P) log-prob tensor into per-sequence label lists via greedy
    CTC decoding: argmax per frame → drop repeats → drop blanks.

    Returns a list of 1-D LongTensors (one per batch item), on CPU.
    """
    if log_probs.dim() == 2:
        log_probs = log_probs.unsqueeze(0)
    argmax = log_probs.argmax(dim=-1)                      # (B, T)
    out: list[torch.Tensor] = []
    for seq in argmax:
        collapsed = []
        prev = None
        for tok in seq.tolist():
            if tok != prev and tok != blank:
                collapsed.append(tok)
            prev = tok
        out.append(torch.tensor(collapsed, dtype=torch.long))
    return out


def _now_ms(device: torch.device | None) -> float:
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() * 1000.0


# ── Consolidator ─────────────────────────────────────────────────────────────

@dataclass
class TTAConfig:
    n_aug:                int   = 64      # DietCorp uses 64 time-masked augmentations
    mask_frac:            float = 0.53    # ~53% of patches masked (DietCorp time-masking)
    mask_span:            int   = 4       # contiguous masked-span length (bins)
    lr:                   float = 1e-3
    grad_clip:            float = 1.0
    blank:                int   = 0       # CTC blank index
    confidence_threshold: float = 0.0     # skip consolidation below this mean max-prob
    min_pseudo_len:       int   = 1       # skip if the pseudo-label collapses to fewer tokens


class TTAConsolidator:
    """
    The DietCorp TTA loop generalised to N consolidation passes.

    Parameters
    ----------
    logits_fn : Callable[[Tensor], Tensor]
        Maps neural input (B, T, C) → CTC logits (B, T', P). This is the
        encoder→ctc_head path; T' is the patched length.
    target_params : Iterable[nn.Parameter]
        The ONLY parameters consolidation is allowed to touch — DietCorp updates
        the input patch-embedding module and freezes the rest. Pass e.g. the
        encoder read-in / patch-embed params or the dietcorp_recal affine.
    """

    def __init__(self, logits_fn: Callable[[torch.Tensor], torch.Tensor],
                 target_params: Iterable[nn.Parameter],
                 config: TTAConfig | None = None):
        self.logits_fn = logits_fn
        self.target_params = [p for p in target_params if p.requires_grad]
        if not self.target_params:
            raise ValueError("TTAConsolidator: no trainable target_params given — "
                             "consolidation has nothing to update.")
        self.cfg = config or TTAConfig()
        self.ctc = nn.CTCLoss(blank=self.cfg.blank, zero_infinity=True)
        self._device = self.target_params[0].device

    # ── Augmentation ─────────────────────────────────────────────────────────
    def augment(self, neural: torch.Tensor,
                generator: torch.Generator | None = None) -> torch.Tensor:
        """
        Build `n_aug` time-masked copies of a single trial.

        neural : (T, C) or (1, T, C). Returns (n_aug, T, C). Masking zeros out
        contiguous spans covering ~mask_frac of the timeline — the regulariser
        that forces the model to recover structure rather than memorise frames.
        """
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

    # ── Pseudo-label ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def pseudo_label(self, neural: torch.Tensor) -> tuple[torch.Tensor, float]:
        """
        Decode the CLEAN trial into a phoneme pseudo-label (CTC greedy) and report
        a confidence = mean per-frame max softmax probability. DietCorp uses an
        n-gram LM to refine this; self-decoding is the cheap, LM-free analog and
        the hook where a KenLM rescorer can later slot in.
        """
        logits = self.logits_fn(neural if neural.dim() == 3 else neural.unsqueeze(0))
        log_prob = F.log_softmax(logits, dim=-1)
        labels = ctc_greedy_decode(log_prob, blank=self.cfg.blank)[0].to(self._device)
        confidence = float(log_prob.exp().max(dim=-1).values.mean().item())
        return labels, confidence

    # ── The N-loop sleep ─────────────────────────────────────────────────────
    def consolidate(self, neural: torch.Tensor, n_steps: int = 1) -> dict:
        """
        Run N consolidation passes on `target_params` from the trial's own
        pseudo-label, then report metrics. N=1 reproduces DietCorp; N>1 is sleep.

        Returns a metrics dict:
            skipped            : bool   (confidence/length gate tripped)
            confidence         : float
            pseudo_len         : int
            loss_before/after  : float  (CTC loss on augmentations, pre/post)
            n_steps            : int
            params_changed     : int    (target params whose value moved)
            wake_latency_ms    : float  (single clean forward — must be ~N-independent)
            consolidate_ms     : float  (total sleep wall-clock — grows ~linearly in N)
        """
        if neural.dim() == 2:
            neural = neural.unsqueeze(0)
        neural = neural.to(self._device)

        labels, confidence = self.pseudo_label(neural)
        skipped = (confidence < self.cfg.confidence_threshold
                   or labels.numel() < self.cfg.min_pseudo_len)

        # Wake-time latency: one clean forward, no grad. Independent of N by design.
        t0 = _now_ms(self._device)
        with torch.no_grad():
            _ = self.logits_fn(neural)
        wake_latency_ms = _now_ms(self._device) - t0

        if skipped:
            return {
                "skipped": True, "confidence": confidence,
                "pseudo_len": int(labels.numel()), "loss_before": None,
                "loss_after": None, "n_steps": 0, "params_changed": 0,
                "wake_latency_ms": wake_latency_ms, "consolidate_ms": 0.0,
            }

        before = [p.detach().clone() for p in self.target_params]
        opt = torch.optim.AdamW(self.target_params, lr=self.cfg.lr)
        aug = self.augment(neural)                                  # (n_aug, T, C)
        target = labels.unsqueeze(0).repeat(self.cfg.n_aug, 1)     # (n_aug, L)
        tgt_len = torch.full((self.cfg.n_aug,), labels.numel(),
                             dtype=torch.long)

        def _ctc_on_aug() -> torch.Tensor:
            logits = self.logits_fn(aug)                            # (n_aug, T', P)
            log_prob = F.log_softmax(logits, dim=-1).permute(1, 0, 2)  # (T', n_aug, P)
            in_len = torch.full((self.cfg.n_aug,), logits.size(1), dtype=torch.long)
            return self.ctc(log_prob, target, in_len, tgt_len)

        with torch.no_grad():
            loss_before = float(_ctc_on_aug().item())

        t1 = _now_ms(self._device)
        for _ in range(max(1, n_steps)):
            opt.zero_grad()
            loss = _ctc_on_aug()
            loss.backward()
            nn.utils.clip_grad_norm_(self.target_params, self.cfg.grad_clip)
            opt.step()
        consolidate_ms = _now_ms(self._device) - t1

        with torch.no_grad():
            loss_after = float(_ctc_on_aug().item())
        params_changed = sum(int(not torch.equal(b, p.detach()))
                             for b, p in zip(before, self.target_params))

        return {
            "skipped": False, "confidence": confidence,
            "pseudo_len": int(labels.numel()), "loss_before": loss_before,
            "loss_after": loss_after, "n_steps": int(max(1, n_steps)),
            "params_changed": params_changed,
            "wake_latency_ms": wake_latency_ms, "consolidate_ms": consolidate_ms,
        }


def select_patch_embed_params(module: nn.Module,
                              name_hints: tuple[str, ...] = (
                                  "patch", "read_in", "readin", "recal",
                                  "day_scale", "day_shift", "embed")) -> list[nn.Parameter]:
    """
    DietCorp consolidates only the input patch-embedding module. Pick that subset
    by name from any encoder/projector: params whose qualified name contains a
    hint. Falls back to all params if nothing matches (caller should warn).
    """
    chosen = [p for n, p in module.named_parameters()
              if any(h in n.lower() for h in name_hints)]
    return chosen if chosen else list(module.parameters())
