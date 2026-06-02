"""
tools/drift_eval.py
-------------------
Track G2/G3: the day-to-day DRIFT evaluation harness — the experiment that
actually answers H_main ("does deeper sleep-consolidation help under drift?").

DietCorp's headline number is a curve, not a scalar: WER climbs 22.7% → 66.5%
over 8 held-out days WITHOUT adaptation, and stays flat WITH it. This module
reproduces that protocol locally and lets us sweep the consolidation depth N:

    for each consolidation depth N:
        restore model to baseline
        for each day in chronological order:
            evaluate  (held-out, BEFORE adapting on this day)
            adapt     (TTAConsolidator.consolidate(trial, n_steps=N) per trial)

H_main is read off the resulting per-N curves: error@last-day should fall as N
rises, while wake latency (reported by the consolidator) stays flat.

Pure-torch and model-agnostic on purpose: it takes a `logits_fn(neural)->(B,T',P)`
and works on a tiny synthetic teacher (CPU unit tests) or the real BIT encoder +
CTC head (run.py --adapt). HDF5/dataset glue lives in run.py, not here, so the
unit tests don't need the sibling repo or a GPU.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from adapt.dietcorp_tta import TTAConsolidator, TTAConfig, ctc_greedy_decode


# ── Metrics ──────────────────────────────────────────────────────────────────

def levenshtein(a: list[int], b: list[int]) -> int:
    """Classic edit distance (token-level), used for phoneme error rate."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def per(hyp: torch.Tensor, ref: torch.Tensor) -> float:
    """Phoneme error rate = edit_distance(hyp, ref) / len(ref). 1.0 if ref empty."""
    ref_list = ref.tolist()
    if not ref_list:
        return 1.0 if hyp.numel() else 0.0
    return levenshtein(hyp.tolist(), ref_list) / len(ref_list)


# ── Day construction ─────────────────────────────────────────────────────────

def split_by_session(trials: list[tuple]) -> "OrderedDict[str, list[tuple]]":
    """
    Group (neural, ref_labels, session) triples into an ordered day→trials map.
    Days are sorted by session key so the sequence is chronological-ish.
    """
    by_day: "OrderedDict[str, list[tuple]]" = OrderedDict()
    for neural, ref, session in trials:
        by_day.setdefault(str(session), []).append((neural, ref))
    return OrderedDict(sorted(by_day.items(), key=lambda kv: kv[0]))


def synthesize_drift(base_trials: list[tuple], n_days: int = 8,
                     scale_std: float = 0.15, shift_std: float = 0.15,
                     noise_std: float = 0.05, seed: int = 0
                     ) -> "OrderedDict[str, list[tuple]]":
    """
    The HTML "safe path" (Decision 3): manufacture an ordered drift sequence by
    progressively perturbing real day-0 trials with a per-day channel affine +
    Gaussian noise, preserving per-trial structure and labels. Day 0 is clean;
    perturbation magnitude grows linearly with the day index.

    base_trials : list[(neural (T,C), ref_labels (L,))].
    Returns day→[(neural, ref)], day 'd0' .. 'd{n_days-1}'.
    """
    # Reproducible noise: draw on CPU with a seeded generator, move to the trial's
    # device when applying (base trials may live on GPU).
    g = torch.Generator().manual_seed(seed)
    days: "OrderedDict[str, list[tuple]]" = OrderedDict()
    C = base_trials[0][0].shape[-1]
    for d in range(n_days):
        frac = d / max(1, n_days - 1)                       # 0 .. 1
        scale = 1.0 + torch.randn(C, generator=g) * scale_std * frac
        shift = torch.randn(C, generator=g) * shift_std * frac
        trials = []
        for neural, ref in base_trials:
            dev = neural.device
            noise = torch.randn(neural.shape, generator=g) * noise_std * frac
            x = neural * scale.to(dev) + shift.to(dev) + noise.to(dev)
            trials.append((x, ref))
        days[f"d{d}"] = trials
    return days


# ── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_day(logits_fn: Callable[[torch.Tensor], torch.Tensor],
                 trials: list[tuple], blank: int = 0) -> dict:
    """Mean PER (where refs exist) and mean confidence over a day's trials."""
    pers, confs = [], []
    for neural, ref in trials:
        x = neural if neural.dim() == 3 else neural.unsqueeze(0)
        logits = logits_fn(x)
        log_prob = F.log_softmax(logits, dim=-1)
        hyp = ctc_greedy_decode(log_prob, blank=blank)[0]
        confs.append(float(log_prob.exp().max(dim=-1).values.mean().item()))
        if ref is not None:
            pers.append(per(hyp, ref))
    return {
        "per":  (sum(pers) / len(pers)) if pers else None,
        "confidence": sum(confs) / max(1, len(confs)),
        "n_trials": len(trials),
    }


def run_drift_eval(model: nn.Module,
                   logits_fn: Callable[[torch.Tensor], torch.Tensor],
                   days: "OrderedDict[str, list[tuple]]",
                   target_params,
                   n_steps_list: list[int] = (0, 1, 2, 4, 8),
                   tta_config: Optional[TTAConfig] = None,
                   blank: int = 0) -> dict:
    """
    The core sweep. For each N in n_steps_list (N=0 = no adaptation control):
    restore the model to its baseline state, then walk the days in order,
    evaluating each day BEFORE adapting on it, then consolidating N steps/trial.

    Returns:
        {
          "by_n": { N: [ {day, per, confidence, ...}, ... ] },
          "wake_latency_ms": { N: mean },     # must be ~flat across N
          "consolidate_ms":  { N: mean },     # grows ~linearly with N
          "summary": { N: {"per_first", "per_last", "per_delta"} },
        }
    """
    target_params = list(target_params)
    base_state = copy.deepcopy(model.state_dict())
    out = {"by_n": {}, "wake_latency_ms": {}, "consolidate_ms": {}, "summary": {}}

    for N in n_steps_list:
        model.load_state_dict(base_state)
        consolidator = TTAConsolidator(logits_fn, target_params, tta_config)
        curve, wake, cons = [], [], []
        for day, trials in days.items():
            ev = evaluate_day(logits_fn, trials, blank=blank)
            ev["day"] = day
            curve.append(ev)
            if N > 0:
                for neural, _ in trials:
                    m = consolidator.consolidate(neural, n_steps=N)
                    wake.append(m["wake_latency_ms"])
                    if not m["skipped"]:
                        cons.append(m["consolidate_ms"])
        out["by_n"][N] = curve
        out["wake_latency_ms"][N] = (sum(wake) / len(wake)) if wake else None
        out["consolidate_ms"][N] = (sum(cons) / len(cons)) if cons else None
        pers = [c["per"] for c in curve if c["per"] is not None]
        out["summary"][N] = {
            "per_first": pers[0] if pers else None,
            "per_last":  pers[-1] if pers else None,
            "per_delta": (pers[-1] - pers[0]) if len(pers) >= 2 else None,
        }
    return out
