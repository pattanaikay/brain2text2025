"""
tools/smoke_assert.py
---------------------
Step-50 pre-flight gate. Called by run.py before committing to a full epoch.

Checks (from profiles/toy.yaml):
  1. ce_loss < max_ce_loss     (LLM is not completely off)
  2. ctc_loss < max_ctc_loss   (CTC head initialises reasonably)
  3. no NaN in any loss
  4. throughput > min_throughput_toks  (not stalled/hung)

If any check fails, raises SmokeAssertionError and run.py aborts gracefully
before burning more GPU time.
"""

from __future__ import annotations

import math
import time


class SmokeAssertionError(RuntimeError):
    pass


class SmokeAssert:
    """
    Accumulates metrics over the first N steps and checks them at step N.

    Usage in training loop:
        smoke = SmokeAssert(profile["smoke_assert"], check_at_step=50)
        for step, batch in enumerate(loader):
            ...
            smoke.record(step, ce_loss=ce.item(), ctc_loss=ctc.item(),
                         n_tokens=sum(len(t) for t in texts))
            # Raises SmokeAssertionError if thresholds violated at step 50
    """

    def __init__(self, thresholds: dict, check_at_step: int = 50):
        self.thresholds    = thresholds
        self.check_at_step = check_at_step
        self._losses: dict[str, list[float]] = {}
        self._tokens: list[int]               = []
        self._t0 = time.time()
        self._done = False

    def record(self, step: int, n_tokens: int = 0, **losses: float):
        if self._done:
            return
        for k, v in losses.items():
            self._losses.setdefault(k, []).append(v)
        self._tokens.append(n_tokens)

        if step >= self.check_at_step:
            self._check()
            self._done = True

    def _check(self):
        elapsed  = max(time.time() - self._t0, 1e-6)
        avg      = {k: sum(v) / len(v) for k, v in self._losses.items()}
        tok_rate = sum(self._tokens) / elapsed

        errors = []

        # NaN check
        if self.thresholds.get("nan_check", True):
            for k, v in avg.items():
                if math.isnan(v) or math.isinf(v):
                    errors.append(f"NaN/Inf in {k}={v}")

        # CE loss ceiling
        max_ce = self.thresholds.get("max_ce_loss", 10.0)
        if "ce_loss" in avg and avg["ce_loss"] > max_ce:
            errors.append(f"ce_loss={avg['ce_loss']:.4f} > threshold {max_ce}")

        # CTC loss ceiling
        max_ctc = self.thresholds.get("max_ctc_loss", 5.0)
        if "ctc_loss" in avg and avg["ctc_loss"] > max_ctc:
            errors.append(f"ctc_loss={avg['ctc_loss']:.4f} > threshold {max_ctc}")

        # Throughput floor
        min_tok = self.thresholds.get("min_throughput_toks", 50)
        if tok_rate < min_tok:
            errors.append(f"throughput={tok_rate:.1f} tok/s < threshold {min_tok}")

        if errors:
            msg = "\n".join([f"  ✗ {e}" for e in errors])
            raise SmokeAssertionError(
                f"Smoke assert FAILED at step {self.check_at_step}:\n{msg}\n"
                "Fix the issue before running a full epoch."
            )

        print(
            f"[smoke] ✓ PASSED at step {self.check_at_step} — "
            f"ce={avg.get('ce_loss',0):.3f}, "
            f"ctc={avg.get('ctc_loss',0):.3f}, "
            f"throughput={tok_rate:.0f} tok/s"
        )
