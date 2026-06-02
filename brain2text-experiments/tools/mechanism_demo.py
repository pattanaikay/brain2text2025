"""
tools/mechanism_demo.py
-----------------------
Track G3 — mechanism validation for the sleep-consolidation thesis.

The real --adapt run on a RANDOM, untrained BIT encoder shows the drift curve
working (N=0 degrades) but naive self-labeled TTA COLLAPSING for N>=1 — because
an untrained decoder produces garbage pseudo-labels (confirmation bias). DietCorp
avoids this with (a) a trained model and (b) an n-gram-LM-refined pseudo-label.

This demo isolates the CONSOLIDATION MECHANISM from pseudo-label quality by
running the *exact same* code path (TTAConsolidator + run_drift_eval) on a
COMPETENT synthetic decoder: a frozen prototype-matching core whose clean decode
is correct by construction, preceded by an adaptable input affine (the patch-embed
analog). Now day-d pseudo-labels are mostly right, so consolidation can invert
the drift — and we can read whether deeper sleep (larger N) recovers more.

This is a SYNTHETIC mechanism check, NOT a scientific result on neural data. The
real test needs a trained BIT+CTC checkpoint (set encoder.pretrained_ckpt in
specs/G3) or LM-refined pseudo-labels — see the plan's Phase 5.

Run:  py -3 tools/mechanism_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from adapt.dietcorp_tta import TTAConfig, ctc_greedy_decode
from tools.drift_eval import synthesize_drift, run_drift_eval


class CompetentModel(nn.Module):
    """Adaptable input affine (patch-embed analog) ∘ frozen prototype-matching core."""
    def __init__(self, protos: torch.Tensor, temp: float = 8.0):
        super().__init__()
        P, C = protos.shape
        self.in_scale = nn.Parameter(torch.ones(C))
        self.in_shift = nn.Parameter(torch.zeros(C))
        self.core = nn.Linear(C, P, bias=False)
        with torch.no_grad():
            self.core.weight.copy_(protos * temp)      # argmax(core(x)) = nearest prototype
        for p in self.core.parameters():
            p.requires_grad_(False)

    def forward(self, neural):                         # (B,T,C) -> (B,T,P)
        return self.core(neural * self.in_scale + self.in_shift)


def build_trials(protos, n_trials=24, T=40, noise=0.05, seed=1):
    g = torch.Generator().manual_seed(seed)
    P, C = protos.shape
    trials = []
    for _ in range(n_trials):
        seq = torch.randint(1, P, (T,), generator=g)               # non-blank phonemes
        feats = protos[seq] + torch.randn(T, C, generator=g) * noise
        trials.append((feats, None))
    return trials


def main():
    torch.manual_seed(0)
    C, P = 16, 8
    g = torch.Generator().manual_seed(0)
    protos = torch.randn(P, C, generator=g)
    protos = protos / protos.norm(dim=1, keepdim=True)

    model = CompetentModel(protos)
    base = build_trials(protos, n_trials=24, T=40, noise=0.05)
    # Strong, mostly-additive drift: large enough that the un-adapted decoder
    # collapses, dominated by a per-channel SHIFT the affine can learn to cancel.
    # NOTE: self-labeled TTA is regime-sensitive — it recovers drift only when
    # degradation is real AND pseudo-labels stay usable. This config sits in the
    # "deeper consolidation helps up to a stability limit" regime (see read-off).
    days = synthesize_drift(base, n_days=6, scale_std=0.20, shift_std=0.70,
                            noise_std=0.05, seed=2)

    # Reference = the competent model's CLEAN day-0 decode (correct by construction).
    with torch.no_grad():
        refs = [ctc_greedy_decode(F.log_softmax(model(x.unsqueeze(0)), dim=-1))[0]
                for x, _ in list(days.values())[0]]
    for day_trials in days.values():
        for i in range(len(day_trials)):
            day_trials[i] = (day_trials[i][0], refs[i])

    cfg = TTAConfig(n_aug=32, mask_frac=0.5, mask_span=2, lr=1e-2,
                    confidence_threshold=0.0)
    res = run_drift_eval(model, model, days,
                         target_params=[model.in_scale, model.in_shift],
                         n_steps_list=[0, 1, 2, 4, 8], tta_config=cfg)

    print("\nMechanism demo — competent synthetic decoder (NOT a result on neural data)")
    print(f"{'N':>4} {'PER@day0':>9} {'PER@last':>9} {'delta(L-0)':>11} "
          f"{'wake_ms':>9} {'cons_ms':>9}")
    for N in (0, 1, 2, 4, 8):
        s = res["summary"][N]
        wake, cons = res["wake_latency_ms"][N], res["consolidate_ms"][N]
        def _f(x, p=4): return f"{x:.{p}f}" if isinstance(x, (int, float)) else "  n/a"
        print(f"{N:>4} {_f(s['per_first']):>9} {_f(s['per_last']):>9} "
              f"{_f(s['per_delta']):>11} {_f(wake,2):>9} {_f(cons,2):>9}")
    print("\nRead-off: with a competent decoder, adaptation (N>=1) should beat the")
    print("no-adapt control (N=0) at the last day, and deeper sleep (larger N) should")
    print("recover more — the mechanism the thesis predicts, in its well-posed regime.")


if __name__ == "__main__":
    main()
