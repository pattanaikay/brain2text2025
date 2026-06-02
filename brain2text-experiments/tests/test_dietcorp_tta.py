"""
tests/test_dietcorp_tta.py
--------------------------
Mechanism tests for the native DietCorp TTA consolidator (Track G2/G3).

These assert the CONSOLIDATION MECHANISM is correct — not the scientific result
(whether N helps under real drift is what the experiment measures). Specifically:
  1. ctc_greedy_decode collapses repeats and drops blanks.
  2. augment() time-masks ~mask_frac of the trial and preserves shape.
  3. pseudo_label() returns a label sequence + a confidence in [0, 1].
  4. consolidate() lowers the pseudo-label CTC loss and moves ONLY the target
     (patch-embed) params, leaving the frozen core untouched.
  5. The confidence gate skips consolidation (and changes nothing) when tripped.
  6. Wake latency is reported and N-independent in spirit (single clean forward).

Run:  py -3 -m pytest tests/test_dietcorp_tta.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from adapt.dietcorp_tta import (
    TTAConsolidator, TTAConfig, ctc_greedy_decode, select_patch_embed_params,
)


class TinyPhonemeModel(nn.Module):
    """Adaptable input affine (the patch-embed analog) + a FROZEN linear core."""
    def __init__(self, C=8, P=8):
        super().__init__()
        self.in_scale = nn.Parameter(torch.ones(C))
        self.in_shift = nn.Parameter(torch.zeros(C))
        self.core = nn.Linear(C, P)
        for p in self.core.parameters():
            p.requires_grad_(False)

    def forward(self, neural):                       # (B,T,C) -> (B,T,P)
        return self.core(neural * self.in_scale + self.in_shift)


def _model_and_consolidator(n_aug=8, conf_thr=0.0):
    torch.manual_seed(0)
    model = TinyPhonemeModel()
    cfg = TTAConfig(n_aug=n_aug, mask_frac=0.5, mask_span=2,
                    confidence_threshold=conf_thr)
    cons = TTAConsolidator(model, [model.in_scale, model.in_shift], cfg)
    return model, cons


def test_ctc_greedy_decode_collapses_and_drops_blank():
    # frames: blank,a,a,blank,b -> [a, b]   (blank=0, a=1, b=2)
    P = 3
    seq = torch.tensor([0, 1, 1, 0, 2])
    log_probs = torch.full((1, len(seq), P), -10.0)
    for t, tok in enumerate(seq):
        log_probs[0, t, tok] = 0.0
    out = ctc_greedy_decode(log_probs, blank=0)[0]
    assert out.tolist() == [1, 2]


def test_augment_shape_and_mask_fraction():
    model, cons = _model_and_consolidator()
    neural = torch.randn(20, 8)
    aug = cons.augment(neural)
    assert aug.shape == (8, 20, 8)
    # A meaningful fraction of timesteps should be fully zeroed by masking.
    zero_rows = (aug.abs().sum(dim=-1) == 0).float().mean().item()
    assert 0.15 < zero_rows < 0.95, f"unexpected masked fraction {zero_rows}"


def test_pseudo_label_shape_and_confidence():
    model, cons = _model_and_consolidator()
    neural = torch.randn(20, 8)
    labels, conf = cons.pseudo_label(neural)
    assert labels.dim() == 1
    assert 0.0 <= conf <= 1.0


def test_consolidate_lowers_loss_and_touches_only_target_params():
    model, cons = _model_and_consolidator()
    neural = torch.randn(20, 8)
    core_before = [p.detach().clone() for p in model.core.parameters()]

    m = cons.consolidate(neural, n_steps=10)

    assert not m["skipped"]
    assert m["loss_after"] < m["loss_before"], m
    assert m["params_changed"] == 2          # both in_scale and in_shift moved
    assert m["wake_latency_ms"] >= 0.0
    assert m["consolidate_ms"] >= 0.0
    # Frozen core must be byte-identical after consolidation.
    for b, p in zip(core_before, model.core.parameters()):
        assert torch.equal(b, p), "consolidation modified the frozen core"


def test_confidence_gate_skips_and_changes_nothing():
    model, cons = _model_and_consolidator(conf_thr=1.1)   # impossible threshold
    neural = torch.randn(20, 8)
    scale_before = model.in_scale.detach().clone()

    m = cons.consolidate(neural, n_steps=5)

    assert m["skipped"] is True
    assert m["params_changed"] == 0
    assert torch.equal(scale_before, model.in_scale.detach())


def test_select_patch_embed_params_picks_by_name():
    model = TinyPhonemeModel()
    chosen = select_patch_embed_params(model, name_hints=("in_scale", "in_shift"))
    ids = {id(p) for p in chosen}
    assert id(model.in_scale) in ids and id(model.in_shift) in ids
    assert id(next(model.core.parameters())) not in ids


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
