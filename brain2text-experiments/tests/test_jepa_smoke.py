"""
tests/test_jepa_smoke.py
------------------------
Track F smoke tests. Proves the JEPA *wiring* is real even though the
per-modality perception is stubbed.

What is asserted (CPU, no model downloads, seconds):
  1. The three specs differ in EXACTLY the `modality:` line (controlled A/B).
  2. The JEPA encoder builds for every modality and out_shapes are identical
     across modalities (so audio/video/neural are truly swappable upstreams).
  3. pretrain_step() runs, backprops, and the loss DECREASES over a few steps
     on a single fixed batch — i.e. the EMA + stop-grad path actually learns.
  4. The EMA target-encoder receives NO gradients (stop-gradient is wired).

Run:  python -m pytest tests/test_jepa_smoke.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from tools.diff_specs import assert_differs_only
from stages.encoder.jepa import build as jepa_build

_SPECS = Path(__file__).resolve().parent.parent / "specs"


def test_specs_differ_only_in_modality():
    a = yaml.safe_load((_SPECS / "F1_audio_jepa.yaml").read_text())
    b = yaml.safe_load((_SPECS / "F2_video_jepa.yaml").read_text())
    c = yaml.safe_load((_SPECS / "F3_neural_jepa.yaml").read_text())
    # Each pair may differ ONLY in the modality leaf key.
    assert_differs_only(a, b, ["modality"])
    assert_differs_only(a, c, ["modality"])
    assert_differs_only(b, c, ["modality"])


def test_modalities_share_output_shape():
    shapes = {}
    for modality in ("audio", "video", "neural"):
        _, out_shape = jepa_build({"modality": modality, "patch_size": 4}, (240, 512))
        shapes[modality] = out_shape
    assert len(set(shapes.values())) == 1, (
        f"Modalities must produce identical out_shape for a controlled A/B; got {shapes}"
    )
    assert shapes["audio"] == (60, 384)


def test_pretrain_loss_decreases():
    torch.manual_seed(0)
    encoder, _ = jepa_build({"modality": "audio", "patch_size": 4,
                             "num_layers": 2, "ema_momentum": 0.9}, (240, 512))
    opt = torch.optim.Adam(
        [p for p in encoder.parameters() if p.requires_grad], lr=1e-3
    )
    x = torch.randn(2, 240, 512)               # one fixed batch
    first, last = None, None
    for step in range(20):
        opt.zero_grad()
        loss = encoder.pretrain_step(x)
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    assert last < first, f"JEPA loss did not decrease: {first:.4f} -> {last:.4f}"


def test_target_encoder_has_no_grad():
    encoder, _ = jepa_build({"modality": "video", "patch_size": 4,
                             "num_layers": 2}, (240, 512))
    x = torch.randn(2, 240, 512)
    loss = encoder.pretrain_step(x)
    loss.backward()
    grads = [p.grad for p in encoder.target_encoder.parameters()]
    assert all(g is None for g in grads), (
        "Stop-gradient broken: EMA target-encoder received gradients."
    )


if __name__ == "__main__":
    import pytest as _pt
    _pt.main([__file__, "-v"])
