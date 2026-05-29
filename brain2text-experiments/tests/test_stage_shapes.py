"""
tests/test_stage_shapes.py
--------------------------
Shape-contract tests for every stage builder.

Each stage's build(spec, prev_shape) must:
  1. Return a (Module, out_shape) tuple
  2. Module.forward() on a fake input must produce output matching out_shape
  3. Catch projector dimension mismatches before GPU time is spent

Run: python -m pytest tests/test_stage_shapes.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch


# ── Encoder stage shape tests ───────────────────────────────────────────────

@pytest.mark.parametrize("variant,spec,prev_shape,expected_T_ratio", [
    ("bit",       {"patch_size": 4}, (240, 512), 60),
    ("conformer", {"patch_size": 4, "num_layers": 2}, (240, 512), 60),
    ("hrm",       {"patch_size": 4}, (240, 512), 60),
    ("mamba",     {"patch_size": 4, "n_layers": 2, "ssm_backbone": "gru"}, (240, 512), 60),
    ("moe",       {"patch_size": 4, "num_layers": 2, "n_specific": 2, "n_shared": 1},
                  (240, 512), 60),
    ("zenbrain",  {"patch_size": 4, "num_layers": 2}, (240, 512), 60),
])
def test_encoder_shape(variant, spec, prev_shape, expected_T_ratio):
    import importlib
    mod = importlib.import_module(f"stages.encoder.{variant}")
    encoder, out_shape = mod.build(spec, prev_shape)
    assert len(out_shape) == 2, f"Expected (T_patch, embed_dim), got {out_shape}"
    T_patch, embed_dim = out_shape
    assert T_patch == expected_T_ratio, f"{variant}: T_patch={T_patch} ≠ {expected_T_ratio}"
    assert embed_dim > 0

    # Forward smoke
    B, T, C = 2, prev_shape[0], prev_shape[1]
    x = torch.randn(B, T, C)
    with torch.no_grad():
        out = encoder(x)
    assert out.shape == (B, T_patch, embed_dim), (
        f"{variant} forward: {out.shape} ≠ {(B, T_patch, embed_dim)}"
    )


# ── Projector stage shape tests ──────────────────────────────────────────────

@pytest.mark.parametrize("variant,spec,prev_shape", [
    ("mlp",      {"output_dim": 1536}, (60, 384)),
    ("deep_mlp", {"output_dim": 1536}, (60, 384)),
    ("gated",    {"output_dim": 1536}, (60, 384)),
    ("qformer",  {"output_dim": 1536, "n_queries": 8, "n_heads": 6}, (60, 384)),
])
def test_projector_shape(variant, spec, prev_shape):
    import importlib
    mod = importlib.import_module(f"stages.projector.{variant}")
    projector, out_shape = mod.build(spec, prev_shape)

    B = 2
    T, C = prev_shape
    x = torch.randn(B, T, C)

    with torch.no_grad():
        if variant == "qformer":
            out = projector(x)
        else:
            out = projector(x)

    # out_shape = (T_out, output_dim)
    assert out.shape[0] == B
    assert out.shape[-1] == spec["output_dim"], (
        f"{variant}: output dim {out.shape[-1]} ≠ {spec['output_dim']}"
    )


def test_qformer_constant_length():
    """Q-Former must always output N_QUERIES tokens regardless of input length."""
    from stages.projector.qformer import build
    n_q = 16
    proj, out_shape = build({"n_queries": n_q, "output_dim": 1536, "n_heads": 6}, (60, 384))
    assert out_shape[0] == n_q

    for T in [30, 60, 90]:
        x = torch.randn(2, T, 384)
        with torch.no_grad():
            out = proj(x)
        assert out.shape == (2, n_q, 1536), f"T={T}: {out.shape}"


# ── Loss stage: smoke (no full forward needed) ──────────────────────────────

def test_ctc_anneal_weight_schedule():
    from stages.loss.ctc_anneal import CTCAnnealLoss
    loss = CTCAnnealLoss(start_weight=0.3, anneal_epochs=75)
    loss.set_epoch(0);  assert abs(loss._current_weight - 0.3) < 1e-6
    loss.set_epoch(37); assert abs(loss._current_weight - 0.15) < 0.02
    loss.set_epoch(75); assert abs(loss._current_weight - 0.0)  < 1e-6
    loss.set_epoch(100);assert abs(loss._current_weight - 0.0)  < 1e-6


def test_compose_ce_only():
    """compose([ce]) should return loss_ce == outputs['ce_loss']."""
    from compose import compose_from_spec
    composed = compose_from_spec([{"variant": "ce"}])

    fake_loss = torch.tensor(2.5)
    outputs   = {"ce_loss": fake_loss}
    total, bd = composed({}, None, outputs)
    assert abs(total.item() - 2.5) < 1e-5, f"Expected 2.5, got {total.item()}"
    assert "loss_ce" in bd


def test_compose_multi_loss_sum():
    """compose([ce, ctc_anneal(weight=0)]) should equal ce alone."""
    from compose import compose_from_spec
    import torch
    composed = compose_from_spec([
        {"variant": "ce"},
        {"variant": "ctc_anneal", "start_weight": 0.0},
    ])
    outputs = {"ce_loss": torch.tensor(3.0)}
    total, bd = composed({}, None, outputs)
    assert abs(total.item() - 3.0) < 1e-4


# ── Stack.from_spec shape propagation ────────────────────────────────────────

def test_stack_builds_and_shapes():
    """Stack should build with small config and report correct shapes."""
    from stack import Stack
    spec = {
        "encoder":   {"variant": "bit",    "num_layers": 2, "patch_size": 4},
        "projector": {"variant": "qformer","n_queries": 8, "n_heads": 6, "output_dim": 1536},
    }
    # No decoder — stack without decoder is valid (decoder tested separately)
    stack = Stack.from_spec(spec, prev_shape=(240, 512))
    assert stack.output_shape("encoder")   == (60, 384)
    assert stack.output_shape("projector") == (8, 1536)

    # Verify forward pass
    x = torch.randn(2, 240, 512)
    with torch.no_grad():
        enc_out  = stack.encoder(x)
        proj_out = stack.projector(enc_out)
    assert enc_out.shape  == (2, 60, 384)
    assert proj_out.shape == (2, 8, 1536)


if __name__ == "__main__":
    import pytest as _pt
    _pt.main([__file__, "-v"])
