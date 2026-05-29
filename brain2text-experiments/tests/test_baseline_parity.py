"""
tests/test_baseline_parity.py
------------------------------
Verify that the Stack-based BIT+MLP+Qwen15 forward pass produces the same
encoder outputs as the upstream BrainToTextE2E.

This test ensures the docks/ shims + stage adapters introduce no numerical
divergence vs the reference implementation.

Run: python -m pytest tests/test_baseline_parity.py -v -s
NOTE: This test requires CUDA and the Qwen2.5-1.5B model to be downloaded.
      Skip in CI by: pytest ... -m "not slow"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch


@pytest.mark.slow
def test_encoder_parity():
    """
    Encoder output from stages/encoder/bit.py (via Stack) must match
    BIT_Transformer from bit_dock.py when called with identical weights.
    """
    from docks.bit_dock    import BIT_Transformer
    from stages.encoder.bit import build as bit_build

    spec = {"patch_size": 4, "num_layers": 2, "num_heads": 6}
    enc_stage, _ = bit_build(spec, (240, 512))
    enc_ref      = BIT_Transformer(patch_size=4, num_layers=2, num_heads=6)

    # Copy weights
    enc_stage.load_state_dict(enc_ref.state_dict(), strict=False)

    x = torch.randn(2, 240, 512)
    enc_stage.eval(); enc_ref.eval()
    with torch.no_grad():
        out_stage = enc_stage(x)
        out_ref   = enc_ref(x)

    assert torch.allclose(out_stage, out_ref, atol=1e-5), (
        f"Max diff: {(out_stage - out_ref).abs().max().item():.2e}"
    )


@pytest.mark.slow
def test_projector_mlp_parity():
    """MLP projector from stages must match MLPProjector from bit_dock."""
    from docks.bit_dock      import MLPProjector
    from stages.projector.mlp import build as mlp_build

    proj_stage, _ = mlp_build({"input_dim": 384, "hidden_dim": 1024, "output_dim": 1536},
                               (60, 384))
    proj_ref = MLPProjector(input_dim=384, hidden_dim=1024, output_dim=1536)
    proj_stage.load_state_dict(proj_ref.state_dict())

    x = torch.randn(2, 60, 384)
    proj_stage.eval(); proj_ref.eval()
    with torch.no_grad():
        out_stage = proj_stage(x)
        out_ref   = proj_ref(x)

    assert torch.allclose(out_stage, out_ref, atol=1e-6)


def test_spec_hash_deterministic():
    """Same spec must always produce the same hash (cache key stability)."""
    from stack import Stack
    spec = {"encoder": {"variant": "bit", "patch_size": 4},
            "projector": {"variant": "mlp"}}
    h1 = Stack.spec_hash(spec)
    h2 = Stack.spec_hash(spec)
    assert h1 == h2

    # Different spec → different hash
    spec2 = {"encoder": {"variant": "conformer", "patch_size": 4},
              "projector": {"variant": "mlp"}}
    h3 = Stack.spec_hash(spec2)
    assert h1 != h3


if __name__ == "__main__":
    import pytest as _pt
    _pt.main([__file__, "-v", "-s"])
