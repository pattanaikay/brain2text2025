"""
tests/test_zenbrain_stub_tripwire.py
------------------------------------
Track H — LIVE contract (formerly the stub tripwire, now inverted).

The skeleton has been resolved: the ZenBrain episodic-memory stage is real and
backprop-safe, the write/evict policy is session-keyed + confidence-gated, and
the episodic-consistency loss is live. This test now asserts the LIVE contract —
it fails loudly if someone reverts the stage to a stub without flipping the
registry/health state back, the mirror image of the original tripwire.

Asserted (CPU, seconds):
  1. The memory stage builds with the correct passthrough out_shape.
  2. forward() runs WITHOUT any opt-in flag, returns the right shape, and
     backprops into the read head (it is no longer a guarded stub).
  3. forward() emits memory_query / memory_retrieved for the consistency loss.
  4. The write policy is confidence-gated: high-confidence trials are stored,
     low-confidence trials are skipped.
  5. The episodic_consistency loss is now NON-zero when memory tensors are
     present (and still a backprop-safe zero when the memory stage is absent).
  6. registry.yaml H1 state is 'partial' (no longer 'skeleton') and the
     health.json agrees with a non-empty resume_note.

Run:  py -3 -m pytest tests/test_zenbrain_stub_tripwire.py -v
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from stages.memory.episodic_buffer import build as mem_build, EpisodicBuffer
from stages.loss.episodic_consistency import build as loss_build

_ROOT  = Path(__file__).resolve().parent.parent
_SPECS = _ROOT / "specs"


def test_memory_stage_shape_passthrough():
    module, out_shape = mem_build({"embed_dim": 384, "buffer_size": 64}, (60, 384))
    assert out_shape == (60, 384), f"memory stage must passthrough shape, got {out_shape}"


def test_forward_is_live_and_backprops():
    module, _ = mem_build({"embed_dim": 384, "buffer_size": 64}, (60, 384))
    x = torch.randn(2, 60, 384, requires_grad=True)
    out = module(x)
    assert out.shape == (2, 60, 384), f"read path wiring broken: {out.shape}"
    out.sum().backward()
    # Gradient must reach a trainable read-path parameter.
    grads = [p.grad for p in module.read_head.parameters() if p.grad is not None]
    assert grads, "no gradient reached the cross-attention read head — not live"


def test_forward_emits_memory_tensors():
    module, _ = mem_build({"embed_dim": 384, "buffer_size": 64}, (60, 384))
    module(torch.randn(2, 60, 384))
    assert "memory_query" in module.last_read
    assert "memory_retrieved" in module.last_read
    assert module.last_read["memory_retrieved"].shape == (2, 60, 384)


def test_write_policy_is_confidence_gated():
    module = EpisodicBuffer(embed_dim=8, buffer_size=16, n_heads=2,
                            confidence_threshold=0.5)
    x = torch.randn(3, 5, 8)
    # All high-confidence → 3 slots written.
    module.write_policy.write(module, x, confidence=0.9, session_id=0)
    assert int(module.write_ptr.item()) == 3
    assert int((module.buf_session == 0).sum().item()) == 3
    # All low-confidence → nothing written (pointer unchanged).
    module.write_policy.write(module, x, confidence=0.1, session_id=1)
    assert int(module.write_ptr.item()) == 3


def test_episodic_loss_is_live_when_memory_present():
    loss_fn, _ = loss_build({"weight": 1.0}, None)
    q = torch.randn(2, 6, 8, requires_grad=True)
    r = torch.randn(2, 6, 8)
    out = loss_fn({}, None, {"memory_query": q, "memory_retrieved": r})
    val = out["loss_episodic"]
    assert float(val) > 0.0, "episodic loss must be non-zero when memory is wired"
    val.backward()
    assert q.grad is not None, "episodic loss must backprop into the query latent"


def test_episodic_loss_safe_zero_when_memory_absent():
    loss_fn, _ = loss_build({}, None)
    out = loss_fn({}, None, {})                 # no memory tensors
    val = out["loss_episodic"]
    assert float(val) == 0.0
    val.backward()                              # must not break the graph


def test_registry_state_is_partial_now():
    registry = yaml.safe_load((_ROOT / "registry.yaml").read_text())
    h1 = registry["experiments"]["H1"]
    assert h1.get("state") == "partial", (
        "H1 memory stage is live but registry state is not 'partial' — "
        "keep registry/health in sync with the code, or revert the stage to a stub."
    )


def test_health_breadcrumb_present_and_partial():
    health = json.loads((_SPECS / "H1_zenbrain_episodic.health.json").read_text())
    assert health["state"] == "partial"
    assert health.get("resume_note", "").strip(), "resume_note must be non-empty"


if __name__ == "__main__":
    import pytest as _pt
    _pt.main([__file__, "-v"])
