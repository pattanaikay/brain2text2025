"""Encoder contract tests (§14.1).

Run from the repo root:
    python -m pytest tests/test_encoder_contracts.py -v

Every encoder in ENCODER_REGISTRY must:
  - Accept (B=2, T=100, C=512) + session_id + neural_lengths + mask_patches
  - Return finite (2, 25, 384) for patch_size=4
  - Have a mask_token parameter
"""

import sys
import os

# Ensure repo root is on sys.path so `src.*` imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from src.models.registry import ENCODER_REGISTRY


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B, T, C = 2, 100, 512
PATCH_SIZE = 4
T_PATCH = T // PATCH_SIZE  # 25

SESSION_IDS = ["s1", "s2"]
NEURAL_LENS = torch.tensor([100, 80])

INPUT = torch.randn(B, T, C)
MASK = torch.zeros(B, T_PATCH, dtype=torch.bool)
MASK[0, 5:10] = True   # mask 5 patches in sample 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_encoder(name):
    """Instantiate encoder with common kwargs; gracefully absorb extras."""
    return ENCODER_REGISTRY[name](
        session_ids=SESSION_IDS,
        patch_size=PATCH_SIZE,
    ).eval()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(ENCODER_REGISTRY))
def test_output_shape(name):
    enc = _make_encoder(name)
    with torch.no_grad():
        out = enc(INPUT.clone(), session_id=SESSION_IDS, neural_lengths=NEURAL_LENS)
    assert out.shape == (B, T_PATCH, 384), (
        f"{name}: expected (2, 25, 384), got {tuple(out.shape)}"
    )


@pytest.mark.parametrize("name", list(ENCODER_REGISTRY))
def test_output_finite(name):
    enc = _make_encoder(name)
    with torch.no_grad():
        out = enc(INPUT.clone(), session_id=SESSION_IDS, neural_lengths=NEURAL_LENS)
    assert torch.isfinite(out).all(), f"{name}: non-finite values in output"


@pytest.mark.parametrize("name", list(ENCODER_REGISTRY))
def test_mask_token_accepted(name):
    """mask_patches arg must be accepted without error (SSL contract)."""
    enc = _make_encoder(name)
    with torch.no_grad():
        out = enc(INPUT.clone(), session_id=SESSION_IDS,
                  neural_lengths=NEURAL_LENS, mask_patches=MASK)
    assert out.shape == (B, T_PATCH, 384), (
        f"{name}: mask_patches broke output shape: {tuple(out.shape)}"
    )


@pytest.mark.parametrize("name", list(ENCODER_REGISTRY))
def test_mask_token_exists(name):
    """Every encoder must expose a learnable mask_token parameter."""
    enc = _make_encoder(name)
    assert hasattr(enc, "mask_token") and isinstance(enc.mask_token, torch.nn.Parameter), (
        f"{name}: missing nn.Parameter 'mask_token'"
    )


@pytest.mark.parametrize("name", list(ENCODER_REGISTRY))
def test_no_session_id(name):
    """session_id=None must not raise."""
    enc = _make_encoder(name)
    with torch.no_grad():
        out = enc(INPUT.clone(), session_id=None, neural_lengths=NEURAL_LENS)
    assert out.shape == (B, T_PATCH, 384), f"{name}: session_id=None broke shape"


@pytest.mark.parametrize("name", list(ENCODER_REGISTRY))
def test_patch_size_attribute(name):
    enc = _make_encoder(name)
    assert hasattr(enc, "patch_size"), f"{name}: missing patch_size attribute"
    assert enc.patch_size == PATCH_SIZE, (
        f"{name}: patch_size={enc.patch_size}, expected {PATCH_SIZE}"
    )


@pytest.mark.parametrize("name", list(ENCODER_REGISTRY))
def test_embed_dim_attribute(name):
    enc = _make_encoder(name)
    assert hasattr(enc, "embed_dim"), f"{name}: missing embed_dim attribute"
    assert enc.embed_dim == 384, f"{name}: embed_dim={enc.embed_dim}, expected 384"
