"""
docks/bit_dock.py
-----------------
Re-exports the BIT baseline from brain2text-modeltraining under stable names.
Experiments import BIT_Transformer, MLPProjector, BrainToTextE2E from here.

Pin: see docks/PINS.txt  (brain2text-modeltraining)
"""

import sys
import os
from pathlib import Path

# ── Resolve sibling repo root ──────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent          # .../brain2text-experiments/docks
_EXPT_ROOT = _HERE.parent                        # .../brain2text-experiments
_REPO_ROOT  = _EXPT_ROOT.parent                  # .../brain2text2025

_MODELTRAINING = _REPO_ROOT / "brain2text-modeltraining"

# Add src/ so `from src.models.xxx import ...` works exactly as in the sibling
_MODELTRAINING_SRC = _MODELTRAINING / "src"
for _p in [str(_MODELTRAINING), str(_MODELTRAINING_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Stable re-exports ──────────────────────────────────────────────────────
# Keep these names stable even if upstream renames internals.

try:
    from src.models.encoder import BIT_Transformer          # noqa: F401
    from src.models.projector import MLPProjector           # noqa: F401
    from src.models.bit_e2e import BrainToTextE2E           # noqa: F401
    from src.models.bit_e2e import ModalityAlignmentLoss    # noqa: F401
except ImportError as e:
    raise ImportError(
        f"[bit_dock] Failed to import from brain2text-modeltraining.\n"
        f"Expected path: {_MODELTRAINING}\n"
        f"Original error: {e}\n"
        "Check docks/PINS.txt and ensure the sibling repo is present."
    ) from e

# ── Import-time signature assertions ──────────────────────────────────────
import inspect as _inspect

def _assert_sig(cls, expected_params: list, name: str):
    sig = _inspect.signature(cls.__init__)
    actual = list(sig.parameters.keys())
    for p in expected_params:
        assert p in actual, (
            f"[bit_dock] {name}.__init__ is missing expected param '{p}'. "
            "The upstream interface changed — update the stage adapter."
        )

_assert_sig(BIT_Transformer, ["input_dim", "embed_dim", "num_layers", "patch_size", "session_ids"], "BIT_Transformer")
_assert_sig(MLPProjector,    ["input_dim", "hidden_dim", "output_dim"],                             "MLPProjector")
