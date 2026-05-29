"""
docks/multiarch_dock.py
-----------------------
Re-exports every architecture from brain2text-modeltraining-multiarchitectures
under stable names. Also re-exports the training dataset and utilities.

Pin: see docks/PINS.txt  (brain2text-modeltraining-multiarch)
"""

import sys
import os
from pathlib import Path

# ── Resolve sibling repo root ──────────────────────────────────────────────
_HERE       = Path(__file__).resolve().parent
_EXPT_ROOT  = _HERE.parent
_REPO_ROOT  = _EXPT_ROOT.parent

_MULTIARCH  = _REPO_ROOT / "brain2text-modeltraining-multiarchitectures"
_MULTIARCH_SRC = _MULTIARCH / "src"

for _p in [str(_MULTIARCH), str(_MULTIARCH_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Encoders ───────────────────────────────────────────────────────────────
try:
    from src.models.architectures.conformer   import ConformerEncoder       # noqa: F401
    from src.models.architectures.hrm         import HRMEncoder             # noqa: F401
    from src.models.architectures.mamba_possm import MambaPOSSMEncoder      # noqa: F401
    from src.models.architectures.moe         import MoEEncoder             # noqa: F401
    from src.models.architectures.zenbrain_memory import ZenBrainEncoder    # noqa: F401
except ImportError as e:
    raise ImportError(
        f"[multiarch_dock] Failed to import encoder from multiarchitectures.\n"
        f"Expected path: {_MULTIARCH}\n"
        f"Original error: {e}"
    ) from e

# ── Audio LLM decoder wrappers ─────────────────────────────────────────────
try:
    from src.models.architectures.audio_llm.qwen2_audio_e2e  import Qwen2AudioE2E    # noqa: F401
    from src.models.architectures.audio_llm.phi4_mm_e2e      import Phi4MME2E        # noqa: F401
    from src.models.architectures.audio_llm.whisper_qwen_e2e import WhisperQwenE2E  # noqa: F401
except ImportError:
    # Audio LLMs may not be installed locally (OOM on 6 GB). Soft-fail.
    Qwen2AudioE2E  = None  # type: ignore
    Phi4MME2E      = None  # type: ignore
    WhisperQwenE2E = None  # type: ignore

# ── TopoLoss ───────────────────────────────────────────────────────────────
try:
    from src.models.architectures.topoloss.topo_loss import TopoLoss                 # noqa: F401
    from src.models.architectures.topoloss.hooks     import collect_ffn_first_linears # noqa: F401
except ImportError:
    TopoLoss                 = None  # type: ignore
    collect_ffn_first_linears = None # type: ignore

# ── Dataset & utilities ────────────────────────────────────────────────────
try:
    from src.preprocessing.dataloader import (                               # noqa: F401
        Preprocessed_BCI_Dataset,
        bci_collate_fn,
    )
    from src.utils.metrics import calculate_wer, calculate_cer               # noqa: F401
    from src.utils.logging_utils import setup_logging                        # noqa: F401
except ImportError as e:
    raise ImportError(
        f"[multiarch_dock] Failed to import dataset/utils.\n"
        f"Original error: {e}"
    ) from e

# ── Registry (full encoder map) ────────────────────────────────────────────
try:
    from src.models.registry import ENCODER_REGISTRY, build_encoder          # noqa: F401
except ImportError as e:
    raise ImportError(
        f"[multiarch_dock] Failed to import encoder registry.\n"
        f"Original error: {e}"
    ) from e
