"""
docks/dietcorp_dock.py
----------------------
Isolation dock for github.com/ebrahimfeghhi/transformers_with_dietcorp.

DietCorp ≈ corrupted-input regularization + per-day/per-session affine
recalibration for neural decoding on small data. The strategy here (ADHD
convergence 2026-05-30, frame: faithful-upstream) is:

    Vendor the upstream repo UNTOUCHED under docks/dietcorp_upstream/, and put
    ALL the seam-cutting in this one file. The smoke runs the ORIGINAL training
    loop for a couple iterations so you can see exactly which seams to cut for
    full integration later, instead of clean-room reimplementing and drifting.

Nothing here imports torch-heavy upstream code at module load — imports are
lazy and guarded, so the 25 existing experiments' CI stays green even when
the dietcorp deps / submodule are absent.

Vendoring:
    git submodule add https://github.com/ebrahimfeghhi/transformers_with_dietcorp \\
        brain2text-experiments/docks/dietcorp_upstream
    # then record the SHA in docks/PINS.txt (see dietcorp line there).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

_HERE     = Path(__file__).resolve().parent
_UPSTREAM = _HERE / "dietcorp_upstream"


class DietCorpUnavailable(ImportError):
    """Raised on first real use when the vendored repo / deps are missing."""


def _require_upstream():
    """Lazily make the vendored repo importable; raise a friendly error if not."""
    if not _UPSTREAM.exists() or not any(_UPSTREAM.iterdir()):
        raise DietCorpUnavailable(
            "docks/dietcorp_upstream/ is empty.\n"
            "Vendor it first:\n"
            "  git submodule add "
            "https://github.com/ebrahimfeghhi/transformers_with_dietcorp "
            "brain2text-experiments/docks/dietcorp_upstream\n"
            "then pin its SHA in docks/PINS.txt."
        )
    if str(_UPSTREAM) not in sys.path:
        sys.path.insert(0, str(_UPSTREAM))


# ── Provenance ───────────────────────────────────────────────────────────────

def upstream_hash() -> str:
    """Recursive content hash of the vendored repo (for PINS.txt drift checks)."""
    _require_upstream()
    h = hashlib.sha256()
    for p in sorted(_UPSTREAM.rglob("*.py")):
        h.update(p.relative_to(_UPSTREAM).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


# ── Toy data adapter (the seam) ──────────────────────────────────────────────

def get_toy_batches(source: str = "upstream", n_batches: int = 4,
                    batch_size: int = 4):
    """
    Yield toy batches for a smoke run.

    source="upstream": use the repo's OWN shipped toy/sample tensors. This is
        the FAITHFUL reproduction path — losses can be checked against the
        paper oracle. (TODO: locate the upstream sample loader and wire it.)
    source="ours": wrap our (B,T,512) neural tensors reshaped to the repo's
        speech-BCI input layout. Flagged experimental — shape mapping + a
        synthesised day-index are required and are NOT yet faithful.

    Until the upstream is vendored this returns random-tensor stand-ins so the
    wiring (dataloader → model → loss) is exercisable offline.
    """
    import torch
    if source == "upstream":
        try:
            _require_upstream()
            # TODO(seam-1): replace with the upstream repo's shipped toy loader,
            # e.g. `from dietcorp.data import sample_batches`. Random for now.
        except DietCorpUnavailable:
            pass
    # Placeholder batches — shape mimics a speech-BCI neural window + day index.
    for _ in range(n_batches):
        yield {
            "neural":    torch.randn(batch_size, 240, 512),
            "day_index": torch.randint(0, 8, (batch_size,)),
            "targets":   torch.randint(0, 41, (batch_size, 20)),
        }


def run_smoke_iters(n: int = 2, source: str = "upstream") -> dict:
    """
    Drive the upstream training loop for `n` iterations and return the losses.

    Until the repo is vendored, this runs the native recalibration projector
    (stages/projector/dietcorp_recal.py) on placeholder batches so the wiring
    is provable. Returns {"losses": [...], "diverged": bool|None}.
    """
    import torch
    from stages.projector.dietcorp_recal import build as recal_build

    proj, _ = recal_build({"input_dim": 512, "output_dim": 384, "n_days": 8},
                          (240, 512))
    opt = torch.optim.Adam(proj.parameters(), lr=1e-3)

    losses = []
    for batch in get_toy_batches(source=source, n_batches=n):
        opt.zero_grad()
        out = proj(batch["neural"], day_index=batch["day_index"])
        # Stand-in objective: reconstruct the mean — proves day-index routes
        # and gradients flow. Real objective comes from upstream loop later.
        loss = (out.mean(1).pow(2).mean())
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return {"losses": losses, "diverged": None}
