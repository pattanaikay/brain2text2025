"""
stack.py
--------
The Stack runtime — an OrderedDict of named stages that form the full
brain-to-text pipeline:

    neural_in → encoder → projector → decoder → (loss hooks applied externally)

Each stage file exports:
    build(spec: dict, prev_shape: tuple) -> (nn.Module, next_shape: tuple)

Shape is a tuple of ints representing the non-batch dimensions, e.g.:
    encoder output  : (T_patch, 384)
    projector output: (T_out, llm_dim)
    decoder         : (vocab_size,)   [logit space; actual training uses embed injection]

Shape contracts are checked at build time so projector mismatches surface
before any GPU time is spent.
"""

from __future__ import annotations

import importlib
import json
import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# ── Stage registry ─────────────────────────────────────────────────────────
# Maps (stage_kind, variant_name) → dotted module path inside stages/
_STAGE_MODULES = {
    # Encoders
    ("encoder", "bit"):       "stages.encoder.bit",
    ("encoder", "conformer"): "stages.encoder.conformer",
    ("encoder", "hrm"):       "stages.encoder.hrm",
    ("encoder", "mamba"):     "stages.encoder.mamba",
    ("encoder", "moe"):       "stages.encoder.moe",
    ("encoder", "zenbrain"):  "stages.encoder.zenbrain",
    # Projectors
    ("projector", "mlp"):      "stages.projector.mlp",
    ("projector", "deep_mlp"): "stages.projector.deep_mlp",
    ("projector", "gated"):    "stages.projector.gated",
    ("projector", "qformer"):  "stages.projector.qformer",
    # Decoders
    ("decoder", "qwen15"):       "stages.decoder.qwen15",
    ("decoder", "qwen2_audio"):  "stages.decoder.qwen2_audio",
    ("decoder", "phi4mm"):       "stages.decoder.phi4mm",
    ("decoder", "whisper_qwen"): "stages.decoder.whisper_qwen",
    # Losses — built separately via compose.py; registered here for introspection
    ("loss", "ce"):            "stages.loss.ce",
    ("loss", "ctc_anneal"):    "stages.loss.ctc_anneal",
    ("loss", "contrastive"):   "stages.loss.contrastive",
    ("loss", "topo"):          "stages.loss.topo",
    ("loss", "label_smooth"):  "stages.loss.label_smooth",
}


def _import_stage_module(kind: str, variant: str):
    key = (kind, variant)
    if key not in _STAGE_MODULES:
        available = [v for k, v in _STAGE_MODULES.items() if k[0] == kind]
        raise ValueError(
            f"Unknown {kind} variant '{variant}'. "
            f"Available {kind}s: {available}"
        )
    mod_path = _STAGE_MODULES[key]
    return importlib.import_module(mod_path)


class Stack(nn.Module):
    """
    Ordered pipeline of named stages.  Each stage is an nn.Module produced
    by its build() function.  Forward() is intentionally NOT defined here —
    the decoder stage owns the training forward pass.  Use stack.encoder,
    stack.projector, stack.decoder directly.

    Usage
    -----
    spec = {
        "encoder":   {"variant": "bit",    "patch_size": 4, ...},
        "projector": {"variant": "qformer","n_queries": 32,  ...},
        "decoder":   {"variant": "qwen15", "quantize": True,  ...},
    }
    stack = Stack.from_spec(spec)
    stack.encoder(neural_data, session_id=..., neural_lengths=...)
    """

    def __init__(self):
        super().__init__()
        self._stage_order: list[str] = []
        self._shapes: dict[str, tuple] = {}

    # ── Build from spec dict ───────────────────────────────────────────────
    @classmethod
    def from_spec(cls, spec: dict, prev_shape: tuple = (240, 512)) -> "Stack":
        """
        Build the stack from a spec dict.  prev_shape is the raw neural input
        shape (T_bins, n_channels) before any stage.

        Validates that each stage's output shape is compatible with the
        next stage's expected input.
        """
        stack = cls()
        current_shape = prev_shape

        stage_order = ["encoder", "projector", "decoder"]
        for stage_kind in stage_order:
            if stage_kind not in spec:
                continue
            stage_spec = dict(spec[stage_kind])         # copy so we can pop 'variant'
            variant = stage_spec.pop("variant")
            mod = _import_stage_module(stage_kind, variant)
            module, out_shape = mod.build(stage_spec, current_shape)
            stack.add_module(stage_kind, module)
            stack._stage_order.append(stage_kind)
            stack._shapes[stage_kind] = out_shape
            current_shape = out_shape

        return stack

    # ── Shape introspection ────────────────────────────────────────────────
    def output_shape(self, stage: str) -> tuple:
        return self._shapes[stage]

    # ── Spec hash (for cache keying / run.lock) ───────────────────────────
    @staticmethod
    def spec_hash(spec: dict) -> str:
        canonical = json.dumps(spec, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    # ── Convenience accessors ──────────────────────────────────────────────
    @property
    def encoder(self) -> nn.Module:
        return self._modules["encoder"]

    @property
    def projector(self) -> nn.Module:
        return self._modules["projector"]

    @property
    def decoder(self) -> nn.Module:
        return self._modules["decoder"]

    def has_stage(self, name: str) -> bool:
        return name in self._modules

    def __repr__(self):
        lines = ["Stack("]
        for name in self._stage_order:
            shape = self._shapes.get(name, "?")
            lines.append(f"  [{name}] → {shape}  {type(self._modules[name]).__name__}")
        lines.append(")")
        return "\n".join(lines)
