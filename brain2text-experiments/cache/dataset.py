"""
cache/dataset.py
----------------
CachedNeuralDataset: reads pre-computed BIT encoder outputs from .npy files.

Used by Track A (probing), Track D (loss ablations), and Track E (projector)
experiments where the encoder is frozen and we don't want to rerun it.

Drop-in replacement for the HDF5 dataset in the training loop:
    dataset = CachedNeuralDataset(cache_dir, split="train")
    loader  = DataLoader(dataset, batch_size=4, collate_fn=cached_collate_fn)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class CachedNeuralDataset(Dataset):
    """
    Reads (T_patch, embed_dim) float32 numpy arrays from a cache directory
    produced by cache_encoder.py.

    Verifies the cache manifest on construction so a stale cache is caught
    immediately rather than silently corrupting experiments.
    """

    def __init__(
        self,
        cache_dir:          str,
        split:              str = "train",
        expected_embed_dim: int = 384,
        expected_patch_size: int = 4,
    ):
        self.cache_dir  = Path(cache_dir)
        self.split      = split

        # ── Verify manifest ────────────────────────────────────────────
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Cache manifest not found at {manifest_path}.\n"
                "Run cache/cache_encoder.py first."
            )
        with open(manifest_path) as f:
            manifest = json.load(f)

        if manifest.get("embed_dim") != expected_embed_dim:
            raise ValueError(
                f"Cache embed_dim={manifest.get('embed_dim')} ≠ "
                f"expected {expected_embed_dim}. Wrong cache directory?"
            )
        if manifest.get("patch_size") != expected_patch_size:
            raise ValueError(
                f"Cache patch_size={manifest.get('patch_size')} ≠ "
                f"expected {expected_patch_size}."
            )
        self.manifest = manifest

        # ── Load meta ─────────────────────────────────────────────────
        meta_path = self.cache_dir / f"meta_{split}.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Meta file not found: {meta_path}.\n"
                f"Run cache_encoder.py with --split {split}."
            )
        with open(meta_path) as f:
            raw = json.load(f)

        # Keys are strings in JSON; convert to int and sort
        self.meta: list[dict] = [raw[k] for k in sorted(raw, key=int)]
        self.embed_dim  = expected_embed_dim
        self.patch_size = expected_patch_size

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx: int) -> dict:
        item      = self.meta[idx]
        npy_path  = item["npy_path"]
        tokens    = np.load(npy_path)                     # (T_patch_max, embed_dim) fp32

        return {
            "neural_tokens":  torch.from_numpy(tokens),   # (T_patch_max, 384)
            "neural_length":  item["neural_length"],       # raw bin count
            "T_patch":        item["T_patch"],             # actual patch count
            "text":           item["text"],
            "session_id":     item["session_id"],
        }


def cached_collate_fn(batch: list[dict]) -> dict:
    """
    Collate a list of CachedNeuralDataset items.
    Pads neural_tokens to the max T_patch in the batch.
    """
    max_T = max(b["neural_tokens"].size(0) for b in batch)
    embed = batch[0]["neural_tokens"].size(1)

    tokens_padded = torch.zeros(len(batch), max_T, embed)
    T_patches     = torch.zeros(len(batch), dtype=torch.long)
    raw_lengths   = torch.zeros(len(batch), dtype=torch.long)

    for i, b in enumerate(batch):
        t = b["neural_tokens"]
        tokens_padded[i, :t.size(0)] = t
        T_patches[i]   = b["T_patch"]
        raw_lengths[i] = b["neural_length"]

    return {
        "neural_tokens":  tokens_padded,   # (B, max_T, 384)
        "T_patches":      T_patches,       # (B,) actual patch counts (for masking)
        "neural_lengths": raw_lengths,     # (B,) raw bin counts
        "text":           [b["text"]       for b in batch],
        "session_id":     [b["session_id"] for b in batch],
    }
