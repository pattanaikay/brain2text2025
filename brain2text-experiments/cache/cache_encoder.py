"""
cache/cache_encoder.py
----------------------
F4.1 Speedrun: Freeze BIT encoder and precompute (B, T_patch, 384) outputs
as memory-mapped .npy files. Tracks A/D/E (projector+LLM-only experiments)
read from cache instead of running encoder forward — saves ~80% per-step FLOPs.

Usage:
    python cache/cache_encoder.py \\
        --h5_dir  data/ \\
        --ckpt    ../brain2text-modeltraining/outputs/ctc/best_model_per.pth \\
        --out_dir cache/bit_v<sha>/ \\
        --split   train

After running for both train and val splits, set `use_cache: true` in your
experiment spec and all Track A/D/E experiments will skip the encoder forward.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Add sibling paths ──────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))                  # brain2text-experiments/

from docks.bit_dock    import BIT_Transformer
from docks.multiarch_dock import Preprocessed_BCI_Dataset, bci_collate_fn


def _encoder_sha(ckpt_path: str | None) -> str:
    """Compute a short hash of the encoder checkpoint for cache versioning."""
    if ckpt_path and os.path.exists(ckpt_path):
        sha = hashlib.sha256()
        with open(ckpt_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)
        return sha.hexdigest()[:10]
    return "random"


def cache_split(
    h5_files:   list[str],
    out_dir:    str,
    encoder:    BIT_Transformer,
    device:     torch.device,
    patch_size: int = 4,
    batch_size: int = 8,
    num_workers: int = 2,
    split_name: str = "train",
):
    os.makedirs(out_dir, exist_ok=True)

    dataset    = Preprocessed_BCI_Dataset(h5_files, patch_size=patch_size, augment=False)
    loader     = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=bci_collate_fn, num_workers=num_workers,
                             pin_memory=True)

    # We store one .npy per trial, keyed by trial index (position in dataset).
    # meta.json maps trial_index → {text, session_id, neural_length, T_patch}
    meta = {}
    trial_idx = 0

    encoder.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Caching {split_name}"):
            neural      = batch["neural"].to(device)          # (B, T, 512)
            lengths     = batch["neural_lengths"].to(device)  # (B,)
            session_ids = batch["session_id"]                 # list[str]
            texts       = batch["text"]                       # list[str]

            tokens = encoder(neural, session_id=session_ids,
                              neural_lengths=lengths)         # (B, T_patch, 384)
            tokens_np = tokens.cpu().float().numpy()          # fp32 for memmmap safety

            for i in range(tokens.size(0)):
                T_patch = int(((lengths[i].item() + patch_size - 1) // patch_size))
                fpath   = os.path.join(out_dir, f"{split_name}_{trial_idx:06d}.npy")
                np.save(fpath, tokens_np[i])                  # full (T_patch_max, 384)

                meta[trial_idx] = {
                    "text":          texts[i],
                    "session_id":    session_ids[i],
                    "neural_length": int(lengths[i].item()),
                    "T_patch":       T_patch,
                    "npy_path":      fpath,
                }
                trial_idx += 1

    meta_path = os.path.join(out_dir, f"meta_{split_name}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[cache] Saved {trial_idx} cached trials → {out_dir}")
    print(f"[cache] Meta → {meta_path}")
    return meta_path


def main():
    parser = argparse.ArgumentParser(description="Cache BIT encoder outputs as memmaps")
    parser.add_argument("--h5_dir",     required=True,  help="Path to HDF5 data dir or file")
    parser.add_argument("--ckpt",       default=None,   help="BIT CTC checkpoint path")
    parser.add_argument("--out_dir",    default=None,   help="Output dir (default: cache/bit_v<sha>/)")
    parser.add_argument("--split",      default="train", choices=["train","val","both"])
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers",type=int, default=2)
    parser.add_argument("--input_dim",  type=int, default=512)
    parser.add_argument("--embed_dim",  type=int, default=384)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sha    = _encoder_sha(args.ckpt)

    out_dir = args.out_dir or str(Path(__file__).parent / f"bit_v{sha}")
    os.makedirs(out_dir, exist_ok=True)

    # Write manifest so CachedNeuralDataset can verify it's reading the right cache
    manifest = {
        "encoder_sha":  sha,
        "ckpt_path":    args.ckpt,
        "patch_size":   args.patch_size,
        "embed_dim":    args.embed_dim,
        "input_dim":    args.input_dim,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # Build encoder
    encoder = BIT_Transformer(
        input_dim  = args.input_dim,
        embed_dim  = args.embed_dim,
        patch_size = args.patch_size,
    ).to(device)

    if args.ckpt and os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        sd   = ckpt.get("model_state_dict", ckpt)
        enc_sd = {k.replace("encoder.", "", 1): v
                  for k, v in sd.items() if not k.startswith("head.")}
        encoder.load_state_dict(enc_sd, strict=False)
        print(f"[cache] Loaded encoder from {args.ckpt}")
    else:
        print("[cache] Warning: no checkpoint — using randomly initialised encoder")

    import glob
    def _find_h5(root, pattern):
        if os.path.isfile(root):
            return [root]
        return sorted(glob.glob(os.path.join(root, f"**/{pattern}"), recursive=True))

    splits_to_run = ["train", "val"] if args.split == "both" else [args.split]
    for split in splits_to_run:
        pat  = "data_train.hdf5" if split == "train" else "data_val.hdf5"
        h5s  = _find_h5(args.h5_dir, pat)
        if not h5s:
            print(f"[cache] No files found for split={split}"); continue
        cache_split(h5s, out_dir, encoder, device,
                    patch_size=args.patch_size, batch_size=args.batch_size,
                    num_workers=args.num_workers, split_name=split)

    print(f"\n[cache] Done. Cache dir: {out_dir}")
    print(f"[cache] To use: set `use_cache: true` and `cache_dir: {out_dir}` in your spec.")


if __name__ == "__main__":
    main()
