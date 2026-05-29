"""
tools/make_toy_dataset.py
-------------------------
Create a 15% toy subset of the full HDF5 dataset for local 6 GB experiments.
Must be run once before any toy training.

Usage:
    python tools/make_toy_dataset.py \\
        --full_path data/data_train.hdf5 \\
        --toy_path  data/toy_train.hdf5 \\
        --fraction  0.15 \\
        --seed      42
"""

import argparse
import random
import h5py
from pathlib import Path


def create_toy(full_path: str, toy_path: str, fraction: float = 0.15, seed: int = 42):
    random.seed(seed)
    with h5py.File(full_path, "r") as f:
        keys  = list(f.keys())
        kept  = random.sample(keys, max(1, int(len(keys) * fraction)))
        with h5py.File(toy_path, "w") as g:
            for k in kept:
                f.copy(k, g)
    print(f"Toy dataset: {len(kept)}/{len(keys)} samples → {toy_path}")
    return toy_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_path", required=True)
    parser.add_argument("--toy_path",  required=True)
    parser.add_argument("--fraction",  type=float, default=0.15)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()
    create_toy(args.full_path, args.toy_path, args.fraction, args.seed)


if __name__ == "__main__":
    main()
