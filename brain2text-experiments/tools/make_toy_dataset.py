"""
tools/make_toy_dataset.py
-------------------------
Create toy subsets of HDF5 datasets for local experiments.

Usage (single file, split into train/val):
    python tools/make_toy_dataset.py \\
        --full_path C:\Projects\Brain2Text2025\brain2text2025\brain2text-modeltraining\data\preprocessed_data.h5 \\
        --toy_train data/toy_train.hdf5 \\
        --toy_val   data/toy_val.hdf5 \\
        --fraction  0.15 \\
        --train_ratio 0.8 \\
        --seed      42

Usage (separate files):
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
    """Create a single toy dataset from full_path."""
    random.seed(seed)
    # Ensure output directory exists
    Path(toy_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(full_path, "r") as f:
        keys  = list(f.keys())
        kept  = random.sample(keys, max(1, int(len(keys) * fraction)))
        with h5py.File(toy_path, "w") as g:
            for k in kept:
                f.copy(k, g)
    print(f"Toy dataset: {len(kept)}/{len(keys)} samples → {toy_path}")
    return toy_path


def create_toy_split(full_path: str, toy_train_path: str, toy_val_path: str, 
                     fraction: float = 0.15, train_ratio: float = 0.8, seed: int = 42):
    """Create toy train and val datasets from a single full dataset, split by train_ratio."""
    random.seed(seed)
    # Ensure output directories exist
    Path(toy_train_path).parent.mkdir(parents=True, exist_ok=True)
    Path(toy_val_path).parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(full_path, "r") as f:
        keys = list(f.keys())
        # Sample fraction of total
        sampled = random.sample(keys, max(1, int(len(keys) * fraction)))
        # Split sampled into train/val
        split_idx = int(len(sampled) * train_ratio)
        train_keys = sampled[:split_idx]
        val_keys = sampled[split_idx:]
        
        # Write train
        with h5py.File(toy_train_path, "w") as g:
            for k in train_keys:
                f.copy(k, g)
        print(f"Toy train: {len(train_keys)}/{len(keys)} samples → {toy_train_path}")
        
        # Write val
        with h5py.File(toy_val_path, "w") as g:
            for k in val_keys:
                f.copy(k, g)
        print(f"Toy val: {len(val_keys)}/{len(keys)} samples → {toy_val_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_path",   required=True)
    parser.add_argument("--toy_path",    default=None)      # for single output
    parser.add_argument("--toy_train",   default=None)      # for split outputs
    parser.add_argument("--toy_val",     default=None)      # for split outputs
    parser.add_argument("--fraction",    type=float, default=0.15)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()
    
    if args.toy_train and args.toy_val:
        # Split mode: create both train and val from single file
        create_toy_split(args.full_path, args.toy_train, args.toy_val, 
                        args.fraction, args.train_ratio, args.seed)
    elif args.toy_path:
        # Single mode: create one toy file
        create_toy(args.full_path, args.toy_path, args.fraction, args.seed)
    else:
        print("Error: specify either --toy_path (single) or both --toy_train and --toy_val (split)")


if __name__ == "__main__":
    main()

