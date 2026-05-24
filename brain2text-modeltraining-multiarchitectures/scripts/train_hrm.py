"""HRM training script (Arch-7).

Uses 1-step DEQ gradient — no BPTT between patches.
LR default raised to 5e-4 (noisier gradients in early epochs; see HRM pitfalls §Arch-7).

Usage:
    python scripts/train_hrm.py --train_h5 data/ --val_h5 data/ \\
        --output_dir outputs/e2e_hrm --epochs 150 --lr 5e-4
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_e2e import train_e2e
import argparse

from src.models.registry import ENCODER_REGISTRY

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HRM encoder (DEQ 1-step gradient)")
    parser.add_argument("--train_h5", type=str, required=True)
    parser.add_argument("--val_h5", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/e2e_hrm")
    parser.add_argument("--pretrained_encoder", type=str, default="")
    parser.add_argument("--session_stats", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)   # higher LR for 1-step grad
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--no_quantize", action="store_true")
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--encoder", choices=list(ENCODER_REGISTRY.keys()), default="hrm")
    parser.add_argument("--topo_weight", type=float, default=0.0)
    parser.add_argument("--topo_sigma", type=float, default=1.0)
    parser.add_argument("--aux_loss_weight", type=float, default=0.0)
    parser.add_argument("--llm", choices=["qwen2.5-1.5b", "qwen2-audio-7b", "whisper+qwen", "phi4-mm"],
                        default="qwen2.5-1.5b")
    args = parser.parse_args()
    train_e2e(args)
