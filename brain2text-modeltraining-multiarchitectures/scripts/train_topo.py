"""Thin wrapper: run train_e2e with TopoLoss enabled (Arch-3).

Usage:
    python scripts/train_topo.py --train_h5 data/ --val_h5 data/ \\
        --output_dir outputs/e2e_bit_topo \\
        --topo_weight 0.05 --topo_sigma 1.0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_e2e import train_e2e
import argparse

from src.models.registry import ENCODER_REGISTRY

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with TopoLoss regularisation")
    parser.add_argument("--train_h5", type=str, required=True)
    parser.add_argument("--val_h5", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/e2e_topo")
    parser.add_argument("--pretrained_encoder", type=str, default="outputs/ctc/best_model_per.pth")
    parser.add_argument("--session_stats", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--no_quantize", action="store_true")
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--encoder", choices=list(ENCODER_REGISTRY.keys()), default="bit")
    parser.add_argument("--topo_weight", type=float, default=0.05)
    parser.add_argument("--topo_sigma", type=float, default=1.0)
    parser.add_argument("--aux_loss_weight", type=float, default=0.0)
    parser.add_argument("--llm", choices=["qwen2.5-1.5b", "qwen2-audio-7b", "whisper+qwen", "phi4-mm"],
                        default="qwen2.5-1.5b")
    args = parser.parse_args()
    train_e2e(args)
