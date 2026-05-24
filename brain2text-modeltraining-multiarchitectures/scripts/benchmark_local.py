"""Local encoder benchmark on synthetic BCI data — no GPU, no LLM, no HDF5 needed.

Trains every encoder in ENCODER_REGISTRY with a CTC phoneme head for N epochs
on in-memory random data.  Generates a multi-panel comparison figure so you can
see relative convergence and throughput before scaling to a cluster.

Why CTC-only:
  The LLM (Qwen2.5-1.5B) requires CUDA + bitsandbytes to run efficiently.
  CTC training tests the encoder's full forward/backward path with the same
  patch-space shapes it will use during E2E training — it's the right sanity
  check before committing GPU hours.

Usage:
  py -3.11 scripts/benchmark_local.py
  py -3.11 scripts/benchmark_local.py --epochs 30 --n_train 120 --n_val 40 --batch_size 4
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models.registry import ENCODER_REGISTRY


# ── Synthetic dataset ──────────────────────────────────────────────────────────

class SyntheticBCIDataset(Dataset):
    """Random BCI-like data generated once at init (reproducible with seed).

    Each sample:
      neural  : (T_bins, 512)  float32  Gaussian noise ~ N(0,1)
      phonemes: (n_phon,)      int64    random from [1, 41]

    Constraint: n_phon < T_patch = ceil(T_bins / patch_size)  — required for CTC.
    """

    def __init__(self, n_samples=100, patch_size=4, seed=42):
        rng = torch.Generator()
        rng.manual_seed(seed)
        self.patch_size = patch_size
        self.samples = []
        for _ in range(n_samples):
            # Variable-length sequences (80–200 bins) for realistic padding coverage
            T_bins = int(torch.randint(80, 201, (1,), generator=rng).item())
            T_patch = (T_bins + patch_size - 1) // patch_size
            # Phoneme length must be < T_patch for CTC validity
            max_phon = max(1, T_patch - 2)
            n_phon = int(torch.randint(4, min(max_phon, 15) + 1, (1,), generator=rng).item())
            neural = torch.randn(T_bins, 512, generator=rng)
            phonemes = torch.randint(1, 42, (n_phon,), generator=rng)
            self.samples.append((neural, phonemes, T_bins))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        neural, phonemes, T_bins = self.samples[idx]
        return {"neural": neural, "phonemes": phonemes, "T_bins": T_bins}


def collate_fn(batch, patch_size=4):
    """Pad neural to same length; stack phonemes with lengths."""
    max_T = max(s["neural"].shape[0] for s in batch)
    # Pad to multiple of patch_size
    max_T_pad = ((max_T + patch_size - 1) // patch_size) * patch_size

    neural_padded = torch.zeros(len(batch), max_T_pad, 512)
    neural_lengths = torch.zeros(len(batch), dtype=torch.long)
    max_phon = max(s["phonemes"].shape[0] for s in batch)
    phonemes_padded = torch.zeros(len(batch), max_phon, dtype=torch.long)
    phoneme_lengths = torch.zeros(len(batch), dtype=torch.long)

    for i, s in enumerate(batch):
        T = s["neural"].shape[0]
        neural_padded[i, :T] = s["neural"]
        neural_lengths[i] = T
        n = s["phonemes"].shape[0]
        phonemes_padded[i, :n] = s["phonemes"]
        phoneme_lengths[i] = n

    return {
        "neural": neural_padded,
        "neural_lengths": neural_lengths,
        "phonemes": phonemes_padded,
        "phoneme_lengths": phoneme_lengths,
    }


# ── CTC model wrapper ──────────────────────────────────────────────────────────

class CTCModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.embed_dim, 42)
        self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    def forward(self, neural, neural_lengths, phonemes, phoneme_lengths):
        encoded = self.encoder(neural, neural_lengths=neural_lengths)   # (B, T_patch, 384)
        logits = self.head(encoded)                                     # (B, T_patch, 42)
        log_probs = F.log_softmax(logits.float(), dim=-1).permute(1, 0, 2)  # (T, B, 42)
        patch_size = self.encoder.patch_size
        patched_lengths = (neural_lengths + patch_size - 1) // patch_size
        loss = self.ctc_loss(log_probs, phonemes, patched_lengths.cpu(), phoneme_lengths.cpu())
        return loss


# ── Training loop ──────────────────────────────────────────────────────────────

def run_one_encoder(name, train_loader, val_loader, args, device):
    print(f"\n{'='*60}", flush=True)
    print(f"  Training encoder: {name}", flush=True)
    print(f"{'='*60}", flush=True)

    enc_cls = ENCODER_REGISTRY[name]
    # HRM is patch-sequential (GRUCell per bin) — limit fixed-point iters on CPU
    kwargs = {}
    if name == "hrm" and str(device) == "cpu":
        import src.models.architectures.hrm.deq as _deq
        _orig = _deq.fixed_point_solver
        def _fast_solver(f, h0, tol=1e-3, max_iter=10):
            return _orig(f, h0, tol=tol, max_iter=min(max_iter, 3))
        _deq.fixed_point_solver = _fast_solver   # patch to max 3 iters on CPU

    encoder = enc_cls(session_ids=["synth"], patch_size=args.patch_size, **kwargs)
    n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}", flush=True)

    model = CTCModel(encoder).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_losses, val_losses = [], []

    # Throughput measurement on first batch
    with torch.no_grad():
        first_batch = next(iter(train_loader))
        fbn = first_batch["neural"].to(device)
        fbl = first_batch["neural_lengths"].to(device)
        fbp = first_batch["phonemes"].to(device)
        fbpl = first_batch["phoneme_lengths"].to(device)
        # Warmup
        _ = model(fbn, fbl, fbp, fbpl)
        t0 = time.perf_counter()
        for _ in range(5):
            _ = model(fbn, fbl, fbp, fbpl)
        elapsed = (time.perf_counter() - t0) / 5
    throughput = len(fbn) / elapsed  # samples/sec

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            neural = batch["neural"].to(device)
            neural_lengths = batch["neural_lengths"].to(device)
            phonemes = batch["phonemes"].to(device)
            phoneme_lengths = batch["phoneme_lengths"].to(device)
            optimizer.zero_grad()
            loss = model(neural, neural_lengths, phonemes, phoneme_lengths)
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                neural = batch["neural"].to(device)
                neural_lengths = batch["neural_lengths"].to(device)
                phonemes = batch["phonemes"].to(device)
                phoneme_lengths = batch["phoneme_lengths"].to(device)
                loss = model(neural, neural_lengths, phonemes, phoneme_lengths)
                if torch.isfinite(loss):
                    val_loss += loss.item()
        avg_val_loss = val_loss / max(1, len(val_loader))
        val_losses.append(avg_val_loss)

        if epoch % max(1, args.epochs // 5) == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs}  "
                  f"train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}", flush=True)

    return {
        "name": name,
        "n_params": n_params,
        "throughput_samples_per_sec": round(throughput, 1),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "final_val_loss": val_losses[-1],
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_results(results, output_dir, epochs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np
    except ImportError:
        print("matplotlib not installed — skipping plots. pip install matplotlib")
        return None

    os.makedirs(output_dir, exist_ok=True)
    names = [r["name"] for r in results]
    colors = cm.tab10(np.linspace(0, 1, len(names)))
    ep = list(range(1, epochs + 1))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Multi-Architecture Encoder Benchmark (CTC, Synthetic BCI Data)", fontsize=13)

    # 1. Train loss curves
    ax = axes[0, 0]
    for r, c in zip(results, colors):
        ax.plot(ep, r["train_losses"], label=r["name"], color=c)
    ax.set_title("Train CTC Loss vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CTC Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)

    # 2. Validation loss curves
    ax = axes[0, 1]
    for r, c in zip(results, colors):
        ax.plot(ep, r["val_losses"], label=r["name"], color=c, linestyle="--")
    ax.set_title("Val CTC Loss vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CTC Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)

    # 3. Throughput bar chart
    ax = axes[1, 0]
    thr = [r["throughput_samples_per_sec"] for r in results]
    bars = ax.bar(names, thr, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title("Inference Throughput (samples/sec, CPU)")
    ax.set_ylabel("Samples / second")
    ax.set_xticklabels(names, rotation=20, ha="right")
    for bar, v in zip(bars, thr):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax.grid(True, axis="y", alpha=0.4)

    # 4. Parameter count bar chart
    ax = axes[1, 1]
    params_m = [r["n_params"] / 1e6 for r in results]
    bars = ax.bar(names, params_m, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_title("Encoder Parameter Count (M)")
    ax.set_ylabel("Parameters (millions)")
    ax.set_xticklabels(names, rotation=20, ha="right")
    for bar, v in zip(bars, params_m):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.1f}M", ha="center", va="bottom", fontsize=8)
    ax.grid(True, axis="y", alpha=0.4)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "encoder_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved -> {plot_path}")
    return plot_path


def print_summary_table(results):
    print("\n" + "=" * 72)
    print(f"{'Encoder':<14} {'Params':>10} {'Throughput':>16} {'Final Val Loss':>15}")
    print("-" * 72)
    for r in sorted(results, key=lambda x: x["final_val_loss"]):
        print(f"{r['name']:<14} {r['n_params']/1e6:>9.1f}M "
              f"{r['throughput_samples_per_sec']:>14.1f}/s "
              f"{r['final_val_loss']:>15.4f}")
    print("=" * 72)


# ── Entry point ────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Encoders to benchmark: {enc_names if args.encoders else 'all'}", flush=True)
    print(f"Epochs: {args.epochs}  |  Batch: {args.batch_size}  |  "
          f"Train: {args.n_train}  |  Val: {args.n_val}", flush=True)

    _collate = lambda b: collate_fn(b, patch_size=args.patch_size)

    train_ds = SyntheticBCIDataset(args.n_train, patch_size=args.patch_size, seed=42)
    val_ds   = SyntheticBCIDataset(args.n_val,   patch_size=args.patch_size, seed=99)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=_collate)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=_collate)

    enc_names = args.encoders if args.encoders else list(ENCODER_REGISTRY.keys())
    unknown = [n for n in enc_names if n not in ENCODER_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown encoder(s): {unknown}. Available: {list(ENCODER_REGISTRY)}")

    all_results = []
    for name in enc_names:
        try:
            result = run_one_encoder(name, train_loader, val_loader, args, device)
            all_results.append(result)
        except Exception as e:
            import traceback
            print(f"\n[ERROR] {name} failed: {e}")
            traceback.print_exc()

    # Save JSON
    out_json = os.path.join(args.output_dir, "benchmark_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results saved -> {out_json}")

    print_summary_table(all_results)
    plot_results(all_results, args.output_dir, args.epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Local encoder benchmark — no GPU, no LLM, no HDF5 required"
    )
    parser.add_argument("--epochs", type=int, default=20,
                        help="Training epochs per encoder (default 20, ~2 min on CPU)")
    parser.add_argument("--n_train", type=int, default=120,
                        help="Number of synthetic training samples")
    parser.add_argument("--n_val", type=int, default=40,
                        help="Number of synthetic validation samples")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size (keep small for CPU, 4 recommended)")
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (1e-3 works well for fast toy convergence)")
    parser.add_argument("--output_dir", type=str, default="outputs/benchmark")
    parser.add_argument("--encoders", nargs="*", default=None,
                        help="Subset of encoders to run (default: all). E.g. --encoders bit conformer moe")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU even if CUDA is available")
    args = parser.parse_args()
    main(args)
