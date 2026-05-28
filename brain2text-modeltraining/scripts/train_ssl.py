import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
from tqdm import tqdm
import json
import numpy as np
from pathlib import Path
import sys
import h5py
import requests

# Add parent directory to path so we can import src
base_path = Path(__file__).parent.parent
sys.path.insert(0, str(base_path))

from src.models.encoder import BIT_Transformer
from src.preprocessing.dataloader import Preprocessed_BCI_Dataset, bci_collate_fn
from src.utils.logging_utils import setup_logging


def compute_ssl_loss(model, recon_head, batch, device, compute_dtype):
    """Compute masked-patch MSE loss for one batch. Returns (loss, patch_mask)."""
    neural_data = batch['neural'].to(device)            # (B, T, C)
    neural_lengths = batch['neural_lengths'].to(device) # (B,)
    session_id = batch['session_id']
    B, T, C = neural_data.shape

    # Compute patched lengths to build a valid-patch mask (don't mask padding).
    patched_lengths = (neural_lengths + model.patch_size - 1) // model.patch_size
    pad_len = (model.patch_size - (T % model.patch_size)) % model.patch_size
    T_patch = (T + pad_len) // model.patch_size

    # Build per-sample patch mask with variable ratio 0.3–0.7 of contiguous spans,
    # only over valid (non-padded) patches.
    patch_mask = torch.zeros(B, T_patch, dtype=torch.bool, device=device)
    for i in range(B):
        valid_len = patched_lengths[i].item()
        if valid_len < 2:
            continue
        ratio = float(torch.empty(1).uniform_(0.3, 0.7))
        n_mask = max(1, int(valid_len * ratio))
        n_mask = min(n_mask, valid_len - 1)
        start = int(torch.randint(0, valid_len - n_mask + 1, (1,)))
        patch_mask[i, start:start + n_mask] = True

    with torch.autocast(device_type='cuda', dtype=compute_dtype):
        encoded = model(neural_data, session_id=session_id,
                        mask_patches=patch_mask, neural_lengths=neural_lengths)
        reconstructed = recon_head(encoded)   # (B, T_patch, patch_size * C)

        # Target = original neural data in patch space
        target_data = neural_data
        if pad_len > 0:
            target_data = torch.nn.functional.pad(target_data, (0, 0, 0, pad_len))
        target_data = target_data.view(B, T_patch, model.patch_size * C)

        # MSE loss ONLY on masked patch positions, normalised by # masked elements
        mask_expanded = patch_mask.unsqueeze(-1).float()
        mse = (reconstructed - target_data) ** 2
        n_masked = mask_expanded.sum() * model.patch_size * C
        loss = (mse * mask_expanded).sum() / (n_masked + 1e-8)

    return loss


def train_ssl(args):
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir, log_name="ssl_pretrain")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 1. Multi-file discovery (mirrors train_ctc.py)
    if os.path.isdir(args.train_h5):
        train_h5_files = sorted(glob.glob(os.path.join(args.train_h5, "**/data_train.hdf5"), recursive=True))
        val_h5_files = sorted(glob.glob(os.path.join(args.val_h5, "**/data_val.hdf5"), recursive=True))
    else:
        train_h5_files = [args.train_h5]
        val_h5_files = [args.val_h5]
    train_h5_files = [f for f in train_h5_files if os.path.isfile(f)]
    val_h5_files = [f for f in val_h5_files if os.path.isfile(f)]
    if not train_h5_files:
        logger.error(f"No data_train.hdf5 files in {args.train_h5}")
        sys.exit(1)
    logger.info(f"Found {len(train_h5_files)} train files, {len(val_h5_files)} val files.")

    # Discover all session IDs across all files
    session_ids = set()
    for path in train_h5_files:
        try:
            with h5py.File(path, 'r') as f:
                trial_keys = list(f.keys())
                if trial_keys:
                    sid = f[trial_keys[0]].attrs.get('session', os.path.basename(os.path.dirname(path)))
                    session_ids.add(str(sid))
        except Exception as e:
            logger.warning(f"Could not read session ID from {path}: {e}")
    logger.info(f"Total sessions: {len(session_ids)}")

    # Load session stats for z-score normalization (same as CTC/E2E stages)
    session_stats = None
    if args.session_stats and os.path.exists(args.session_stats):
        logger.info(f"Loading session stats from {args.session_stats}")
        with open(args.session_stats, 'r') as f:
            try:
                data = json.load(f)
                if isinstance(data, dict):
                    session_stats = data
                    for sid in session_stats:
                        session_stats[sid]['mean'] = np.array(session_stats[sid]['mean'])
                        session_stats[sid]['std'] = np.array(session_stats[sid]['std'])
                else:
                    logger.error("Format Error: session_stats must be a dict keyed by session ID.")
            except Exception as e:
                logger.error(f"Failed to parse session stats: {e}")
    else:
        logger.warning("No --session_stats provided. SSL will train on un-normalized data.")

    train_dataset = Preprocessed_BCI_Dataset(train_h5_files, session_stats=session_stats, patch_size=args.patch_size, augment=True)
    val_dataset = Preprocessed_BCI_Dataset(val_h5_files, session_stats=session_stats, patch_size=args.patch_size, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               collate_fn=bci_collate_fn, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=bci_collate_fn, num_workers=args.num_workers, pin_memory=True)

    # 2. Model
    model = BIT_Transformer(session_ids=list(session_ids), patch_size=args.patch_size).to(device)
    recon_head = nn.Linear(model.embed_dim, model.input_dim * model.patch_size).to(device)

    params = list(model.parameters()) + list(recon_head.parameters())
    optimizer = AdamW(params, lr=args.lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # A100 Optimization: AMP with bfloat16
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    logger.info(f"Using compute dtype: {compute_dtype}")

    # Resume from checkpoint if exists
    start_epoch = 1
    checkpoint_path = os.path.join(args.output_dir, "checkpoint_ssl_latest.pth")
    if os.path.exists(checkpoint_path):
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        recon_head.load_state_dict(checkpoint['recon_head_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    # 3. Training Loop
    best_loss = float('inf')
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        recon_head.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            loss = compute_ssl_loss(model, recon_head, batch, device, compute_dtype)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch} train_loss: {avg_loss:.4f}")

        # Validation (every epoch)
        model.eval()
        recon_head.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                loss = compute_ssl_loss(model, recon_head, batch, device, compute_dtype)
                val_loss += loss.item()
                val_batches += 1
        val_loss /= max(1, val_batches)
        logger.info(f"Epoch {epoch} val_loss: {val_loss:.4f}")

        scheduler.step(val_loss)

        # Save checkpoints
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'recon_head_state_dict': recon_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
            'val_loss': val_loss,
        }
        torch.save(checkpoint, checkpoint_path)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_encoder_ssl.pth"))
            logger.info(f"New best SSL model saved at epoch {epoch} (val_loss={val_loss:.4f})")

    logger.info("SSL Pretraining Complete.")

    # Auto-pause JarvisLabs instance when done (skip if called from run_pipeline.py)
    if not args.no_autopause:
        import subprocess as _sp
        instance_id = args.instance_id if hasattr(args, 'instance_id') and args.instance_id else "415608"
        logger.info(f"Auto-pausing instance {instance_id} via jl CLI...")
        try:
            r = _sp.run(["jl", "pause", instance_id, "--yes", "--json"],
                        capture_output=True, text=True, timeout=60)
            if r.returncode == 0:
                logger.info("Instance paused successfully.")
            else:
                logger.warning(f"jl pause failed (code {r.returncode}): {r.stderr.strip() or r.stdout.strip()}")
        except Exception as e:
            logger.error(f"Failed to auto-pause: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_h5", type=str, required=True)
    parser.add_argument("--val_h5", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="scripts/models/ssl")
    parser.add_argument("--session_stats", type=str, default=None, help="Path to session_stats.json for z-score normalization")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=4, help="Patch size for temporal compression")
    parser.add_argument("--no_autopause", action="store_true", help="Disable auto-pause (use when called from run_pipeline.py)")
    args = parser.parse_args()
    train_ssl(args)
