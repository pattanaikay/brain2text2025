import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
from tqdm import tqdm
import json
from pathlib import Path
import sys

# Add parent directory to path so we can import src
base_path = Path(__file__).parent.parent
sys.path.insert(0, str(base_path))

from src.models.encoder import BIT_Transformer
from src.preprocessing.dataloader import Preprocessed_BCI_Dataset, bci_collate_fn
from src.utils.logging_utils import setup_logging

def train_ssl(args):
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir, log_name="ssl_pretrain")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 1. Load Data
    import h5py
    with h5py.File(args.train_h5, 'r') as f:
        train_trials = list(f.keys())
    
    with h5py.File(args.val_h5, 'r') as f:
        val_trials = list(f.keys())

    # Get unique session IDs
    session_ids = set()
    with h5py.File(args.train_h5, 'r') as f:
        for t in train_trials[:100]:
            session_ids.add(str(f[t].attrs.get('session', 'unknown')))
    
    logger.info(f"Detected {len(session_ids)} session IDs")

    train_dataset = Preprocessed_BCI_Dataset(args.train_h5, train_trials)
    val_dataset = Preprocessed_BCI_Dataset(args.val_h5, val_trials)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=bci_collate_fn, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=bci_collate_fn, num_workers=args.num_workers, pin_memory=True)

    # 2. Model
    model = BIT_Transformer(session_ids=list(session_ids)).to(device)
    recon_head = nn.Linear(model.embed_dim, model.input_dim * model.patch_size).to(device)
    
    params = list(model.parameters()) + list(recon_head.parameters())
    optimizer = AdamW(params, lr=args.lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # A100 Optimization: AMP with bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    logger.info(f"Using compute dtype: {compute_dtype}")

    # Resume from checkpoint if exists
    start_epoch = 1
    checkpoint_path = os.path.join(args.output_dir, "checkpoint_ssl_latest.pth")
    if os.path.exists(checkpoint_path):
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        recon_head.load_state_dict(checkpoint['recon_head_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    # 3. Training Loop
    best_loss = float('inf')
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            neural_data = batch['neural'].to(device)
            session_id = batch['session_id']
            
            # Contiguous span masking (ratio=0.5)
            B, T, C = neural_data.shape
            mask = torch.zeros((B, T, 1), device=device, dtype=torch.bool)
            mask_len = int(T * 0.5)
            
            for i in range(B):
                start_idx = torch.randint(0, max(1, T - mask_len + 1), (1,)).item()
                mask[i, start_idx:start_idx + mask_len, :] = True
            
            mask = mask.expand(-1, -1, C)
            masked_data = neural_data.clone()
            masked_data[mask] = 0
            
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=compute_dtype):
                encoded = model(masked_data, session_id=session_id)
                reconstructed = recon_head(encoded)
                
                # Target preparation
                pad_len = (model.patch_size - (T % model.patch_size)) % model.patch_size
                target_data = neural_data
                if pad_len > 0:
                    target_data = torch.nn.functional.pad(target_data, (0, 0, 0, pad_len))
                target_data = target_data.view(B, -1, model.patch_size * C)
                
                loss = nn.MSELoss()(reconstructed, target_data)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch} Avg Loss: {avg_loss:.4f}")
        
        scheduler.step(avg_loss)
        
        # Save checkpoints
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'recon_head_state_dict': recon_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss
        }
        torch.save(checkpoint, checkpoint_path)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_encoder_ssl.pth"))
            logger.info(f"New best SSL model saved at epoch {epoch}")

    logger.info("SSL Pretraining Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_h5", type=str, required=True)
    parser.add_argument("--val_h5", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="scripts/models/ssl")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    train_ssl(args)

