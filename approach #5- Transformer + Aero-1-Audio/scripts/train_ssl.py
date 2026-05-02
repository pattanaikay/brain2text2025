import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
from tqdm import tqdm
import json

from src.models.encoder import BIT_Transformer
from src.preprocessing.dataloader import Preprocessed_BCI_Dataset, bci_collate_fn

def train_ssl(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Data
    # Assuming trials are passed or loaded from a file
    # For now, I'll assume we need to find them in the HDF5
    import h5py
    with h5py.File(args.train_h5, 'r') as f:
        train_trials = list(f.keys())
    
    with h5py.File(args.val_h5, 'r') as f:
        val_trials = list(f.keys())

    # Get unique session IDs for subject-specific layers
    session_ids = set()
    with h5py.File(args.train_h5, 'r') as f:
        for t in train_trials[:100]: # Sample some to get session IDs, or read metadata
            session_ids.add(str(f[t].attrs.get('session', 'unknown')))
    
    train_dataset = Preprocessed_BCI_Dataset(args.train_h5, train_trials)
    val_dataset = Preprocessed_BCI_Dataset(args.val_h5, val_trials)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=bci_collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=bci_collate_fn, num_workers=4)

    # 2. Model
    model = BIT_Transformer(session_ids=list(session_ids)).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # 3. Training Loop
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            neural_data = batch['neural'].to(device)
            session_id = batch['session_id']
            
            # Masked Modeling
            # Mask continuous spans of time patches
            # After grouping, we have T_patch. Let's mask at the raw bin level for simplicity or patch level.
            # The blueprint says "Mask continuous spans of the Time Patches".
            # So I should apply masking AFTER the patch embedding? 
            # Or mask the raw features and then patch them.
            
            # Let's mask the raw features for simplicity, but in blocks.
            mask = torch.rand_like(neural_data) < 0.15
            masked_data = neural_data.clone()
            masked_data[mask] = 0
            
            optimizer.zero_grad()
            # Forward pass
            # We want to reconstruct the masked parts
            # But BIT_Transformer outputs embeddings.
            # For SSL pretraining, we usually add a reconstruction head.
            # I'll add a simple linear head for reconstruction.
            
            # Temporary reconstruction head
            recon_head = nn.Linear(model.embed_dim, model.input_dim * model.patch_size).to(device)
            
            encoded = model(masked_data, session_id=session_id) # (B, T_patch, 384)
            reconstructed = recon_head(encoded) # (B, T_patch, 512 * 5)
            
            # Target is the original data, reshaped to patches
            batch_size, time_steps, channels = neural_data.shape
            pad_len = (model.patch_size - (time_steps % model.patch_size)) % model.patch_size
            target_data = neural_data
            if pad_len > 0:
                target_data = torch.nn.functional.pad(target_data, (0, 0, 0, pad_len))
            target_data = target_data.view(batch_size, -1, model.patch_size * channels)
            
            loss = nn.MSELoss()(reconstructed, target_data)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} Avg Loss: {avg_loss:.4f}")
        
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_encoder_ssl.pth"))

    print("SSL Pretraining Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_h5", type=str, required=True)
    parser.add_argument("--val_h5", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="scripts/models/ssl")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    train_ssl(args)
