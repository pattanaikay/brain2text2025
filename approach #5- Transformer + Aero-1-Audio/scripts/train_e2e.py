import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
from tqdm import tqdm
import json
import time
import numpy as np

from src.models.baseline import BITModel
from src.preprocessing.dataloader import Preprocessed_BCI_Dataset, bci_collate_fn
from src.utils.metrics import calculate_wer, calculate_cer

def train_e2e(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # 1. Load Data
    import h5py
    with h5py.File(args.train_h5, 'r') as f:
        train_trials = list(f.keys())
    
    with h5py.File(args.val_h5, 'r') as f:
        val_trials = list(f.keys())

    session_ids = set()
    with h5py.File(args.train_h5, 'r') as f:
        for t in train_trials[:100]:
            session_ids.add(str(f[t].attrs.get('session', 'unknown')))
    
    # Load session stats if available
    session_stats = None
    if args.session_stats and os.path.exists(args.session_stats):
        with open(args.session_stats, 'r') as f:
            session_stats = json.load(f)
            # Convert lists back to numpy arrays
            for sid in session_stats:
                session_stats[sid]['mean'] = np.array(session_stats[sid]['mean'])
                session_stats[sid]['std'] = np.array(session_stats[sid]['std'])

    train_dataset = Preprocessed_BCI_Dataset(args.train_h5, train_trials, session_stats=session_stats)
    val_dataset = Preprocessed_BCI_Dataset(args.val_h5, val_trials, session_stats=session_stats)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=bci_collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=bci_collate_fn, num_workers=4)

    # 2. Model
    model = BITModel(session_ids=list(session_ids), quantize=True).to(device)
    
    # Load Pretrained SSL Encoder if available
    if args.ssl_checkpoint and os.path.exists(args.ssl_checkpoint):
        print(f"Loading SSL Pretrained Encoder from {args.ssl_checkpoint}")
        model.neural_encoder.load_state_dict(torch.load(args.ssl_checkpoint, map_location=device))

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # 3. Training Loop
    history = {
        'train_loss': [], 'val_loss': [],
        'train_cer': [], 'val_cer': [],
        'train_wer': [], 'val_wer': []
    }
    
    best_wer = float('inf')
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        total_ce = 0
        total_contrastive = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            neural_data = batch['neural'].to(device)
            labels = batch['text']
            session_id = batch['session_id']
            
            optimizer.zero_grad()
            loss, ce_loss, contrastive_loss = model(neural_data, labels=labels, session_id=session_id)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_contrastive += contrastive_loss.item()
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'ce': f"{ce_loss.item():.4f}",
                'cntr': f"{contrastive_loss.item():.4f}"
            })

        avg_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        
        # Validation
        if epoch % args.val_interval == 0:
            val_wer, val_cer, val_loss_avg = validate(model, val_loader, device)
            print(f"\nValidation Epoch {epoch}: WER={val_wer:.4f}, CER={val_cer:.4f}, Loss={val_loss_avg:.4f}")
            
            history['val_wer'].append(val_wer)
            history['val_cer'].append(val_cer)
            history['val_loss'].append(val_loss_avg)
            
            scheduler.step(val_wer)
            
            if val_wer < best_wer:
                best_wer = val_wer
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model_wer.pth"))
                print("New Best Model Saved!")
            else:
                patience_counter += 1
            
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break
        else:
            # Keep history lengths consistent for plotting
            history['val_loss'].append(history['val_loss'][-1] if history['val_loss'] else 0)
            history['val_wer'].append(history['val_wer'][-1] if history['val_wer'] else 1.0)
            history['val_cer'].append(history['val_cer'][-1] if history['val_cer'] else 1.0)

        # Save History
        with open(os.path.join("outputs", "training_history.json"), 'w') as f:
            json.dump(history, f)
            
        if epoch % 50 == 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pth"))

    print("Training Complete.")

def validate(model, val_loader, device):
    model.eval()
    predictions = []
    targets = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            neural_data = batch['neural'].to(device)
            labels = batch['text']
            session_id = batch['session_id']
            
            loss, _, _ = model(neural_data, labels=labels, session_id=session_id)
            total_loss += loss.item()
            
            preds = model.generate(neural_data, session_id=session_id)
            predictions.extend(preds)
            targets.extend(labels)
            
    wer = calculate_wer(predictions, targets)
    cer = calculate_cer(predictions, targets)
    return wer, cer, total_loss / len(val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_h5", type=str, required=True)
    parser.add_argument("--val_h5", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="scripts/models/e2e")
    parser.add_argument("--ssl_checkpoint", type=str, default="scripts/models/ssl/best_encoder_ssl.pth")
    parser.add_argument("--session_stats", type=str, default=None, help="Path to session_stats.json")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--val_interval", type=int, default=10)
    args = parser.parse_args()
    train_e2e(args)
