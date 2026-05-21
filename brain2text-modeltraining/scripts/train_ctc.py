import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
from tqdm import tqdm
from pathlib import Path
import sys
import requests
import json
import numpy as np

# Add parent directory to path so we can import src
base_path = Path(__file__).parent.parent
sys.path.insert(0, str(base_path))

from src.models.encoder import BIT_Transformer
from src.preprocessing.dataloader import Preprocessed_BCI_Dataset, bci_collate_fn
from src.utils.metrics import calculate_per
from src.utils.logging_utils import setup_logging

class CTCPhonemeModel(nn.Module):
    def __init__(self, encoder, num_phonemes=42):
        super().__init__()
        self.encoder = encoder
        # num_phonemes = 42 (0 is BLANK, 1-41 are real phonemes)
        self.head = nn.Linear(encoder.embed_dim, num_phonemes)
        
    def forward(self, x, session_id=None):
        encoded = self.encoder(x, session_id=session_id)
        return self.head(encoded)

def train_ctc(args):
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir, log_name="ctc_finetune")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    import glob
    import h5py
    
    # 1. Discover Data Files
    if os.path.isdir(args.train_h5):
        logger.info(f"Searching for HDF5 files in {args.train_h5}...")
        train_h5_files = sorted(glob.glob(os.path.join(args.train_h5, "**/data_train.hdf5"), recursive=True))
        val_h5_files = sorted(glob.glob(os.path.join(args.val_h5, "**/data_val.hdf5"), recursive=True))
    else:
        train_h5_files = [args.train_h5]
        val_h5_files = [args.val_h5]
        
    # Filter to ensure all discovered paths are actually files
    train_h5_files = [f for f in train_h5_files if os.path.isfile(f)]
    val_h5_files = [f for f in val_h5_files if os.path.isfile(f)]
        
    if not train_h5_files:
        logger.error(f"No valid data_train.hdf5 files found in {args.train_h5}")
        sys.exit(1)
        
    logger.info(f"Found {len(train_h5_files)} training files and {len(val_h5_files)} validation files.")

    # 2. Discover Session IDs (from all training files)
    session_ids = set()
    for path in train_h5_files:
        try:
            with h5py.File(path, 'r') as f:
                trial_keys = list(f.keys())
                if trial_keys:
                    first_trial = trial_keys[0]
                    # Try to get session from attribute, fallback to folder name
                    sid = f[first_trial].attrs.get('session', os.path.basename(os.path.dirname(path)))
                    session_ids.add(str(sid))
        except Exception as e:
            logger.warning(f"Could not read session ID from {path}: {e}")
    
    logger.info(f"Total Unique Sessions identified: {len(session_ids)}")
            
    # Load session stats if provided
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
                    logger.error(f"Format Error: {args.session_stats} contains a list, but a dictionary of stats is expected. Ensure you ran compute_session_stats.py first.")
            except Exception as e:
                logger.error(f"Failed to parse session stats: {e}")
    else:
        logger.warning("No session_stats provided. Training might be unstable due to probe drift.")

    train_dataset = Preprocessed_BCI_Dataset(train_h5_files, session_stats=session_stats, patch_size=args.patch_size, filter_ctc=True)
    val_dataset = Preprocessed_BCI_Dataset(val_h5_files, session_stats=session_stats, patch_size=args.patch_size, filter_ctc=True)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=bci_collate_fn, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=bci_collate_fn, num_workers=args.num_workers, pin_memory=True)

    encoder = BIT_Transformer(session_ids=list(session_ids), patch_size=args.patch_size)
    if args.ssl_checkpoint and os.path.exists(args.ssl_checkpoint):
        logger.info(f"Loading SSL Pretrained Encoder from {args.ssl_checkpoint}")
        state_dict = torch.load(args.ssl_checkpoint, map_location=device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        # Use strict=False to handle session read-in layer mismatches
        encoder.load_state_dict(state_dict, strict=False)
        
    model = CTCPhonemeModel(encoder, num_phonemes=42).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    best_per = float('inf')
    epochs_no_improve = 0
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    logger.info(f"Using compute dtype: {compute_dtype}")

    # Resume from checkpoint if exists
    start_epoch = 1
    checkpoint_path = os.path.join(args.output_dir, "checkpoint_ctc_latest.pth")
    if os.path.exists(checkpoint_path):
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Smart weight loading: filter out keys with size mismatches (like the changed head)
        state_dict = checkpoint['model_state_dict']
        model_state = model.state_dict()
        filtered_state_dict = {}
        mismatch_detected = False
        for k, v in state_dict.items():
            if k in model_state:
                if v.shape == model_state[k].shape:
                    filtered_state_dict[k] = v
                else:
                    logger.warning(f"Skipping key {k} due to size mismatch: {v.shape} vs {model_state[k].shape}")
                    mismatch_detected = True
            else:
                logger.warning(f"Skipping key {k} as it's not in the current model.")

        model.load_state_dict(filtered_state_dict, strict=False)

        if not mismatch_detected:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info("Successfully loaded optimizer and scheduler state.")
            except Exception as e:
                logger.warning(f"Failed to load optimizer state: {e}. Starting with a fresh optimizer.")
        else:
            logger.warning("Architecture mismatch detected (head size change). Skipping optimizer state loading to prevent crash.")

        start_epoch = checkpoint['epoch'] + 1
        if 'best_per' in checkpoint:
            best_per = checkpoint['best_per']
        if 'epochs_no_improve' in checkpoint:
            epochs_no_improve = checkpoint['epochs_no_improve']
    
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            neural_data = batch['neural'].to(device)
            session_id = batch['session_id']
            
            if 'phonemes' in batch:
                labels = batch['phonemes'].to(device)
                target_lengths = batch['phoneme_lengths'].to(device)
            else:
                logger.error("Real phoneme labels missing from batch! Stopping training.")
                sys.exit(1)
            
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=compute_dtype, enabled=args.use_amp):
                logits = model(neural_data, session_id=session_id) # (B, T_patch, 42)
                
                # CTC expects (T, B, C)
                logits = logits.permute(1, 0, 2)
                log_probs = nn.functional.log_softmax(logits, dim=2)
                
                # Correctly compute input lengths based on original neural lengths and patch_size
                neural_lengths = batch['neural_lengths'].to(device)
                input_lengths = (neural_lengths + model.encoder.patch_size - 1) // model.encoder.patch_size
                
                loss = ctc_loss_fn(log_probs, labels, input_lengths, target_lengths)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch} Avg Loss: {avg_loss:.4f}")
        
        if epoch % args.val_interval == 0:
            val_per = validate(model, val_loader, device, args.use_amp, compute_dtype)
            logger.info(f"Validation Epoch {epoch}: PER={val_per:.4f}")
            scheduler.step(val_per)
            
            if val_per < best_per:
                best_per = val_per
                epochs_no_improve = 0
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model_per.pth"))
                logger.info(f"New best CTC model saved at epoch {epoch}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs")
                    break

            # Save checkpoints
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_per': best_per,
                'val_per': val_per,
                'epochs_no_improve': epochs_no_improve
            }
            torch.save(checkpoint, checkpoint_path)

    logger.info("CTC Fine-tuning Complete.")

    # --- AUTO-PAUSE LOGIC ---
    try:
        instance_id = "410271" 
        logger.info(f"Sending pause request for instance {instance_id}...")
        requests.get(f"https://jarvislabs.ai/pause/{instance_id}") 
    except Exception as e:
        logger.error(f"Failed to auto-pause: {e}")

def validate(model, val_loader, device, use_amp, compute_dtype):
    model.eval()
    predictions = []
    targets_list = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            neural_data = batch['neural'].to(device)
            session_id = batch['session_id']
            
            with torch.autocast(device_type='cuda', dtype=compute_dtype, enabled=use_amp):
                logits = model(neural_data, session_id=session_id)
                preds = logits.argmax(dim=-1)
            
            for i in range(preds.size(0)):
                # Standard CTC Greedy Decoding:
                # 1. Collapse adjacent duplicates
                # 2. Remove blanks (index 0)
                raw_preds = preds[i].tolist()
                decoded_seq = []
                prev_token = None
                for p in raw_preds:
                    if p != prev_token:
                        if p != 0: # 0 is blank
                            decoded_seq.append(str(p))
                    prev_token = p
                
                # THE FIX: If empty, use a placeholder that doesn't match common phonemes
                # or just an empty string if calculate_per handles it.
                # Phoneme 1 might be a real phoneme, so using a very large number is safer.
                predictions.append(" ".join(decoded_seq) if decoded_seq else "999")
                
                if 'phonemes' in batch:
                    tgt = batch['phonemes'][i][:batch['phoneme_lengths'][i]]
                    targets_list.append(" ".join([str(t.item()) for t in tgt]))
                else:
                    targets_list.append("999")
                    
    return calculate_per(predictions, targets_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_h5", type=str, required=True)
    parser.add_argument("--val_h5", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="scripts/models/ctc")
    parser.add_argument("--ssl_checkpoint", type=str, default="scripts/models/ssl/best_encoder_ssl.pth")
    parser.add_argument("--session_stats", type=str, default=None, help="Path to session_stats.json")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_interval", type=int, default=10)
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience (validation intervals)")
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=4, help="Patch size for temporal compression")
    args = parser.parse_args()
    train_ctc(args)
