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
from pathlib import Path
import sys
import requests
import glob
import traceback

# Add parent directory to path so we can import src
base_path = Path(__file__).parent.parent
sys.path.insert(0, str(base_path))

from src.models.baseline import BITModel
from src.preprocessing.dataloader import Preprocessed_BCI_Dataset, bci_collate_fn
from src.utils.metrics import calculate_wer, calculate_cer
from src.utils.logging_utils import setup_logging

def pause_instance(logger, instance_id="410414"):
    """
    Sends a request to JarvisLabs to pause the instance.
    """
    try:
        logger.info(f"Sending pause request for instance {instance_id}...")
        # Note: In a real environment, this URL might need to be called from outside 
        # or via a specific JarvisLabs CLI. This is the implementation based on user requirement.
        requests.get(f"https://jarvislabs.ai/pause/{instance_id}")
    except Exception as e:
        logger.error(f"Failed to auto-pause: {e}")

def train_e2e(args):
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir, log_name="e2e_train")
    
    try:
        _train_logic(args, logger)
        logger.info("Training Complete.")
    except Exception as e:
        logger.error(f"TRAINING FATAL ERROR: {e}")
        logger.error(traceback.format_exc())
    finally:
        # ALWAYS pause, whether success or failure
        pause_instance(logger)

def _train_logic(args, logger):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    import h5py
    
    # 1. Discover Data Files
    if os.path.isdir(args.train_h5):
        logger.info(f"Searching for HDF5 files in {args.train_h5}...")
        train_h5_files = sorted(glob.glob(os.path.join(args.train_h5, "**/data_train.hdf5"), recursive=True))
        val_h5_files = sorted(glob.glob(os.path.join(args.val_h5, "**/data_val.hdf5"), recursive=True))
    else:
        train_h5_files = [args.train_h5]
        val_h5_files = [args.val_h5]
        
    train_h5_files = [f for f in train_h5_files if os.path.isfile(f)]
    val_h5_files = [f for f in val_h5_files if os.path.isfile(f)]
        
    if not train_h5_files:
        logger.error(f"No valid data_train.hdf5 files found in {args.train_h5}")
        return
        
    logger.info(f"Found {len(train_h5_files)} training files and {len(val_h5_files)} validation files.")

    # 2. Discover Session IDs
    session_ids = set()
    for path in train_h5_files:
        try:
            with h5py.File(path, 'r') as f:
                trial_keys = list(f.keys())
                if trial_keys:
                    first_trial = trial_keys[0]
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
                    logger.error(f"Format Error: {args.session_stats} contains a list, but a dictionary of stats is expected.")
            except Exception as e:
                logger.error(f"Failed to parse session stats: {e}")

    train_dataset = Preprocessed_BCI_Dataset(train_h5_files, session_stats=session_stats)
    val_dataset = Preprocessed_BCI_Dataset(val_h5_files, session_stats=session_stats)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=bci_collate_fn, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=bci_collate_fn, num_workers=args.num_workers, pin_memory=True)

    # 3. Model
    model = BITModel(session_ids=list(session_ids), quantize=True).to(device)
    
    # Load Pretrained SSL/CTC Encoder if available
    if args.ssl_checkpoint and os.path.exists(args.ssl_checkpoint):
        logger.info(f"Loading Pretrained Encoder from {args.ssl_checkpoint}")
        checkpoint_data = torch.load(args.ssl_checkpoint, map_location=device)
        
        state_dict = checkpoint_data.get('model_state_dict', checkpoint_data)
            
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('encoder.'):
                new_state_dict[k.replace('encoder.', '')] = v
            else:
                new_state_dict[k] = v
        
        # Use strict=False to handle session read-in mismatches gracefully
        model.neural_encoder.load_state_dict(new_state_dict, strict=False)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    logger.info(f"Using compute dtype: {compute_dtype}")

    # Resume from checkpoint if exists
    start_epoch = 1
    best_wer = float('inf')
    checkpoint_path = os.path.join(args.output_dir, "checkpoint_e2e_latest.pth")
    history = {
        'train_loss': [], 'val_loss': [],
        'train_cer': [], 'val_cer': [],
        'train_wer': [], 'val_wer': []
    }
    
    if os.path.exists(checkpoint_path):
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Smart weight loading for E2E
        state_dict = checkpoint['model_state_dict']
        model_state = model.state_dict()
        filtered_state_dict = {}
        mismatch_detected = False
        for k, v in state_dict.items():
            if k in model_state:
                if v.shape == model_state[k].shape:
                    filtered_state_dict[k] = v
                else:
                    logger.warning(f"Skipping key {k} due to size mismatch.")
                    mismatch_detected = True
            else:
                logger.warning(f"Skipping key {k} as it's not in the current model.")
        
        model.load_state_dict(filtered_state_dict, strict=False)
        
        if not mismatch_detected:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                logger.warning(f"Failed to load optimizer state: {e}")
        
        start_epoch = checkpoint['epoch'] + 1
        best_wer = checkpoint.get('best_wer', float('inf'))
        if 'history' in checkpoint:
            history = checkpoint['history']

    # 4. Training Loop
    patience_counter = 0
    accumulation_steps = args.accumulation_steps
    
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0
        total_ce = 0
        total_contrastive = 0
        
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for i, batch in enumerate(pbar):
            neural_data = batch['neural'].to(device)
            neural_lengths = batch['neural_lengths'].to(device)
            labels = batch['text']
            session_id = batch['session_id']
            
            with torch.autocast(device_type='cuda', dtype=compute_dtype):
                loss, ce_loss, contrastive_loss = model(neural_data, labels=labels, session_id=session_id, neural_lengths=neural_lengths)
                # Scale loss for accumulation
                loss = loss / accumulation_steps
            
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            total_ce += ce_loss.item()
            total_contrastive += contrastive_loss.item()
            
            pbar.set_postfix({
                'loss': f"{loss.item() * accumulation_steps:.4f}",
                'ce': f"{ce_loss.item():.4f}",
                'cntr': f"{contrastive_loss.item():.4f}"
            })

        avg_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        logger.info(f"Epoch {epoch} Avg Loss: {avg_loss:.4f} (CE: {total_ce/len(train_loader):.4f}, Cntr: {total_contrastive/len(train_loader):.4f})")
        
        # Validation
        if epoch % args.val_interval == 0:
            val_wer, val_cer, val_loss_avg = validate(model, val_loader, device, compute_dtype)
            logger.info(f"Validation Epoch {epoch}: WER={val_wer:.4f}, CER={val_cer:.4f}, Loss={val_loss_avg:.4f}")
            
            history['val_wer'].append(val_wer)
            history['val_cer'].append(val_cer)
            history['val_loss'].append(val_loss_avg)
            
            scheduler.step(val_wer)
            
            # Save Checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_wer': best_wer,
                'history': history
            }
            torch.save(checkpoint, checkpoint_path)
            
            if val_wer < best_wer:
                best_wer = val_wer
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model_wer.pth"))
                logger.info("New Best Model Saved!")
            else:
                patience_counter += 1
            
            if patience_counter >= args.patience:
                logger.info("Early stopping triggered.")
                break
        else:
            if history['val_loss']:
                history['val_loss'].append(history['val_loss'][-1])
                history['val_wer'].append(history['val_wer'][-1])
                history['val_cer'].append(history['val_cer'][-1])

        # Save History JSON
        with open(os.path.join(args.output_dir, "training_history.json"), 'w') as f:
            json.dump(history, f)
            
        if epoch % 50 == 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pth"))

def validate(model, val_loader, device, compute_dtype):
    model.eval()
    predictions = []
    targets = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            neural_data = batch['neural'].to(device)
            neural_lengths = batch['neural_lengths'].to(device)
            labels = batch['text']
            session_id = batch['session_id']
            
            with torch.autocast(device_type='cuda', dtype=compute_dtype):
                loss, _, _ = model(neural_data, labels=labels, session_id=session_id, neural_lengths=neural_lengths)
            total_loss += loss.item()
            
            preds = model.generate(neural_data, session_id=session_id, neural_lengths=neural_lengths)
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
    parser.add_argument("--ssl_checkpoint", type=str, default="scripts/models/ctc/best_model_per.pth")
    parser.add_argument("--session_stats", type=str, default=None, help="Path to session_stats.json")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--val_interval", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Number of steps to accumulate gradients")
    args = parser.parse_args()
    train_e2e(args)
