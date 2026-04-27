import torch
import torch.nn as nn
import torch.optim as optim
import torch.amp as amp
from torch.utils.data import DataLoader
import os
import sys
import json
import time
import logging
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import h5py
from dataclasses import dataclass, asdict

# Add parent directory to path so we can import src
base_path = Path(__file__).parent.parent
sys.path.insert(0, str(base_path))

from src.utils.decoders import beam_search_decoder
from src.preprocessing.dataloader import BCI_Dataset, Preprocessed_BCI_Dataset, bci_collate_fn, TextTokenizer
from src.models.baseline import BrainToTextModel
from src.utils.metrics import calculate_cer, calculate_wer

@dataclass
class Config:
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    beam_width: int = 10
    alpha: float = 0.5
    num_workers: int = 0  # Set to 0 for stability on Windows
    cache_data: bool = True
    use_preprocessed: bool = True
    preprocessed_path: str = "data/preprocessed_data.h5"
    model_dir: str = "models"
    output_dir: str = "outputs"

# Setup logging
def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'train.log')
    
    # Configure logging to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def greedy_decode(logits, tokenizer):
    """Fast greedy decoder for training-time CER tracking."""
    indices = torch.argmax(logits, dim=-1)
    batch_preds = []
    for i in range(indices.size(0)):
        tokens = indices[i].cpu().tolist()
        decoded_tokens = []
        last_token = None
        for t in tokens:
            if t != 0 and t != last_token:
                decoded_tokens.append(t)
            last_token = t
        batch_preds.append(tokenizer.decode(decoded_tokens))
    return batch_preds

def plot_metrics(history, output_dir):
    """Visualization helper for training and validation metrics."""
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(16, 12))
    
    metrics = [
        ('Loss', 'train_loss', 'val_loss'),
        ('CER', 'train_cer', 'val_cer'),
        ('WER', 'train_cer', 'val_wer'), # Comparing train CER with val WER
        ('Val Metrics', 'val_cer', 'val_wer')
    ]
    
    for i, (title, train_key, val_key) in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        plt.plot(epochs, history[train_key], label=f'Train {train_key.split("_")[1].upper()}')
        plt.plot(epochs, history[val_key], label=f'Val {val_key.split("_")[1].upper()}')
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_metrics.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def validate(model, val_loader, tokenizer, device, ngram_model, criterion, config):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0
    
    with torch.no_grad():
        for neural_inputs, targets, input_lengths, target_lengths in val_loader:
            neural_inputs = neural_inputs.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)
            
            logits = model(neural_inputs)
            
            # Calculate validation loss
            log_probs_loss = logits.transpose(0, 1).log_softmax(2)
            loss = criterion(log_probs_loss, targets, input_lengths, target_lengths)
            total_loss += loss.item()
            
            log_probs = logits.log_softmax(2)
            for i in range(log_probs.size(0)):
                pred_text = beam_search_decoder(
                    log_probs[i], 
                    tokenizer, 
                    ngram_model, 
                    beam_width=config.beam_width, 
                    alpha=config.alpha 
                )
                target_text = tokenizer.decode(targets[i].cpu().tolist())
                all_preds.append(pred_text)
                all_targets.append(target_text)
    
    avg_loss = total_loss / len(val_loader)
    cer = calculate_cer(all_preds, all_targets)
    wer = calculate_wer(all_preds, all_targets)
    return avg_loss, cer, wer

def train():
    config = Config()
    logger = setup_logging(config.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Config: {asdict(config)}")
    
    tokenizer = TextTokenizer()
    ngram_path = base_path / "src" / "utils" / "ngram_3gram.pkl"
    if not ngram_path.exists():
        logger.error(f"N-gram model not found at {ngram_path}")
        return
        
    with open(ngram_path, 'rb') as f:
        ngram_model = pickle.load(f)

    model = BrainToTextModel(num_classes=len(tokenizer.char_to_int)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    # Data Preparation
    h5_list_path = base_path / 'src' / 'utils' / 'h5_list_data.json'
    with open(h5_list_path, 'r') as f:
        h5_files = json.load(f)

    train_pairs, val_pairs = [], []
    for h5_path in h5_files:
        with h5py.File(h5_path, 'r') as h5:
            trials = [(h5_path, trial) for trial in h5.keys()]
            if 'data_train.hdf5' in str(h5_path):
                train_pairs.extend(trials)
            elif 'data_val.hdf5' in str(h5_path):
                val_pairs.extend(trials)

    logger.info(f"Loaded {len(train_pairs)} training trials and {len(val_pairs)} validation trials.")
    session_stats_path = base_path / "src" / "preprocessing" / "session_stats.json"

    # Decide which dataset class to use
    preprocessed_file = base_path / config.preprocessed_path
    if config.use_preprocessed and preprocessed_file.exists():
        logger.info(f"Using preprocessed data from {preprocessed_file}")
        # Construct unique names matching the preprocessing format (filename__trialname)
        train_trial_names = [f"{os.path.basename(p[0])}__{p[1]}" for p in train_pairs]
        val_trial_names = [f"{os.path.basename(p[0])}__{p[1]}" for p in val_pairs]
        
        train_dataset = Preprocessed_BCI_Dataset(
            str(preprocessed_file), train_trial_names, tokenizer, cache_data=config.cache_data
        )
        val_dataset = Preprocessed_BCI_Dataset(
            str(preprocessed_file), val_trial_names, tokenizer, cache_data=config.cache_data
        )
    else:
        if config.use_preprocessed:
            logger.warning(f"Preprocessed file not found at {preprocessed_file}. Falling back to raw data loading.")
        
        train_dataset = BCI_Dataset(train_pairs, str(session_stats_path), tokenizer, cache_data=config.cache_data)
        val_dataset = BCI_Dataset(val_pairs, str(session_stats_path), tokenizer, cache_data=config.cache_data)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, 
        collate_fn=bci_collate_fn, num_workers=config.num_workers, 
        pin_memory=True, persistent_workers=config.num_workers > 0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, 
        collate_fn=bci_collate_fn, num_workers=config.num_workers, 
        pin_memory=True, persistent_workers=config.num_workers > 0
    )

    history = {'train_loss': [], 'val_loss': [], 'train_cer': [], 'val_cer': [], 'val_wer': []}
    best_wer = float('inf')
    start_time = time.time()
    
    model_dir = Path(config.model_dir)
    model_dir.mkdir(exist_ok=True)

    try:
        for epoch in range(config.epochs):
            epoch_start = time.time()
            model.train()
            total_train_loss = 0
            train_preds, train_targets = [], []
            
            for batch_idx, (neural_inputs, targets, input_lengths, target_lengths) in enumerate(train_loader):
                neural_inputs, targets = neural_inputs.to(device), targets.to(device)
                input_lengths, target_lengths = input_lengths.to(device), target_lengths.to(device)
                
                optimizer.zero_grad()
                with torch.autocast(device_type=device.type if device.type != 'cpu' else 'cpu', enabled=device.type != 'cpu'):
                    logits = model(neural_inputs)
                    log_probs = logits.transpose(0, 1).log_softmax(2)
                    loss = criterion(log_probs, targets, input_lengths, target_lengths)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_train_loss += loss.item()
                with torch.no_grad():
                    train_preds.extend(greedy_decode(logits, tokenizer))
                    train_targets.extend([tokenizer.decode(t.cpu().tolist()) for t in targets])

            avg_train_loss = total_train_loss / len(train_loader)
            avg_train_cer = calculate_cer(train_preds, train_targets)
            
            # Validation
            avg_val_loss, avg_val_cer, avg_val_wer = validate(model, val_loader, tokenizer, device, ngram_model, criterion, config)
            
            for key, val in zip(history.keys(), [avg_train_loss, avg_val_loss, avg_train_cer, avg_val_cer, avg_val_wer]):
                history[key].append(val)
            
            scheduler.step(avg_val_wer)
            
            logger.info(
                f"Epoch {epoch+1}/{config.epochs} | "
                f"Loss: {avg_train_loss:.4f}/{avg_val_loss:.4f} | "
                f"CER: {avg_train_cer:.4f}/{avg_val_cer:.4f} | "
                f"WER: {avg_val_wer:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"Time: {(time.time()-epoch_start)/60:.2f}m"
            )
            
            if avg_val_wer < best_wer:
                best_wer = avg_val_wer
                torch.save(model.state_dict(), model_dir / "best_model_wer.pth")
                logger.info(f"*** New Best WER: {best_wer:.4f}! Model saved ***")
            
            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), model_dir / f"checkpoint_epoch_{epoch+1}.pth")
                plot_metrics(history, config.output_dir)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    
    total_time = time.time() - start_time
    logger.info(f"Training Complete! Total Time: {total_time/3600:.2f}h | Best Val WER: {best_wer:.4f}")
    
    # Final saving
    plot_path = plot_metrics(history, config.output_dir)
    with open(os.path.join(config.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f)
    logger.info(f"History and plots saved to {config.output_dir}")

if __name__ == "__main__":
    train()
