import torch
import torch.nn as nn
import torch.optim as optim
import torch.amp as amp
from torch.utils.data import DataLoader, Subset
import os
import sys
import json
from pathlib import Path
import pickle  # For loading the trained N-gram model

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.decoders import beam_search_decoder 
from src.preprocessing.dataloader import BCI_Dataset, bci_collate_fn, TextTokenizer
from src.models.baseline import BrainToTextModel
from src.utils.metrics import calculate_cer
import h5py

# 1. Configuration & Hyperparameters
EPOCHS = 20
BATCH_SIZE = 16 
LEARNING_RATE = 1e-4

# 2. Validation Function
def validate(model, val_loader, tokenizer, device, ngram_model):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for neural_inputs, targets, _, _ in val_loader:
            logits = model(neural_inputs.to(device))
            log_probs = logits.log_softmax(2)
            
            for i in range(log_probs.size(0)):
                # Using your new beam search decoder
                pred_text = beam_search_decoder(
                    log_probs[i], 
                    tokenizer, 
                    ngram_model, 
                    beam_width=10, 
                    alpha=0.5 
                )
                
                target_text = tokenizer.decode(targets[i].tolist())
                all_preds.append(pred_text)
                all_targets.append(target_text)
    
    cer = calculate_cer(all_preds, all_targets)
    return cer

# 3. The Training Loop
def train():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup
    tokenizer = TextTokenizer()
    file_path = r"C:\Projects\Brain2Text2025\brain2text2025\approach #2- CNN + BiLSTM + ngram\src\utils\ngram_3gram.pkl"

    # Load the n-gram model
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"N-gram model not found at {file_path}")
        
    with open(file_path, 'rb') as f:
        ngram_model = pickle.load(f)

    # Model and Optimizer setup
    model = BrainToTextModel(num_classes=len(tokenizer.char_to_int)).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    # Data Preparation - Load from h5_list_data.json
    H5_LIST_FILE = os.path.join(Path(__file__).parent.parent, 'src', 'utils', 'h5_list_data.json')

    if not os.path.exists(H5_LIST_FILE):
        raise FileNotFoundError(f"h5_list_data.json not found at {H5_LIST_FILE}. Please run trainingdata_list.py first.")

    with open(H5_LIST_FILE, 'r') as f:
        h5_files = json.load(f)

    train_pairs = []
    val_pairs = []

    for h5_path in h5_files:
        if 'data_train.hdf5' in h5_path:
            with h5py.File(h5_path, 'r') as h5:
                train_pairs.extend([(h5_path, trial) for trial in h5.keys()])
        elif 'data_val.hdf5' in h5_path:
            with h5py.File(h5_path, 'r') as h5:
                val_pairs.extend([(h5_path, trial) for trial in h5.keys()])

    if not train_pairs or not val_pairs:
        raise ValueError("Missing training or validation pairs.")

    print(f"Loaded {len(train_pairs)} training trials and {len(val_pairs)} validation trials.")

    session_stats_path = r"C:\Projects\Brain2Text2025\brain2text2025\approach #2- CNN + BiLSTM + ngram\src\preprocessing\session_stats.json"

    # Instantiate Datasets
    train_dataset = BCI_Dataset(file_trial_pairs=train_pairs, stats_path=session_stats_path, tokenizer=tokenizer)
    val_dataset = BCI_Dataset(file_trial_pairs=val_pairs, stats_path=session_stats_path, tokenizer=tokenizer)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=bci_collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=bci_collate_fn, num_workers=4, pin_memory=True)

    best_cer = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, (neural_inputs, targets, input_lengths, target_lengths) in enumerate(train_loader):
            neural_inputs = neural_inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            optimizer.zero_grad()

            with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                logits = model(neural_inputs)
                log_probs = logits.transpose(0, 1).log_softmax(2)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"==> Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")
        
        # Validation
        val_cer = validate(model, val_loader, tokenizer, DEVICE, ngram_model)
        print(f"Epoch {epoch+1} | Validation CER: {val_cer:.4f}")
        
        if val_cer < best_cer:
            best_cer = val_cer
            if not os.path.exists('models'): os.makedirs('models')
            torch.save(model.state_dict(), "models/best_model.pth")
            print(f"*** New Best CER! Model saved ***")
        
        torch.save(model.state_dict(), f"models/checkpoint_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()