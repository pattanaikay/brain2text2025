import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from src.dataloader import BCI_Dataset, bci_collate_fn, TextTokenizer
from src.models.baseline import BrainToTextModel
import os
import h5py

# 1. Configuration & Hyperparameters
EPOCHS = 50
BATCH_SIZE = 16 # Adjust if you get OOM
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Setup (Ensure train_pairs and val_pairs are defined elsewhere)
tokenizer = TextTokenizer()
model = BrainToTextModel(num_classes=len(tokenizer.char_to_int)).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)

# AMP Scaler for 6GB VRAM optimization
scaler = GradScaler()

# Create a list of (file_path, trial_name) from your raw data folder
all_train_files = [f for f in os.listdir('data/raw') if 'train' in f]
train_pairs = []
for f in all_train_files:
    with h5py.File(f"data/raw/{f}", 'r') as h5:
        train_pairs.extend([(f"data/raw/{f}", trial) for trial in h5.keys()])

# Instantiate the Factory
train_dataset = BCI_Dataset(
    file_trial_pairs=train_pairs, 
    stats_path='session_stats.json',
    tokenizer=tokenizer # Pass the tokenizer so the dataset can encode text
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    collate_fn=bci_collate_fn,
    num_workers=4,
    pin_memory=True
)

# 3. The Training Loop
def train():
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        
        for batch_idx, (neural_inputs, targets, input_lengths, target_lengths) in enumerate(train_loader):
            neural_inputs = neural_inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            optimizer.zero_grad()

            # --- T-401: Mixed Precision Autocast ---
            with autocast():
                logits = model(neural_inputs)
                # CTC expects (Time, Batch, Classes)
                log_probs = logits.transpose(0, 1).log_softmax(2)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)

            # --- Scaling the loss for AMP ---
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"==> Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        torch.save(model.state_dict(), f"models/checkpoint_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()