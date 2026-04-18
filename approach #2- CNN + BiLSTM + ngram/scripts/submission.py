import sys
from pathlib import Path

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.amp as amp
from torch.utils.data import DataLoader
import os
import json
import h5py

import random

from src.preprocessing.dataloader import BCI_Dataset, bci_collate_fn, test_collate_fn, TextTokenizer
from src.models.baseline import BrainToTextModel
from src.utils.metrics import calculate_cer
from src.utils.decoders import greedy_decoder

# 1. Configuration & Hyperparameters
EPOCHS = 15
BATCH_SIZE = 16 
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = TextTokenizer()
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
test_pairs = []

for h5_path in h5_files:
    # Check for official training files
    if 'data_test.hdf5' in h5_path:
        with h5py.File(h5_path, 'r') as h5:
            for trial in h5.keys():
                trial_obj = h5[trial]
                # Only include actual trial groups that contain neural data
                if isinstance(trial_obj, h5py.Group):
                    if any(key in trial_obj for key in ['input_features', 'tx1', 'neural_features']):
                        test_pairs.append((h5_path, trial))

# Keep all trials (don't deduplicate) but track which trials we've seen
# This preserves all trials while handling duplicate loading
print(f"Loaded {len(test_pairs)} test trials (may include duplicates)")

# Deduplicate to keep only unique (file_path, trial_name) pairs
test_pairs = list(set(test_pairs))
print(f"After deduplication: {len(test_pairs)} unique trials")

# Create a simple mapping from index to trial_name for predictions
unique_ids_map = {i: pair[1] for i, pair in enumerate(test_pairs)}
if not test_pairs:
    raise ValueError("Missing training or validation pairs. Ensure h5_list_data.json contains both 'data_train.hdf5' and 'data_val.hdf5' paths.")

print(f"Loaded {len(test_pairs)} test trials")

# 3. Remove the random.shuffle and 90/10 split code blocks entirely

session_stats_path = r"C:\Projects\Brain2Text2025\brain2text2025\src\preprocessing\session_stats.json"

# 4. Instantiate Datasets using the official pairs directly
test_dataset = BCI_Dataset(file_trial_pairs=test_pairs, stats_path=session_stats_path, tokenizer=tokenizer)

# DataLoaders remain the same
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=test_collate_fn, num_workers=4, pin_memory=True)


def generate_submission(model, test_loader, tokenizer, device, unique_ids_map):
    model.eval()
    results = []

    with torch.no_grad():
        for neural_inputs, trial_names, indices in test_loader:
            logits = model(neural_inputs.to(device))
            
            # Loop through the batch
            for i in range(logits.size(0)):
                pred_text = greedy_decoder(logits[i], tokenizer)
                idx = indices[i].item() if torch.is_tensor(indices[i]) else indices[i]
                unique_id = unique_ids_map[idx]
                results.append({
                    "id": unique_id,
                    "sentence": pred_text
                })
    
    # T-503: Save to CSV
    df = pd.DataFrame(results)
    os.makedirs("submissions", exist_ok=True)
    df.to_csv("submissions/submission.csv", index=False)
    print(f"Submission saved to submissions/submission.csv with {len(df)} trials")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = BrainToTextModel(num_classes=len(tokenizer.char_to_int)).to(device)
    model_path = r"C:\Projects\Brain2Text2025\brain2text2025\scripts\models\best_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))

    generate_submission(model, test_loader, tokenizer, device, unique_ids_map)