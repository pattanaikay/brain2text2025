import sys
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import json
import h5py
import pickle

# Add parent directory to path so we can import src
base_path = Path(__file__).parent.parent
sys.path.insert(0, str(base_path))

from src.preprocessing.dataloader import BCI_Dataset, test_collate_fn, TextTokenizer
from src.models.baseline import BrainToTextModel
from src.utils.decoders import beam_search_decoder

# 1. Configuration & Hyperparameters
BATCH_SIZE = 16 

def generate_submission(model, test_loader, tokenizer, device, unique_ids_map, ngram_model):
    model.eval()
    results = []

    print(f"Generating submission entries...")
    with torch.no_grad():
        for batch_idx, (neural_inputs, trial_names, indices) in enumerate(test_loader):
            logits = model(neural_inputs.to(device))
            
            # Loop through the batch
            for i in range(logits.size(0)):
                pred_text = beam_search_decoder(logits[i], tokenizer, ngram_model)
                idx = indices[i].item() if torch.is_tensor(indices[i]) else indices[i]
                unique_id = unique_ids_map[idx]
                results.append({
                    "id": unique_id,
                    "sentence": pred_text
                })
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(test_loader)} processed.")
    
    # Save to CSV
    df = pd.DataFrame(results)
    output_dir = base_path / "scripts" / "submissions"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "submission.csv"
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} with {len(df)} trials")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = TextTokenizer()

    # Data Preparation - Load from h5_list_data.json
    H5_LIST_FILE = base_path / 'src' / 'utils' / 'h5_list_data.json'

    if not H5_LIST_FILE.exists():
        raise FileNotFoundError(f"h5_list_data.json not found at {H5_LIST_FILE}.")

    with open(H5_LIST_FILE, 'r') as f:
        h5_files = json.load(f)
    
    test_pairs = []
    for h5_path in h5_files:
        if 'data_test.hdf5' in str(h5_path):
            session_name = os.path.basename(os.path.dirname(h5_path))
            with h5py.File(h5_path, 'r') as h5:
                for trial in h5.keys():
                    trial_obj = h5[trial]
                    if isinstance(trial_obj, h5py.Group):
                        # Use a tuple that includes session for sorting and uniqueness
                        test_pairs.append((str(h5_path), trial, session_name))

    # Sort and deduplicate to ensure deterministic order
    test_pairs = sorted(list(set(test_pairs)))
    print(f"Loaded {len(test_pairs)} unique test trials")

    # Create unique IDs and update test_pairs to (h5_path, trial) for the Dataset
    unique_ids_map = {}
    dataset_pairs = []
    for i, (h5_path, trial, session) in enumerate(test_pairs):
        unique_id = f"{session}_{trial}"
        unique_ids_map[i] = unique_id
        dataset_pairs.append((h5_path, trial))
    
    if not dataset_pairs:
        raise ValueError("No test trials found in data_test.hdf5 files.")

    session_stats_path = base_path / "src" / "preprocessing" / "session_stats.json"

    # Instantiate Dataset & DataLoader
    test_dataset = BCI_Dataset(file_trial_pairs=dataset_pairs, stats_path=str(session_stats_path), tokenizer=tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=test_collate_fn, num_workers=0, pin_memory=True)

    # Load the model
    model = BrainToTextModel(num_classes=len(tokenizer.char_to_int)).to(device)
    
    # Try different possible locations for the best model
    model_path = base_path / "scripts" / "models" / "best_model_wer.pth"
    if not model_path.exists():
        model_path = base_path / "models" / "best_model_wer.pth"
    
    if not model_path.exists():
        # Fallback to old name if it exists
        old_model_path = base_path / "scripts" / "models" / "best_model.pth"
        if old_model_path.exists():
            model_path = old_model_path
        else:
            raise FileNotFoundError(f"Best model not found at {model_path}")

    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    # Load the N-gram model
    ngram_path = base_path / 'src' / 'utils' / 'ngram_3gram.pkl'
    if not ngram_path.exists():
        raise FileNotFoundError(f"N-gram model not found at {ngram_path}")
    with open(ngram_path, 'rb') as f:
        ngram_model = pickle.load(f)

    generate_submission(model, test_loader, tokenizer, device, unique_ids_map, ngram_model)
