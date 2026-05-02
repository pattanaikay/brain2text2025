import os
import torch
import argparse
from tqdm import tqdm
import pandas as pd
import h5py

from src.models.baseline import BITModel
from src.preprocessing.dataloader import Preprocessed_BCI_Dataset, bci_collate_fn
from src.utils.metrics import calculate_wer, calculate_cer
from torch.utils.data import DataLoader

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Data
    with h5py.File(args.test_h5, 'r') as f:
        test_trials = list(f.keys())
        
    session_ids = set()
    with h5py.File(args.test_h5, 'r') as f:
        for t in test_trials[:100]:
            session_ids.add(str(f[t].attrs.get('session', 'unknown')))
            
    dataset = Preprocessed_BCI_Dataset(args.test_h5, test_trials)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=bci_collate_fn)

    # 2. Model
    model = BITModel(session_ids=list(session_ids), quantize=True).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # 3. Inference
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            neural_data = batch['neural'].to(device)
            labels = batch['text']
            session_id = batch['session_id']
            
            preds = model.generate(
                neural_data, 
                session_id=session_id, 
                max_new_tokens=100, 
                top_p=0.9, 
                temperature=0.7
            )
            predictions.extend(preds)
            targets.extend(labels)
            
    # 4. Metrics
    wer = calculate_wer(predictions, targets)
    cer = calculate_cer(predictions, targets)
    
    print(f"\nEvaluation Results:")
    print(f"WER: {wer:.4f}")
    print(f"CER: {cer:.4f}")
    
    # Save Results
    results_df = pd.DataFrame({
        'id': test_trials,
        'target': targets,
        'prediction': predictions
    })
    
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    results_df.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to {args.output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_h5", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="scripts/submissions/evaluation_results.csv")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    evaluate(args)
