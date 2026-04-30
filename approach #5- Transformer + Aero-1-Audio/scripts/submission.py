import torch
import os
from src.models.baseline import BrainToTextModel
from src.preprocessing.dataloader import Preprocessed_BCI_Dataset, bci_collate_fn
from torch.utils.data import DataLoader
import pandas as pd

def generate_submission(model_path, test_h5_path, test_trials, output_csv):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BrainToTextModel(quantize=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    dataset = Preprocessed_BCI_Dataset(test_h5_path, test_trials)
    loader = DataLoader(dataset, batch_size=4, collate_fn=bci_collate_fn)
    
    predictions = []
    with torch.no_grad():
        for batch in loader:
            neural_data = batch['neural'].to(device)
            preds = model.generate(neural_data)
            predictions.extend(preds)
            
    df = pd.DataFrame({'unique_id': test_trials, 'prediction': predictions})
    df.to_csv(output_csv, index=False)
    print(f"Submission saved to {output_csv}")
