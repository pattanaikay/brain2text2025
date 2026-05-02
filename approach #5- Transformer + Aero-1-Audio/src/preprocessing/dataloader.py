import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d

class Preprocessed_BCI_Dataset(Dataset):
    def __init__(self, h5_path, trial_list, session_stats=None, sigma=1.5):
        """
        Args:
            h5_path: Path to the HDF5 file.
            trial_list: List of trial names.
            session_stats: Dictionary containing 'mean' and 'std' for each session.
                           If None, Z-scoring is not applied here (assuming pre-applied or handled elsewhere).
            sigma: Standard deviation for Gaussian smoothing.
        """
        self.h5_path = h5_path
        self.trial_list = trial_list
        self.session_stats = session_stats
        self.sigma = sigma
        
    def __len__(self):
        return len(self.trial_list)
        
    def __getitem__(self, idx):
        trial_name = self.trial_list[idx]
        with h5py.File(self.h5_path, 'r') as f:
            # The keys might vary based on HDF5 structure. 
            # In the provided neuraldata_viz.py, it looks for 'input_features', etc.
            # I'll use a generic way or assume 'neural' and 'text' as in previous dataloader.py
            group = f[trial_name]
            
            # Find neural data key
            neural_key = None
            for key in ['neural', 'input_features', 'tx1', 'spikePow']:
                if key in group:
                    neural_key = key
                    break
            
            if neural_key is None:
                raise KeyError(f"No neural data found in trial {trial_name}")
                
            neural_data = group[neural_key][:] # (Time, 512)
            
            # Find text key
            text_key = None
            for key in ['text', 'sentenceText', 'transcription']:
                if key in group:
                    text_key = key
                    break
            
            if text_key is None:
                text = ""
            else:
                raw_text = group[text_key][()]
                if isinstance(raw_text, bytes):
                    text = raw_text.decode('utf-8').strip()
                elif isinstance(raw_text, np.ndarray):
                    # Might be character indices
                    text = str(raw_text) # Placeholder for more complex decoding if needed
                else:
                    text = str(raw_text).strip()
            
            session_id = group.attrs.get('session', 'unknown')

        # 1. Z-score normalization (Per-session)
        if self.session_stats and session_id in self.session_stats:
            mean = self.session_stats[session_id]['mean']
            std = self.session_stats[session_id]['std']
            neural_data = (neural_data - mean) / (std + 1e-8)
        
        # 2. Gaussian Smoothing
        if self.sigma > 0:
            neural_data = gaussian_filter1d(neural_data, sigma=self.sigma, axis=0)
            
        return {
            'neural': torch.tensor(neural_data, dtype=torch.float32),
            'text': text,
            'session_id': str(session_id)
        }

def bci_collate_fn(batch):
    """
    Collate function that handles variable length neural sequences.
    Pads to the maximum length in the batch.
    """
    neural_data = [item['neural'] for item in batch]
    texts = [item['text'] for item in batch]
    session_ids = [item['session_id'] for item in batch]
    
    # Pad neural data
    # (Time, Channels) -> (Batch, Max_Time, Channels)
    max_len = max(x.size(0) for x in neural_data)
    channels = neural_data[0].size(1)
    
    padded_neural = torch.zeros(len(batch), max_len, channels)
    for i, x in enumerate(neural_data):
        padded_neural[i, :x.size(0), :] = x
        
    return {
        'neural': padded_neural,
        'text': texts,
        'session_id': session_ids
    }
