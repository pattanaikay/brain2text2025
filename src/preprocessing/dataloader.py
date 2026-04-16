import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter1d
from torch.nn.utils.rnn import pad_sequence
import json, h5py, numpy as np

class TextTokenizer:
    def __init__(self):
        # 0 is reserved for CTC Blank
        self.chars = " abcdefghijklmnopqrstuvwxyz'" 
        self.char_to_int = {char: i + 1 for i, char in enumerate(self.chars)}
        self.int_to_char = {i + 1: char for i, char in enumerate(self.chars)}
        self.char_to_int['[blank]'] = 0
        self.int_to_char[0] = ''

    def encode(self, text):
        return [self.char_to_int[c] for c in text.lower() if c in self.char_to_int]

    def decode(self, tokens):
        return "".join([self.int_to_char[t] for t in tokens])

class BCI_Dataset(Dataset):
    def __init__(self, file_trial_pairs, stats_path, sigma=1.5):
        self.file_trial_pairs = file_trial_pairs # List of (file_path, trial_name)
        self.sigma = sigma
        
        with open(stats_path, 'r') as f:
            self.stats = json.load(f)

    def __getitem__(self, idx):
        file_path, trial_name = self.file_trial_pairs[idx]
        
        with h5py.File(file_path, 'r') as f:
            data = f[trial_name]['tx1'][:]
            # Convert to string or check labels as needed
            
        # 1. Z-Score Normalization
        mean = np.array(self.stats[file_path]['mean'])
        std = np.array(self.stats[file_path]['std'])
        std[std == 0] = 1e-8 # Prevent division by zero
        
        normalized = (data - mean) / std
        
        # 2. Gaussian Smoothing
        smoothed = gaussian_filter1d(normalized, sigma=self.sigma, axis=0)
        
        return torch.tensor(smoothed, dtype=torch.float32)

    def __len__(self):
        return len(self.file_trial_pairs)
    

def bci_collate_fn(batch):
    """
    batch: list of tuples (neural_signal, target_tokens)
    neural_signal shape: (Time, 512)
    target_tokens: List[int]
    """
    # 1. Separate signals and targets
    neural_signals = [item[0] for item in batch]
    targets = [torch.tensor(item[1]) for item in batch]

    # 2. Track original lengths (Required for CTCLoss)
    # neural_signals are (Time, Features) -> length is at dim 0
    input_lengths = torch.tensor([s.size(0) for s in neural_signals], dtype=torch.long)
    target_lengths = torch.tensor([t.size(0) for t in targets], dtype=torch.long)

    # 3. Pad neural signals with zeros
    # batch_first=True makes the shape (Batch, Max_Time, 512)
    padded_neural = pad_sequence(neural_signals, batch_first=True, padding_value=0.0)

    # 4. Pad target tokens with 0 (or your CTC blank index)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)

    return padded_neural, padded_targets, input_lengths, target_lengths