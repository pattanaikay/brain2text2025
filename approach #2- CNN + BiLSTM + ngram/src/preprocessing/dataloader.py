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
    def __init__(self, file_trial_pairs, stats_path, tokenizer=None, sigma=1.5, cache_data=False):
        self.file_trial_pairs = file_trial_pairs # List of (file_path, trial_name)
        self.sigma = sigma
        self.tokenizer = tokenizer or TextTokenizer()
        self.cache_data = cache_data
        self.cache = {}
        
        with open(stats_path, 'r') as f:
            raw_stats = json.load(f)
            
        # Pre-process stats into numpy arrays for faster access
        self.stats = {}
        for file_path, stat in raw_stats.items():
            mean = np.array(stat['mean'], dtype=np.float32)
            std = np.array(stat['std'], dtype=np.float32)
            std[std == 0] = 1e-8
            self.stats[file_path] = (mean, std)

    def __getitem__(self, idx):
        if self.cache_data and idx in self.cache:
            return self.cache[idx]

        file_path, trial_name = self.file_trial_pairs[idx]
        
        with h5py.File(file_path, 'r') as f:
            trial_group = f[trial_name]
            
            # Try different possible neural data keys
            if 'input_features' in trial_group:
                data = trial_group['input_features'][:]
            elif 'tx1' in trial_group:
                data = trial_group['tx1'][:]
            elif 'neural_features' in trial_group:
                data = trial_group['neural_features'][:]
            else:
                raise KeyError(f"No recognized neural data key found in {trial_name}. Available keys: {list(trial_group.keys())}")
            
            # Load transcription if available
            try:
                if 'transcription' in trial_group:
                    transcription = trial_group['transcription'][:]
                    if isinstance(transcription, (bytes, np.bytes_)):
                        transcription = transcription.decode('utf-8')
                    elif isinstance(transcription, np.ndarray):
                        if transcription.dtype.kind in ['S', 'U']: # string types
                             transcription = "".join([s.decode('utf-8') if isinstance(s, bytes) else s for s in transcription])
                        else:
                             transcription = "".join([chr(int(c)) for c in transcription])
                else:
                    transcription = ""
            except:
                transcription = ""
            
        # 1. Z-Score Normalization using pre-calculated arrays
        mean, std = self.stats[file_path]
        normalized = (data.astype(np.float32) - mean) / std
        
        # 2. Gaussian Smoothing
        smoothed = gaussian_filter1d(normalized, sigma=self.sigma, axis=0)
        
        # Encode transcription to token indices
        target_tokens = torch.tensor(self.tokenizer.encode(transcription), dtype=torch.long)
        
        result = (torch.from_numpy(smoothed), target_tokens, trial_name, idx)
        
        if self.cache_data:
            self.cache[idx] = result
            
        return result

    def __len__(self):
        return len(self.file_trial_pairs)
    

def bci_collate_fn(batch):
    """
    batch: list of tuples (neural_signal, target_tokens, trial_name, idx)
    neural_signal shape: (Time, 512)
    target_tokens: List[int]
    trial_name: str
    idx: int
    """
    # 1. Separate signals, targets, trial_names, and indices
    neural_signals = [item[0] for item in batch]
    targets = [item[1].clone().detach() if isinstance(item[1], torch.Tensor) else torch.tensor(item[1]) for item in batch]
    trial_names = [item[2] for item in batch]
    indices = [item[3] for item in batch]

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


def test_collate_fn(batch):
    """
    Collate function for test data (no targets).
    batch: list of tuples (neural_signal, target_tokens, trial_name, idx)
    Returns: (neural_signals, trial_names, indices)
    """
    # 1. Separate signals, trial_names, and indices
    neural_signals = [item[0] for item in batch]
    trial_names = [item[2] for item in batch]
    indices = [item[3] for item in batch]

    # 2. Track original lengths for model inference
    input_lengths = torch.tensor([s.size(0) for s in neural_signals], dtype=torch.long)

    # 3. Pad neural signals with zeros
    padded_neural = pad_sequence(neural_signals, batch_first=True, padding_value=0.0)

    return padded_neural, trial_names, indices