import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import os
from scipy.ndimage import gaussian_filter1d

class Preprocessed_BCI_Dataset(Dataset):
    def __init__(self, h5_paths, trial_list=None, session_stats=None, sigma=1.5):
        """
        Args:
            h5_paths: Single path (str) or List of paths to HDF5 files.
            trial_list: Optional list of specific trials. If None, uses all trials in all files.
                        Expected format if provided: list of (h5_path, trial_name) tuples.
            session_stats: Dictionary containing 'mean' and 'std' for each session.
            sigma: Standard deviation for Gaussian smoothing.
        """
        if isinstance(h5_paths, str):
            self.h5_paths = [h5_paths]
        else:
            self.h5_paths = h5_paths
            
        self.session_stats = session_stats
        self.sigma = sigma
        
        # Build index of all trials across all files
        self.all_trials = []
        if trial_list:
            # If trial_list is provided as just names, we assume they belong to the first (only) h5 path
            if isinstance(trial_list[0], str):
                self.all_trials = [(self.h5_paths[0], t) for t in trial_list]
            else:
                self.all_trials = trial_list
        else:
            for path in self.h5_paths:
                with h5py.File(path, 'r') as f:
                    for trial_name in f.keys():
                        self.all_trials.append((path, trial_name))
        
    def __len__(self):
        return len(self.all_trials)
        
    def __getitem__(self, idx):
        h5_path, trial_name = self.all_trials[idx]
        with h5py.File(h5_path, 'r') as f:
            group = f[trial_name]
            
            # Find neural data key
            neural_key = None
            for key in ['neural', 'input_features', 'tx1', 'spikePow']:
                if key in group:
                    neural_key = key
                    break
            
            if neural_key is None:
                raise KeyError(f"No neural data found in trial {trial_name} of {h5_path}")
                
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
                else:
                    text = str(raw_text).strip()
            
            # Find phonemes (Phase 1.5 CTC labels)
            phonemes = None
            phoneme_len = 0
            for key in ['phonemes', 'seq_class_ids', 'phonemeLabels']:
                if key in group:
                    phonemes = group[key][:]
                    phoneme_len = len(phonemes)
                    break
            
            # Try to get explicit phoneme length if available
            if 'seq_len' in group.attrs:
                phoneme_len = int(group.attrs['seq_len'])
            elif 'phoneme_lengths' in group:
                phoneme_len = int(group['phoneme_lengths'][()])

            session_id = group.attrs.get('session', os.path.basename(os.path.dirname(h5_path)))

        # 1. Z-score normalization (Per-session)
        if self.session_stats and session_id in self.session_stats:
            mean = self.session_stats[session_id]['mean']
            std = self.session_stats[session_id]['std']
            neural_data = (neural_data - mean) / (std + 1e-8)
        
        # 2. Gaussian Smoothing
        if self.sigma > 0:
            neural_data = gaussian_filter1d(neural_data, sigma=self.sigma, axis=0)
            
        sample = {
            'neural': torch.tensor(neural_data, dtype=torch.float32),
            'text': text,
            'session_id': str(session_id)
        }
        
        if phonemes is not None:
            # Shift IDs by +1 so 0 is reserved for CTC Blank
            phonemes = phonemes + 1
            sample['phonemes'] = torch.tensor(phonemes, dtype=torch.long)
            sample['phoneme_lengths'] = phoneme_len
            
        return sample

def bci_collate_fn(batch):
    """
    Collate function that handles variable length neural and phoneme sequences.
    Pads to the maximum length in the batch.
    """
    neural_data = [item['neural'] for item in batch]
    texts = [item['text'] for item in batch]
    session_ids = [item['session_id'] for item in batch]
    
    # 1. Pad neural data
    neural_lengths = torch.tensor([x.size(0) for x in neural_data], dtype=torch.long)
    max_neural_len = max(neural_lengths).item()
    channels = neural_data[0].size(1)
    
    padded_neural = torch.zeros(len(batch), max_neural_len, channels)
    for i, x in enumerate(neural_data):
        padded_neural[i, :x.size(0), :] = x
        
    output = {
        'neural': padded_neural,
        'neural_lengths': neural_lengths,
        'text': texts,
        'session_id': session_ids
    }
    
    # 2. Pad phonemes if present
    if 'phonemes' in batch[0]:
        phonemes = [item['phonemes'] for item in batch]
        phoneme_lengths = torch.tensor([item['phoneme_lengths'] for item in batch], dtype=torch.long)
        max_phoneme_len = max(phoneme_lengths).item()
        
        padded_phonemes = torch.zeros(len(batch), max_phoneme_len, dtype=torch.long)
        for i, p in enumerate(phonemes):
            # Ensure p is not longer than its declared length or max_phoneme_len
            curr_len = min(len(p), phoneme_lengths[i].item())
            padded_phonemes[i, :curr_len] = p[:curr_len]
            
        output['phonemes'] = padded_phonemes
        output['phoneme_lengths'] = phoneme_lengths
        
    return output
