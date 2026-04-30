import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class TextTokenizer:
    def __init__(self):
        self.chars = " abcdefghijklmnopqrstuvwxyz'" 
        self.char_to_int = {char: i + 1 for i, char in enumerate(self.chars)}
        self.int_to_char = {i + 1: char for i, char in enumerate(self.chars)}
        self.char_to_int['[blank]'] = 0

    def encode(self, text):
        return [self.char_to_int.get(c, 0) for c in text.lower()]
    
    def decode(self, tokens):
        return "".join([self.int_to_char.get(t, '') for t in tokens])

class Preprocessed_BCI_Dataset(Dataset):
    def __init__(self, h5_path, trial_list, tokenizer=None):
        self.h5_path = h5_path
        self.trial_list = trial_list
        self.tokenizer = tokenizer or TextTokenizer()
        
    def __len__(self):
        return len(self.trial_list)
        
    def __getitem__(self, idx):
        trial_name = self.trial_list[idx]
        with h5py.File(self.h5_path, 'r') as f:
            neural_data = f[trial_name]['neural'][:]
            text = f[trial_name]['text'][()].decode('utf-8')
            
        return {
            'neural': torch.tensor(neural_data, dtype=torch.float32),
            'text': text
        }

def bci_collate_fn(batch):
    neural_data = torch.stack([item['neural'] for item in batch])
    texts = [item['text'] for item in batch]
    return {'neural': neural_data, 'text': texts}
