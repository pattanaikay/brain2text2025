import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import os
from scipy.ndimage import gaussian_filter1d

class Preprocessed_BCI_Dataset(Dataset):
    def __init__(self, h5_paths, trial_list=None, session_stats=None, sigma=1.5,
                 patch_size=4, filter_ctc=False, augment=False,
                 channel_mask_p=0.1, time_cutout_max_len=20, gauss_noise_std=0.1):
        """
        Args:
            h5_paths: Single path (str) or List of paths to HDF5 files.
            trial_list: Optional list of specific trials.
            session_stats: Dictionary containing 'mean' and 'std' for each session.
            sigma: Standard deviation for Gaussian smoothing.
            patch_size: Patch size used for temporal compression.
            filter_ctc: If True, filters out trials where patched_len < phoneme_len.
        """
        if isinstance(h5_paths, str):
            self.h5_paths = [h5_paths]
        else:
            self.h5_paths = h5_paths
            
        self.session_stats = session_stats
        self.sigma = sigma
        self.patch_size = patch_size
        self.augment = augment
        self.channel_mask_p = channel_mask_p
        self.time_cutout_max_len = time_cutout_max_len
        self.gauss_noise_std = gauss_noise_std
        
        # Build index of all trials across all files
        temp_trials = []
        if trial_list:
            if isinstance(trial_list[0], str):
                temp_trials = [(self.h5_paths[0], t) for t in trial_list]
            else:
                temp_trials = trial_list
        else:
            for path in self.h5_paths:
                with h5py.File(path, 'r') as f:
                    for trial_name in f.keys():
                        temp_trials.append((path, trial_name))
        
        # CTC Validation: Filter out trials where patched_len < phoneme_len
        self.all_trials = []
        invalid_count = 0
        for h5_path, trial_name in temp_trials:
            if not filter_ctc:
                self.all_trials.append((h5_path, trial_name))
                continue

            with h5py.File(h5_path, 'r') as f:
                group = f[trial_name]
                
                # Check neural length
                neural_key = None
                for key in ['neural', 'input_features', 'tx1', 'spikePow']:
                    if key in group:
                        neural_key = key
                        break
                if not neural_key: continue
                
                neural_len = group[neural_key].shape[0]
                patched_len = (neural_len + patch_size - 1) // patch_size
                
                # Check phoneme length — use seq_len attr if available (arrays are padded to fixed size)
                phoneme_len = 0
                for key in ['phonemes', 'seq_class_ids', 'phonemeLabels']:
                    if key in group:
                        phoneme_len = len(group[key])
                        break
                # Override with the true (unpadded) sequence length if stored as an attribute
                if 'seq_len' in group.attrs:
                    phoneme_len = int(group.attrs['seq_len'])
                elif 'phoneme_lengths' in group:
                    phoneme_len = int(group['phoneme_lengths'][()])

                if phoneme_len > 0 and patched_len < phoneme_len:
                    invalid_count += 1
                    continue
                
                self.all_trials.append((h5_path, trial_name))
        
        if invalid_count > 0:
            print(f"⚠️ Filtered out {invalid_count} trials where neural patches < phoneme targets (CTC impossibility).")
        
        if len(self.all_trials) == 0:
            raise ValueError(f"FATAL: All trials were filtered out! This means patch_size {patch_size} is too high for your phoneme sequence lengths. Use a smaller patch_size for CTC.")
        
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
                # Brain-to-Text '25 cluster HDF5 stores sentenceText as a numpy integer
                # array (raw UTF-8 bytes in a fixed-size buffer padded with NULL bytes).
                # dtype may be int8 / uint8 / int32 across sessions, so cast to uint8
                # (modular reinterpretation) before converting to a bytes object.
                if isinstance(raw_text, np.ndarray):
                    raw_text = raw_text.astype(np.uint8).tobytes()
                if isinstance(raw_text, bytes):
                    raw_text = raw_text.rstrip(b'\x00')
                    text = raw_text.decode('utf-8', errors='replace').strip()
                else:
                    text = str(raw_text).strip()
            
            # Find phonemes
            phonemes = None
            phoneme_len = 0
            for key in ['phonemes', 'seq_class_ids', 'phonemeLabels']:
                if key in group:
                    phonemes = group[key][:]
                    phoneme_len = len(phonemes)
                    break
            
            if 'seq_len' in group.attrs:
                phoneme_len = int(group.attrs['seq_len'])
            elif 'phoneme_lengths' in group:
                phoneme_len = int(group['phoneme_lengths'][()])

            session_id = group.attrs.get('session', os.path.basename(os.path.dirname(h5_path)))

        if self.session_stats and session_id in self.session_stats:
            mean = self.session_stats[session_id]['mean']
            std = self.session_stats[session_id]['std']
            neural_data = (neural_data - mean) / (std + 1e-8)
        
        if self.sigma > 0:
            neural_data = gaussian_filter1d(neural_data, sigma=self.sigma, axis=0)

        # Training-time augmentation (numpy-side, before tensor conversion)
        if self.augment:
            # Channel mask (SpikeDrop)
            if self.channel_mask_p > 0:
                channel_mask = np.random.rand(neural_data.shape[1]) < self.channel_mask_p
                neural_data[:, channel_mask] = 0.0

            # Time cutout (mask 1-2 random spans)
            if self.time_cutout_max_len > 0:
                n_spans = np.random.randint(1, 3)
                for _ in range(n_spans):
                    span_len = np.random.randint(5, self.time_cutout_max_len + 1)
                    max_start = max(0, neural_data.shape[0] - span_len)
                    if max_start > 0:
                        start = np.random.randint(0, max_start)
                        neural_data[start:start + span_len, :] = 0.0

            # Gaussian noise
            if self.gauss_noise_std > 0:
                neural_data = neural_data + np.random.randn(*neural_data.shape).astype(np.float32) * self.gauss_noise_std

        sample = {
            'neural': torch.tensor(neural_data, dtype=torch.float32),
            'text': text,
            'session_id': str(session_id)
        }
        
        if phonemes is not None:
            phonemes = phonemes + 1
            sample['phonemes'] = torch.tensor(phonemes, dtype=torch.long)
            sample['phoneme_lengths'] = phoneme_len
            
        return sample

def bci_collate_fn(batch):
    neural_data = [item['neural'] for item in batch]
    texts = [item['text'] for item in batch]
    session_ids = [item['session_id'] for item in batch]
    
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
    
    if 'phonemes' in batch[0]:
        phonemes = [item['phonemes'] for item in batch]
        phoneme_lengths = torch.tensor([item['phoneme_lengths'] for item in batch], dtype=torch.long)
        max_phoneme_len = max(phoneme_lengths).item()
        
        padded_phonemes = torch.zeros(len(batch), max_phoneme_len, dtype=torch.long)
        for i, p in enumerate(phonemes):
            curr_len = min(len(p), phoneme_lengths[i].item())
            padded_phonemes[i, :curr_len] = p[:curr_len]
            
        output['phonemes'] = padded_phonemes
        output['phoneme_lengths'] = phoneme_lengths
        
    return output
