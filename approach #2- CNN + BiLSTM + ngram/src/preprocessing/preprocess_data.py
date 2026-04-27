import h5py
import numpy as np
import json
import os
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import sys

# Add parent directory to path so we can import src
base_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(base_path))

from src.preprocessing.dataloader import TextTokenizer

def preprocess_to_h5(h5_list_path, stats_path, output_h5_path, sigma=1.5):
    """
    Reads raw H5 files, applies normalization and smoothing, and saves to a single H5.
    """
    tokenizer = TextTokenizer()
    
    with open(h5_list_path, 'r') as f:
        h5_files = json.load(f)
        
    with open(stats_path, 'r') as f:
        raw_stats = json.load(f)
    
    # Pre-process stats into numpy arrays
    stats = {}
    for fp, stat in raw_stats.items():
        mean = np.array(stat['mean'], dtype=np.float32)
        std = np.array(stat['std'], dtype=np.float32)
        std[std == 0] = 1e-8
        stats[fp] = (mean, std)

    print(f"Creating pre-processed H5 at: {output_h5_path}")
    
    with h5py.File(output_h5_path, 'w') as out_f:
        for file_path in h5_files:
            file_name = os.path.basename(file_path)
            print(f"Processing {file_name}...")
            
            mean, std = stats.get(file_path, (None, None))
            if mean is None:
                print(f"  Warning: No stats found for {file_path}. Skipping.")
                continue
                
            with h5py.File(file_path, 'r') as in_f:
                for trial_name in tqdm(in_f.keys()):
                    trial_group = in_f[trial_name]
                    
                    # 1. Load Neural Data
                    if 'input_features' in trial_group:
                        data = trial_group['input_features'][:]
                    elif 'tx1' in trial_group:
                        data = trial_group['tx1'][:]
                    elif 'neural_features' in trial_group:
                        data = trial_group['neural_features'][:]
                    else:
                        continue
                        
                    # 2. Normalize
                    normalized = (data.astype(np.float32) - mean) / std
                    
                    # 3. Smooth
                    smoothed = gaussian_filter1d(normalized, sigma=sigma, axis=0)
                    
                    # 4. Transcription
                    transcription = ""
                    try:
                        if 'transcription' in trial_group:
                            t_data = trial_group['transcription'][:]
                            if isinstance(t_data, (bytes, np.bytes_)):
                                transcription = t_data.decode('utf-8', errors='ignore')
                            elif isinstance(t_data, np.ndarray):
                                if t_data.dtype.kind in ['S', 'U']:
                                    transcription = "".join([s.decode('utf-8', errors='ignore') if isinstance(s, bytes) else s for s in t_data])
                                else:
                                    transcription = "".join([chr(int(c)) for c in t_data if int(c) != 0])
                        # Remove NULL characters that may have been embedded
                        transcription = transcription.replace('\x00', '')
                    except:
                        pass
                    
                    # 5. Save to new H5
                    # Create unique group names using session_file__trialname format
                    session_id = os.path.basename(os.path.dirname(file_path))
                    group_name = f"{session_id}_{os.path.basename(file_path)}__{trial_name}"
                    
                    g = out_f.create_group(group_name)
                    g.create_dataset('neural', data=smoothed, compression="gzip")
                    # Ensure transcription is clean and encoded as UTF-8
                    transcription_bytes = transcription.encode('utf-8', errors='ignore')
                    g.create_dataset('transcription', data=transcription_bytes)
                    g.attrs['original_file'] = file_path

if __name__ == "__main__":
    h5_list = base_path / 'src' / 'utils' / 'h5_list_data.json'
    stats = base_path / 'src' / 'preprocessing' / 'session_stats.json'
    output = base_path / 'data' / 'preprocessed_data.h5'
    
    os.makedirs(output.parent, exist_ok=True)
    preprocess_to_h5(str(h5_list), str(stats), str(output))
