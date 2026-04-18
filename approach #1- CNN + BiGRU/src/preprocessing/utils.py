import h5py
import numpy as np
import json

def compute_session_stats(h5_files):
    stats = {}
    for file_path in h5_files:
        with h5py.File(file_path, 'r') as f:
            all_trials = []
            for trial_name in f.keys():
                # We only need the neural data (tx1) for stats
                all_trials.append(f[trial_name]['tx1'][:])
            
            # Combine all trials in this session to get 'Global' stats
            session_data = np.concatenate(all_trials, axis=0)
            
            stats[file_path] = {
                "mean": session_data.mean(axis=0).tolist(),
                "std": session_data.std(axis=0).tolist()
            }
            print(f"Stats computed for {file_path}")
            
    # Save to JSON so you don't have to re-run this
    with open('session_stats.json', 'w') as j:
        json.dump(stats, j)

# Usage:
# h5_list = ['data/raw/train_session_1.hdf5', 'data/raw/train_session_2.hdf5']
# compute_session_stats(h5_list)

# smoothing

"""
Neural firing is stochastic and jittery. Even if a participant is thinking "A," the electrodes might show erratic spikes. 
A Gaussian filter acts as a low-pass filter, "smearing" the activity across time to help the CNN identify the underlying motor intent rather than the noise.

"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

def apply_smoothing(neural_data, sigma=1.5):
    """
    neural_data: (Time, 512) array
    sigma: Standard deviation for Gaussian kernel. 
           At 20ms bins, sigma=1.5 covers approx 30-40ms of context.
    """
    # We smooth along the time axis (axis 0)
    # This turns discrete 'spike counts' into continuous 'firing rates'
    smoothed_data = gaussian_filter1d(neural_data, sigma=sigma, axis=0)
    return smoothed_data

# normalization

"""
This is the most important step for the T15 dataset. Because neural signals are non-stationary, the "baseline" of channel #42 today might be significantly higher or lower 
than it was yesterday due to electrode impedance or minor physical shifts in the array.
We use the standard Z-score formula:$$Z = \frac{x - \mu}{\sigma}$$

Why it's needed: 1.  Non-stationarity: If you train on Session A and test on Session B without normalization, the model might see a change in baseline voltage and interpret it as a different phoneme.
2.  Gradient Stability: Neural Networks converge much faster when inputs are centered around zero with unit variance.
3.  Dead Channels: If a channel is dead (constant 0), its $\sigma$ is 0. Standardizing ensures these don't blow up your gradients.

"""

def zscore_session(neural_data):
    """
    neural_data: (Time, 512) array
    Note: Always calculate mean/std per SESSION, not per trial.
    """
    # Calculate across the time axis for each channel
    mean = np.mean(neural_data, axis=0)
    std = np.std(neural_data, axis=0)
    
    # Add epsilon to avoid division by zero for dead channels
    std[std == 0] = 1e-8
    
    normalized_data = (neural_data - mean) / std
    return normalized_data