import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset (Update with your local path)
file_path = r"C:\Projects\Brain2Text2025\t15_copyTask_neuralData\hdf5_data_final\t15.2023.10.08\data_train.hdf5"

with h5py.File(file_path, 'r') as f:
    # 2. Extract keys and pick a trial
    # The files are usually structured as 'block_X_trial_Y'
    trial_names = list(f.keys())
    sample_trial = trial_names[10]  # Pick trial 10 for a good sample
    
    data = f[sample_trial]
    
    # Debug: Print available keys in this trial
    print(f"Trial: {sample_trial}")
    print(f"Available keys in trial: {list(data.keys())}")
    
    # Features in this dataset:
    # input_features: Neural data - shape (Time, Channels)
    # seq_class_ids: Sequence classification IDs
    # transcription: The text transcription
    
    # Dynamically find the neural data key
    neural_key = None
    for key in ['input_features', 'tx1', 'spikePow', 'neural_data']:
        if key in data:
            neural_key = key
            break
    
    if neural_key is None:
        print(f"ERROR: Could not find neural data key. Available keys: {list(data.keys())}")
        raise KeyError(f"Neural data key not found in trial {sample_trial}")
    
    neural_data = data[neural_key][:]
    
    # Get transcription (handles both string and bytes formats)
    # Updated transcription retrieval logic
    if 'sentenceText' in data:
        # Most HDF5 strings are stored as bytes and need decoding
        raw_text = data['sentenceText'][()]
        if isinstance(raw_text, bytes):
            transcript = raw_text.decode('utf-8').strip()
        else:
            transcript = str(raw_text).strip()
    elif 'transcription' in data:
        # If only indices are available, you'd need a mapping (charMap) to decode them
        transcript = f"Indices: {data['transcription'][()][:10]}..." 
    else:
        transcript = "No Transcription Found"

# 3. Visualization logic
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})

# Heatmap: Shows activity across all 256 channels over time
sns.heatmap(neural_data.T, ax=ax1, cmap='magma', cbar_kws={'label': 'Firing Rate'})
ax1.set_title(f"Neural Activity Heatmap: {transcript}", fontsize=14)
ax1.set_ylabel("Electrode Channels (0-255)")
ax1.set_xlabel("Time Bins")

# Aggregate Activity: Sum of all channels to see 'Speech Bursts'
agg_activity = np.sum(neural_data, axis=1)
ax2.plot(agg_activity, color='cyan', linewidth=1.5)
ax2.fill_between(range(len(agg_activity)), agg_activity, color='cyan', alpha=0.2)
ax2.set_title("Aggregate Spiking Activity (Total Energy)", fontsize=12)
ax2.set_ylabel("Summed Spikes")
ax2.set_xlabel("Time Bins")

plt.tight_layout()
plt.show()