import h5py
import numpy as np
import json
import os
import argparse
from tqdm import tqdm

def compute_session_stats(h5_files, output_json='session_stats.json'):
    stats = {}
    for file_path in tqdm(h5_files, desc="Processing Sessions"):
        try:
            with h5py.File(file_path, 'r') as f:
                all_trials = []
                for trial_name in f.keys():
                    trial_group = f[trial_name]
                    
                    neural_key = None
                    for key in ['input_features', 'tx1', 'neural', 'neural_features', 'spikePow']:
                        if key in trial_group:
                            neural_key = key
                            break
                    
                    if neural_key:
                        all_trials.append(trial_group[neural_key][:])
                
                if not all_trials:
                    continue
                
                session_data = np.concatenate(all_trials, axis=0)
                session_id = str(f[list(f.keys())[0]].attrs.get('session', os.path.basename(file_path)))
                
                stats[session_id] = {
                    "mean": session_data.mean(axis=0).tolist(),
                    "std": session_data.std(axis=0).tolist()
                }
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    with open(output_json, 'w') as j:
        json.dump(stats, j)
    print(f"Saved session statistics to {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_list", type=str, required=True, help="Path to JSON file with list of HDF5 paths")
    parser.add_argument("--output", type=str, default="session_stats.json")
    args = parser.parse_args()
    
    with open(args.h5_list, "r") as file:
        h5_list = json.load(file)        
    
    compute_session_stats(h5_list, args.output)
