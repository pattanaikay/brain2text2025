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
    parser.add_argument("--h5_list", type=str, help="Path to JSON file with list of HDF5 paths")
    parser.add_argument("--data_dir", type=str, help="Directory containing HDF5 files (searched recursively)")
    parser.add_argument("--output", type=str, default="session_stats.json")
    args = parser.parse_args()
    
    if args.h5_list:
        with open(args.h5_list, "r") as file:
            h5_list = json.load(file)
    elif args.data_dir:
        import glob
        h5_list = glob.glob(os.path.join(args.data_dir, "**/*.hdf5"), recursive=True)
        # Filter for training files to avoid using test/val for stats
        h5_list = [f for f in h5_list if "data_train.hdf5" in f]
        print(f"Found {len(h5_list)} training files in {args.data_dir}")
    else:
        print("Error: Must provide either --h5_list or --data_dir")
        exit(1)
    
    compute_session_stats(h5_list, args.output)
