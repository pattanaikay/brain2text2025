import torch
import h5py
import os
import sys

def check_training_health():
    train_h5_path = '/home/data/data/hdf5_data_final/t15.2023.08.13/data_train.hdf5'
    ckpt_path = '/home/outputs/ctc/checkpoint_ctc_latest.pth'
    patch_size = 5

    print("\n" + "="*50)
    print("🧠 BCI CTC DIAGNOSTIC REPORT")
    print("="*50)

    # 1. Check Data Lengths
    if not os.path.exists(train_h5_path):
        print(f"❌ Error: Cannot find training data at {train_h5_path}")
    else:
        print("\n--- 1. DATA LENGTH CHECK ---")
        try:
            with h5py.File(train_h5_path, 'r') as f:
                trials = list(f.keys())
                
                impossible_trials = 0
                total_trials = min(100, len(trials)) # Check first 100
                
                print(f"Checking first {total_trials} trials...")
                
                for trial in trials[:total_trials]:
                    # Find neural length
                    neural_key = next((k for k in ['input_features', 'tx1', 'neural', 'spikePow'] if k in f[trial]), None)
                    if not neural_key: continue
                    neural_len = f[trial][neural_key].shape[0]
                    
                    # Find phoneme length
                    phoneme_len = 0
                    if 'seq_len' in f[trial].attrs:
                        phoneme_len = int(f[trial].attrs['seq_len'])
                    elif 'seq_class_ids' in f[trial]:
                        phoneme_len = len(f[trial]['seq_class_ids'])
                    
                    patched_len = (neural_len + patch_size - 1) // patch_size
                    
                    if patched_len < phoneme_len:
                        impossible_trials += 1
                        print(f"⚠️ {trial}: Neural Patches ({patched_len}) < Phonemes ({phoneme_len})")
                
                if impossible_trials > 0:
                    print(f"❌ FAILED: Found {impossible_trials}/{total_trials} trials where Patches < Phonemes.")
                    print("    Cause: CTC loss requires input length >= target length.")
                    print("    Fix: We must decrease patch_size or pad the neural sequence.")
                else:
                    print("✅ PASSED: All checked trials have valid sequence lengths.")
        except Exception as e:
            print(f"Error reading HDF5: {e}")

    # 2. Check for NaNs
    print("\n--- 2. WEIGHT EXPLOSION CHECK ---")
    if not os.path.exists(ckpt_path):
        print(f"❌ Error: Cannot find checkpoint at {ckpt_path}")
    else:
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            weights = ckpt.get('model_state_dict', {})
            
            has_nan = False
            for name, param in weights.items():
                if torch.isnan(param).any():
                    has_nan = True
                    print(f"⚠️ NaN found in layer: {name}")
                    break
            
            if has_nan:
                print("❌ FAILED: Model weights contain NaNs (Gradient Explosion).")
                print("    Cause: Learning rate too high, or unhandled 0/inf values in loss.")
                print("    Fix: Lower learning rate or add gradient clipping/NaN checks.")
            else:
                print("✅ PASSED: No NaNs found in model weights.")
        except Exception as e:
            print(f"Error reading checkpoint: {e}")

    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    check_training_health()