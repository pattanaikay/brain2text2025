import h5py
import numpy as np
import os
import json

def create_dummy_data():
    os.makedirs('data/test_samples', exist_ok=True)

    for split in ['train', 'val']:
        h5_path = f'data/test_samples/data_{split}.hdf5'
        with h5py.File(h5_path, 'w') as f:
            for i in range(5):
                trial = f.create_group(f'trial_{i}')
                # 512 channels, 100 time bins
                trial.create_dataset('input_features', data=np.random.randn(100, 512).astype(np.float32))
                trial.create_dataset('sentenceText', data=f"This is dummy {split} sentence {i}".encode('utf-8'))
                trial.attrs['session'] = 'session_test'
                # 10 phonemes
                trial.create_dataset('seq_class_ids', data=np.random.randint(1, 40, (10,)).astype(np.int64))
                trial.attrs['seq_len'] = 10
        print(f"Created dummy {split} data at {h5_path}")

    # Create dummy session stats
    stats = {
        'session_test': {
            'mean': np.zeros(512).tolist(),
            'std': np.ones(512).tolist()
        }
    }
    with open('session_stats.json', 'w') as f:
        json.dump(stats, f)
    
    print(f"Created dummy data at {h5_path}")
    print("Created dummy session_stats.json")

if __name__ == "__main__":
    create_dummy_data()
