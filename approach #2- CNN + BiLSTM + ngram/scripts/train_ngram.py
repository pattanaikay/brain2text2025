import os
import h5py
import pickle
import json
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.n_gram import CharNGramModel

def train_and_save_ngram(n=3):
    # 1. Load the list of HDF5 files
    h5_list_path = r"C:\Projects\Brain2Text2025\brain2text2025\approach #2- CNN + BiLSTM + ngram\src\utils\h5_list_data.json"
    if not os.path.exists(h5_list_path):
        print(f"Error: {h5_list_path} not found.")
        return

    with open(h5_list_path, 'r') as f:
        h5_files = json.load(f)

    all_sentences = []
    print("Extracting sentences from training files...")

    for h5_path in h5_files:
        if 'data_train.hdf5' in h5_path:
            with h5py.File(h5_path, 'r') as f:
                for trial_name in f.keys():
                    if 'transcription' in f[trial_name]:
                        raw_data = f[trial_name]['transcription'][()]
                        
                        # Use logic consistent with BCI_Dataset for extraction
                        if isinstance(raw_data, bytes):
                            sentence = raw_data.decode('utf-8')
                        elif isinstance(raw_data, (np.ndarray, list)):
                            # Convert ASCII indices to characters if it's an array of numbers
                            try:
                                sentence = "".join([chr(int(c)) for c in raw_data])
                            except:
                                sentence = str(raw_data)
                        else:
                            sentence = str(raw_data)
                        
                        all_sentences.append(sentence)
    
    print(f"Total sentences collected: {len(all_sentences)}")
    if not all_sentences:
        print("No sentences found. Check your HDF5 file paths and structure.")
        return

    # 3. Train the model
    print(f"Training {n}-gram model...")
    ngram_model = CharNGramModel(n=n)
    ngram_model.train(all_sentences)

    # 4. Save using Pickle
    save_dir = Path(__file__).parent.parent / 'src' / 'utils'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'ngram_{n}gram.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(ngram_model, f)
    
    print(f"Model saved successfully to {save_path}")

if __name__ == "__main__":
    train_and_save_ngram(n=3)
