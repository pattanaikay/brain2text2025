import torch
import time
import pickle
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing.dataloader import TextTokenizer
from src.utils.decoders import beam_search_decoder
from src.utils.n_gram import CharNGramModel

def diagnose():
    tokenizer = TextTokenizer()
    # Mock logits: (Time=500, Classes=len(tokenizer.char_to_int))
    num_classes = len(tokenizer.char_to_int)
    time_steps = 500
    logits = torch.randn(time_steps, num_classes).log_softmax(dim=-1)
    
    # Load real ngram model
    ngram_path = r"src\utils\ngram_3gram.pkl"
    if os.path.exists(ngram_path):
        with open(ngram_path, 'rb') as f:
            ngram_model = pickle.load(f)
        print(f"Loaded real N-gram model from {ngram_path}")
    else:
        print("Real N-gram model not found.")
        return
    
    print(f"Starting optimized beam search with T={time_steps}, Beam Width=10...")
    start_time = time.time()
    result = beam_search_decoder(logits, tokenizer, ngram_model, beam_width=10, alpha=0.5)
    end_time = time.time()
    
    print(f"Result length: {len(result)}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    diagnose()
