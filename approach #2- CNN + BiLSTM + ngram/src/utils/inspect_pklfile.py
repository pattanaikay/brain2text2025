import sys
from pathlib import Path
import pickle

# Add parent directories to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
file_path = r"C:\Projects\Brain2Text2025\brain2text2025\approach #2- CNN + BiLSTM + ngram\src\utils\ngram_3gram.pkl"

# 1. Load the model
with open(file_path, 'rb') as f:
    ngram = pickle.load(f)

# 2. Test Probabilities
# In English, "th" is very common, "tx" is rare.
prob_th = ngram.get_char_log_prob("th", "e")
prob_tx = ngram.get_char_log_prob("t", "x")

print(f"Log-Prob of 'h' after 't': {prob_th:.4f}")
print(f"Log-Prob of 'x' after 't': {prob_tx:.4f}")

if prob_th > prob_tx:
    print("✅ Success: The model correctly favors common English patterns.")
else:
    print("❌ Warning: The model isn't distinguishing common patterns. Check your training data.")