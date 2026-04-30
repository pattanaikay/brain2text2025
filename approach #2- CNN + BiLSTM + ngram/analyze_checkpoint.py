import torch
import os
import sys

def analyze_checkpoint(checkpoint_path):
    print(f"Analyzing checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print("File not found.")
        return

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    if isinstance(checkpoint, dict):
        print(f"Total keys in state_dict: {len(checkpoint.keys())}")
        print("-" * 50)
        print(f"{'Layer Name':<40} | {'Shape':<20} | {'Mean':<10} | {'Std':<10}")
        print("-" * 85)
        
        for key, value in checkpoint.items():
            if isinstance(value, torch.Tensor):
                mean = value.float().mean().item()
                std = value.float().std().item()
                print(f"{key:<40} | {str(list(value.shape)):<20} | {mean:>10.4f} | {std:>10.4f}")
            else:
                print(f"{key:<40} | {type(value)}")
    else:
        print(f"Checkpoint is not a dictionary. Type: {type(checkpoint)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        # Default to one of the checkpoints if none provided
        path = r"C:\Projects\Brain2Text2025\brain2text2025\approach #2- CNN + BiLSTM + ngram\scripts\models\best_model_wer.pth"
    
    analyze_checkpoint(path)
