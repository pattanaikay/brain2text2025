import torch

def greedy_decoder(logits, tokenizer):
    """
    logits: (Time, Classes) - output of model for ONE trial
    """
    # 1. Get the most likely character index at each time step
    indices = torch.argmax(logits, dim=-1).tolist()
    
    # 2. Collapse repeated characters and remove blanks (0)
    decoded = []
    prev_idx = -1
    for idx in indices:
        if idx != prev_idx: # Remove adjacent duplicates
            if idx != 0:    # Remove CTC blanks
                decoded.append(idx)
        prev_idx = idx
        
    return tokenizer.decode(decoded)