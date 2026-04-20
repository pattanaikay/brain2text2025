import torch
import collections
import math

def logaddexp(a, b):
    """
    Log-sum-exp for two numbers.
    """
    if a == -float('inf'): return b
    if b == -float('inf'): return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    else:
        return b + math.log1p(math.exp(a - b))

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

def beam_search_decoder(logits, tokenizer, ngram_model, beam_width=10, alpha=0.5):
    """
    A proper CTC beam search decoder with N-gram LM integration.
    logits: (Time, Classes) from model.log_softmax()
    """
    # Initial state: empty prefix has prob 1 (log 0) ending in blank
    # Each entry in beams: (prefix, (log_p_blank, log_p_non_blank))
    beams = [("", (0.0, -float('inf')))]
    
    # Prune logits to only top K to speed up significantly
    top_k = min(10, logits.size(1))
    top_probs, top_indices = torch.topk(logits, top_k, dim=-1)
    
    for t in range(logits.size(0)):
        new_beams = collections.defaultdict(lambda: [-float('inf'), -float('inf')])
        
        # Get top characters for this time step
        current_top_probs = top_probs[t]
        current_top_indices = top_indices[t]
        
        for i in range(top_k):
            p = current_top_probs[i].item()
            char_idx = current_top_indices[i].item()
            char = tokenizer.int_to_char[char_idx]
            
            for prefix, (p_b, p_nb) in beams:
                if char_idx == 0: # Blank
                    # If we emit a blank, prefix doesn't change
                    # Probability of ending in blank comes from previous blank or non-blank
                    new_beams[prefix][0] = logaddexp(new_beams[prefix][0], p_b + p)
                    new_beams[prefix][0] = logaddexp(new_beams[prefix][0], p_nb + p)
                else:
                    last_char = prefix[-1] if prefix else None
                    
                    if char == last_char:
                        # Case 1: same char, separated by blank -> new char in sequence
                        # New state is non-blank
                        new_beams[prefix + char][1] = logaddexp(new_beams[prefix + char][1], p_b + p)
                        
                        # Case 2: same char, NOT separated by blank -> collapses into same char
                        # Current prefix, new state is non-blank
                        new_beams[prefix][1] = logaddexp(new_beams[prefix][1], p_nb + p)
                    else:
                        # Different char -> always extends sequence
                        new_prefix = prefix + char
                        
                        # Apply LM score only when extending the prefix with a NEW character
                        lm_score = 0
                        if alpha > 0:
                            lm_score = ngram_model.get_char_log_prob(prefix, char)
                        
                        # Probability of extending comes from both previous blank and non-blank states
                        combined_p = logaddexp(p_b, p_nb)
                        new_beams[new_prefix][1] = logaddexp(new_beams[new_prefix][1], combined_p + p + alpha * lm_score)

        # Sort and prune to beam_width
        # Total probability is logaddexp of blank and non-blank states
        beams = sorted(new_beams.items(), key=lambda x: logaddexp(x[1][0], x[1][1]), reverse=True)[:beam_width]

    # Return the string with the highest combined score
    return beams[0][0]
