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

def beam_search_decoder(logits, tokenizer, ngram_model, beam_width=10, alpha=0.5):
    """
    logits: (Time, Classes) from model.log_softmax()
    alpha: Weight for the N-gram model (tune this!)
    """
    # Each beam: (string, last_char, total_log_prob)
    beams = [("", "", 0.0)]
    
    for t in range(logits.size(0)):
        new_beams = []
        for seq, last_char, score in beams:
            for char_idx in range(logits.size(1)):
                char = tokenizer.int_to_char[char_idx]
                
                # 1. Acoustic Score from neural model
                acoustic_score = logits[t, char_idx].item()
                
                # 2. LM Score (only if char isn't a CTC blank or repeat)
                lm_score = 0
                if char != "" and char != last_char:
                    lm_score = ngram_model.get_char_log_prob(seq, char)
                
                # Combined Score: log(P_acoustic) + alpha * log(P_lm)
                new_score = score + acoustic_score + (alpha * lm_score)
                new_beams.append((seq + char, char, new_score))
        
        # Keep only the top K beams to save memory
        beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_width]

    # Return the string with the highest combined score
    return beams[0][0]