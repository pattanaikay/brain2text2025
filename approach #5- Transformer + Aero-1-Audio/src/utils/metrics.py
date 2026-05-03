import jiwer

def calculate_wer(predictions, targets):
    """Calculate Word Error Rate using jiwer."""
    return jiwer.wer(list(targets), list(predictions))

def calculate_cer(predictions, targets):
    """Calculate Character Error Rate using jiwer."""
    return jiwer.cer(list(targets), list(predictions))

def calculate_per(predictions, targets):
    """Calculate Phoneme Error Rate using jiwer (same as WER if phonemes are space-separated)."""
    return jiwer.wer(list(targets), list(predictions))
