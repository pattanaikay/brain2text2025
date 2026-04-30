import jiwer

def calculate_wer(predictions, targets):
    """Calculate Word Error Rate using jiwer."""
    return jiwer.wer(list(targets), list(predictions))

def calculate_cer(predictions, targets):
    """Calculate Character Error Rate using jiwer."""
    return jiwer.cer(list(targets), list(predictions))
