import editdistance

def calculate_cer(predictions, targets):
    """
    predictions: List of strings
    targets: List of strings
    """
    total_dist = 0
    total_chars = 0
    for pred, target in zip(predictions, targets):
        total_dist += editdistance.eval(pred, target)
        total_chars += len(target)
    return total_dist / total_chars if total_chars > 0 else 0