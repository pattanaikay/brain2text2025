import editdistance
import jiwer

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

def calculate_wer(predictions, targets):
    """
    Kaggle-Style WER using jiwer with standard normalization.
    Ensures lowercasing and punctuation stripping as per the jiwer standard.
    """
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ])
    
    if not predictions or not targets:
        return 1.0
        
    return jiwer.wer(
        targets, 
        predictions, 
        reference_transform=transformation, 
        hypothesis_transform=transformation
    )
