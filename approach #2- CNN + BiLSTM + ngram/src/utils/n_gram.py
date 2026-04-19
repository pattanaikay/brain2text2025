import collections
import numpy as np

class CharNGramModel:
    def __init__(self, n=3):
        self.n = n
        # counts maps context string -> Counter of next characters
        self.counts = collections.defaultdict(collections.Counter)
        self.vocabulary = set()

    def train(self, sentences):
        """
        Trains the n-gram model on a list of sentences.
        Each sentence is stripped and lowercased.
        Padding is added to the beginning of each sentence to handle contexts at the start.
        """
        for sentence in sentences:
            # Case & Space Handling: Ensure inputs are stripped and lowercased
            sentence = sentence.strip().lower()
            if not sentence:
                continue
                
            # Padding Logic: Use a consistent padding character (~) for the start of sentences
            # This ensures that short contexts at the beginning of a sentence are correctly captured.
            padding = (self.n - 1) * "~"
            padded = padding + sentence
            
            # Sliding Window: Capture every n-length sequence
            # Example for n=3: '~~abc' -> ('~~', 'a'), ('~a', 'b'), ('ab', 'c')
            for i in range(len(padded) - self.n + 1):
                context = padded[i : i + self.n - 1]
                target = padded[i + self.n - 1]
                self.counts[context][target] += 1
                self.vocabulary.add(target)
        
        # Ensure the padding character is in the vocabulary for smoothing calculations
        if self.n > 1:
            self.vocabulary.add("~")

    def get_char_log_prob(self, context, char):
        """
        Returns the log-probability of a character given a context.
        The context is standardized to n-1 characters and padded if necessary.
        """
        # Case & Space Handling: Ensure inputs are lowercased
        context = context.lower()
        char = char.lower()
        
        # Standardize Context: Ensure retrieval consistently uses n-1 characters
        # If context is longer than n-1, take the last n-1 characters.
        # If shorter, pad with ~ on the left (matching the training prefix padding).
        if self.n > 1:
            # Handle case where context is shorter than n-1 by rjust with ~
            # and slice to get exactly n-1 characters if it's longer.
            context = context[-(self.n - 1):].rjust(self.n - 1, "~")
        else:
            context = ""
        
        context_counts = self.counts[context]
        total = sum(context_counts.values())

        # Simple Laplace smoothing
        # Use vocabulary size for smoothing; default to 28 if empty
        vocab_size = len(self.vocabulary) if self.vocabulary else 28
        
        # Laplace smoothing with 0.1 epsilon
        prob = (context_counts[char] + 0.1) / (total + 0.1 * vocab_size)
        return np.log(prob)
