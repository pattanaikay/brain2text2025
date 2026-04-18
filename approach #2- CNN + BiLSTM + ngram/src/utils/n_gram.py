
import collections
import numpy as np

class CharNGramModel:
    def __init__(self, n=3):
        self.n = n
        self.counts = collections.defaultdict(collections.Counter)

    def train(self, sentences):
        for sentence in sentences:
            # Add padding to handle sentence starts
            padded = (self.n - 1) * "~" + sentence.lower()
            for i in range(len(padded) - self.n + 1):
                context = padded[i : i + self.n - 1]
                target = padded[i + self.n - 1]
                self.counts[context][target] += 1

    def get_char_log_prob(self, context, char):
        # Retrieve count of char given context, use smoothing to avoid 0
        context = context[-(self.n-1):].ljust(self.n-1, "~")
        context_counts = self.counts[context]
        total = sum(context_counts.values())
        
        # Simple Laplace smoothing
        prob = (context_counts[char] + 0.1) / (total + 0.1 * 28)
        return np.log(prob)