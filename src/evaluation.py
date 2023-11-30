from rouge import Rouge

class Evaluation:
    def __init__(self):
        self.rouge = Rouge()

    def calculate_rouge(self, reference, hypothesis):
        # Calculate ROUGE-N scores
        scores = self.rouge.get_scores(hypothesis, reference)
        return scores
