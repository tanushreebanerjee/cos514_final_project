from rouge import Rouge

class Evaluation:
    def __init__(self):
        self.rouge = Rouge()

    def calculate_rouge(self, reference, hypothesis):
        # Calculate ROUGE-N scores
        scores = self.rouge.get_scores(hypothesis, reference)
        return scores

# Example usage:
evaluation = Evaluation()

# Example reference and hypothesis
reference_text = "This is the reference summary."
hypothesis_text = "This is the generated summary."

# Calculate ROUGE scores
rouge_scores = evaluation.calculate_rouge(reference_text, hypothesis_text)

# Print the result
print("ROUGE Scores:")
print(rouge_scores)
