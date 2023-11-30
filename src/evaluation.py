# Import ROUGE for evaluation
from rouge import Rouge

# Define a function to calculate ROUGE-N scores
def calculate_rouge(reference, hypothesis):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    return scores

# Compare summaries with annotations using ROUGE
for i in range(len(summaries)):
    rouge_scores = calculate_rouge(annotations[i], summaries[i])
    print(f"ROUGE Scores for Example {i}: {rouge_scores}")
