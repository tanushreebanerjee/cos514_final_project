# Main Script (main.py):
import os
from src.data_preparation import DataPreparation
from src.error_injection import ErrorInjection
from src.summarization import Summarization
from src.evaluation import Evaluation

# Main execution flow
corpus_path = os.path.join(os.getcwd(), 'data/raw/AMI_corpus')
data_prep = DataPreparation(corpus_path)
transcripts_df = data_prep.load_transcripts()
baseline_transcripts = data_prep.create_baseline_transcripts(transcripts_df)

error_types = ['speaker_identification', 'speech_recognition']
error_injector = ErrorInjection(error_types)
degree = 0.1 # 10% of the words in the transcript will be erroneous
transcripts_with_errors = [error_injector.inject_errors(t, degree) for t in baseline_transcripts]

llm_model_name = "Xwin-LM-7B-v0.1" 
summarizer = Summarization(llm_model_name)
summaries = [summarizer.summarize(t) for t in transcripts_with_errors]

# Assuming 'annotations' is a column in your DataFrame
annotations = transcripts_df['annotations']

evaluation = Evaluation()
for i in range(len(transcripts_with_errors)):
    rouge_scores = evaluation.calculate_rouge(annotations[i], summaries[i])
    print(f"ROUGE Scores for Example {i}: {rouge_scores}")
