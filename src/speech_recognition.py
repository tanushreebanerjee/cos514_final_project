import os
import pandas as pd
from collections import namedtuple
from nltk.metrics import edit_distance
from transformers import pipeline
from rouge import Rouge

# ASR Pipeline Class
class ASRPipeline:
    def __init__(self, model_name):
        self.asr_model = pipeline(task="automatic-speech-recognition", model=model_name)
        
    def transcribe_audio(self, audio_path):
        return self.asr_model(audio_path)

    @staticmethod
    def calculate_wer(reference, hypothesis):
        return edit_distance(reference.split(), hypothesis.split())

    def calculate_wer_for_corpus(self, reference_corpus, hypothesis_corpus):
        wer_scores = [self.calculate_wer(reference, hypothesis) 
                      for reference, hypothesis in zip(reference_corpus, hypothesis_corpus)]
        avg_wer = sum(wer_scores) / len(wer_scores)
        return avg_wer

# Define a named tuple for ASR results
ASRResult = namedtuple('ASRResult', ['id', 'reference', 'hypothesis'])

# Load Corpus CSV function
def load_corpus_csv(csv_file):
    return pd.read_csv(csv_file)

# Save Evaluation Results to CSV function
def save_evaluation_results_to_csv(results_df, output_file='results/evaluation_results.csv'):
    results_df.to_csv(output_file, index=False)
    print(f'Evaluation Results saved to {output_file}')

# Main Function
def main():
    # Step 1: Speech Recognition
    asr_model = 'whisper-large'
    audio_directory = 'path_to_your_audio_files'
    asr_output_csv = 'results/asr_results.csv'
    
    # ASR Pipeline
    asr_pipeline = ASRPipeline(asr_model)
    asr_results = []

    for audio_file in os.listdir(audio_directory):
        audio_path = os.path.join(audio_directory, audio_file)
        transcription = asr_pipeline.transcribe_audio(audio_path)
        
        # Replace 'original_text' with the column name containing original transcripts in your DataFrame
        original_text = "Replace_this_with_original_text_extraction_from_your_dataset"
        
        asr_results.append(ASRResult(id=audio_file, reference=original_text, hypothesis=" ".join(transcription)))

    # Save ASR results to CSV
    asr_df = pd.DataFrame(asr_results)
    asr_df.to_csv(asr_output_csv, index=False)
    print(f'ASR Results saved to {asr_output_csv}')

    # Load your processed corpus CSV file
    corpus_name = 'ami-corpus'
    processed_corpus_path = os.path.join(os.getcwd(), 'data', 'processed', f'{corpus_name}.csv')
    processed_corpus_df = load_corpus_csv(processed_corpus_path)

    # Load ASR results CSV
    asr_results_df = pd.read_csv(asr_output_csv)

    # Assuming you have 'id' column in your processed_corpus_df and asr_results_df
    merged_df = pd.merge(processed_corpus_df, asr_results_df, on='id')

    # Step 2: Summarization and Evaluation
    # (Add your summarization and evaluation logic here)

    # Save Evaluation Results to CSV
    evaluation_results_df = pd.DataFrame(columns=['Model', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'WER'])
    save_evaluation_results_to_csv(evaluation_results_df)

if __name__ == "__main__":
    main()
