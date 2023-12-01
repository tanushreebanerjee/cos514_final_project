import os
import pandas as pd
from collections import namedtuple
from transformers import pipeline
from nltk.metrics import edit_distance

class ASRPipeline:
    def __init__(self, model_name):
        self.asr_model = pipeline(task="automatic-speech-recognition", model=model_name)
        
    def transcribe_audio(self, audio_path):
        return self.asr_model(audio_path)

    @staticmethod
    def calculate_wer(reference, hypothesis):
        # Assuming reference and hypothesis are strings
        return edit_distance(reference.split(), hypothesis.split())

    def calculate_wer_for_corpus(self, reference_corpus, hypothesis_corpus):
        wer_scores = [self.calculate_wer(reference, hypothesis) 
                      for reference, hypothesis in zip(reference_corpus, hypothesis_corpus)]
        avg_wer = sum(wer_scores) / len(wer_scores)
        return avg_wer

# Define a named tuple for ASR results
ASRResult = namedtuple('ASRResult', ['id', 'reference', 'hypothesis'])

def execute_asr_pipeline(asr_model, audio_dir, output_csv):
    asr_pipeline = ASRPipeline(asr_model)
    asr_results = []

    for audio_file in os.listdir(audio_dir):
        audio_path = os.path.join(audio_dir, audio_file)
        transcription = asr_pipeline.transcribe_audio(audio_path)
        reference_text = " ".join(transcription)
        
        # You may have ground truth references in your dataset, or you can use the original transcripts
        # Replace 'original_text' with the column name containing original transcripts in your DataFrame
        original_text = "Replace_this_with_original_text_extraction_from_your_dataset"
        
        asr_results.append(ASRResult(id=audio_file, reference=original_text, hypothesis=reference_text))

    # Save ASR results to CSV
    asr_df = pd.DataFrame(asr_results)
    asr_df.to_csv(output_csv, index=False)

# Example Usage
audio_directory = 'path_to_your_audio_files'
output_csv_path = 'output_path/asr_results.csv'
execute_asr_pipeline(asr_model='whisper-large', audio_dir=audio_directory, output_csv=output_csv_path)
