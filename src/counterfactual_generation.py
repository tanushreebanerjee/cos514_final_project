import pandas as pd
import numpy as np
import os
from pyannote.core import Segment, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate

class CounterfactualGenerator:
    def __init__(self, corpus_df):
        self.corpus_df = corpus_df.copy()

    def change_speakers_randomly(self):
        # Naive implementation: shuffle speaker IDs randomly
        self.corpus_df['speaker'] = np.random.permutation(self.corpus_df['speaker'])
    
    def change_sentences(self):
        # Naive implementation: shuffle sentences randomly
        self.corpus_df['text'] = np.random.permutation(self.corpus_df['text'])

    def save_counterfactual_to_csv(self, output_csv):
        self.corpus_df.to_csv(output_csv, index=False)

def load_corpus_csv(csv_file):
    return pd.read_csv(csv_file)

def compute_der(reference, hypothesis):
    # Compute Diarization Error Rate (DER)
    metric = DiarizationErrorRate()
    der = metric(reference, hypothesis)
    return der

# Example usage
# Load your processed corpus CSV file
corpus_name = 'ami-corpus'
processed_corpus_path = os.path.join(os.getcwd(), 'data', 'processed', f'{corpus_name}.csv')
processed_corpus_df = load_corpus_csv(processed_corpus_path)

# Create CounterfactualGenerator
counterfactual_generator = CounterfactualGenerator(processed_corpus_df)

# Original transcripts
reference_annotation = Annotation()
for index, row in processed_corpus_df.iterrows():
    reference_annotation[Segment(row['starttime'], row['endtime'])] = row['speaker']

# Generate counterfactuals (change speakers randomly)
print('Generating counterfactuals (change speakers randomly)...')
counterfactual_generator.change_speakers_randomly()

# Counterfactual transcripts
hypothesis_annotation = Annotation()
for index, row in counterfactual_generator.corpus_df.iterrows():
    hypothesis_annotation[Segment(row['starttime'], row['endtime'])] = row['speaker']

# Compute DER
der = compute_der(reference_annotation, hypothesis_annotation)
print(f'Diarization Error Rate (DER): {der}')

# Save counterfactual transcripts to CSV
counterfactual_path = os.path.join(os.getcwd(), 'data', 'counterfactual', f'{corpus_name}-counterfactual-changeSpeakers-der-{der:.2f}.csv')
os.makedirs(os.path.dirname(counterfactual_path), exist_ok=True)
counterfactual_generator.save_counterfactual_to_csv(counterfactual_path)

# Generate counterfactuals (change sentences randomly)
print('Generating counterfactuals (change sentences randomly)...')
counterfactual_generator.change_sentences()

# Counterfactual transcripts
hypothesis_annotation = Annotation()
for index, row in counterfactual_generator.corpus_df.iterrows():
    hypothesis_annotation[Segment(row['start_time'], row['end_time'])] = row['speaker']

# Compute DER
der = compute_der(reference_annotation, hypothesis_annotation)
print(f'Diarization Error Rate (DER): {der}')

# Save counterfactual transcripts to CSV
counterfactual_path = os.path.join(os.getcwd(), 'data', 'counterfactual', f'{corpus_name}-counterfactual-changeSentences-der-{der:.2f}.csv')