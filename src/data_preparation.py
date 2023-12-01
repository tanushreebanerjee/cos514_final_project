import pandas as pd
from datasets import load_dataset

class DataPreparation:
    def __init__(self, corpus_name):
        self.corpus_name = corpus_name

    def load_transcripts(self):
        # Load transcripts from the chosen corpus
        if self.corpus_name == 'AMI':
            dataset = load_dataset('ami', 'meeting_transcription')
            transcripts_df = pd.DataFrame({'transcript': dataset['train']['text']})
            transcripts_df['reference_summary'] = dataset['train']['summary']
        elif self.corpus_name == 'ISCI':
            # Add logic to load ISCI corpus if needed
            pass
        else:
            raise ValueError(f"Unsupported corpus: {self.corpus_name}")

        return transcripts_df

    def create_baseline_transcripts(self, transcripts_df, sample_size=None):
        # Logic to create baseline transcripts
        # For example, select a random sample if sample_size is provided
        if sample_size is not None:
            transcripts_df = transcripts_df.sample(sample_size, random_state=42)

        # Additional preprocessing steps if needed

        return transcripts_df

# Example usage:
corpus_name = 'AMI'  # Change to 'ISCI' if needed
data_prep = DataPreparation(corpus_name)
transcripts_df = data_prep.load_transcripts()
baseline_transcripts_df = data_prep.create_baseline_transcripts(transcripts_df, sample_size=100)
