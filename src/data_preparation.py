import pandas as pd
from datasets import load_dataset
import os

class DataPreparation:
    def __init__(self, corpus_name, cache_dir=None):
        self.corpus_name = corpus_name
        self.cache_dir = cache_dir

    def load_transcripts(self):
        # Load transcripts from the chosen corpus
        if self.corpus_name == 'AMI':
            # Choose a valid configuration, for example, 'headset-single'
            config = 'headset-single'

            # Specify the cache directory
            dataset = load_dataset('ami', config, cache_dir=self.cache_dir)
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

# # Example usage:
# corpus_name = 'AMI'  # Change to 'ISCI' if needed
# cache_dir = os.path.join('data', 'cache')  
# data_prep = DataPreparation(corpus_name, cache_dir=cache_dir)
# transcripts_df = data_prep.load_transcripts()
# baseline_transcripts_df = data_prep.create_baseline_transcripts(transcripts_df, sample_size=100)