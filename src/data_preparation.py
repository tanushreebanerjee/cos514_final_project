# Import necessary libraries
import pandas as pd

# Load transcripts from the chosen corpus (ICSI or AMI)
# Ensure you have the necessary data files and adjust the paths accordingly
transcripts_df = pd.read_csv('path/to/transcripts.csv')

# Separate transcripts and annotations
transcripts = transcripts_df['transcript_column']
annotations = transcripts_df['annotation_column']
