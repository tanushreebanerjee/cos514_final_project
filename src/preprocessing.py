import os
import json
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def process_text(self, text):
        # Tokenization
        tokens = word_tokenize(text)

        # Lowercasing
        tokens = [token.lower() for token in tokens]

        # Removing stopwords
        tokens = [token for token in tokens if token not in self.stop_words]

        # Stemming
        tokens = [self.stemmer.stem(token) for token in tokens]

        return tokens
    
class CorpusProcessor:
    def __init__(self, corpus_type, meeting_id):
        self.corpus_type = corpus_type
        self.meeting_id = meeting_id
        self.text_preprocessor = TextPreprocessor()

    def process_entry(self, entry):
        if 'text' in entry:
            processed_text = self.text_preprocessor.process_text(entry['text'])
            entry['processed_text'] = ' '.join(processed_text)
        else:
            entry['processed_text'] = ''  # Handle the case when 'text' is not present
        return entry

    def process_corpus(self, corpus):
        return [self.process_entry(entry) for entry in corpus]

def load_corpus(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def process_all_files(directory, output_csv):
    data_list = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                corpus_type = os.path.basename(root)
                meeting_id = os.path.splitext(file)[0]
                
                # Create a CorpusProcessor for each file
                corpus_processor = CorpusProcessor(corpus_type, meeting_id)
                
                # Process the current JSON file
                corpus_data = load_corpus(file_path)
                processed_corpus = corpus_processor.process_corpus(corpus_data)
                
                # Append data to the list
                data_list.extend(processed_corpus)

    # Create a Pandas DataFrame
    df = pd.DataFrame(data_list)

    # Save DataFrame as a CSV file
    df.to_csv(output_csv, index=False)

# Example usage
nltk.download('stopwords')
nltk.download('punkt')
ami_raw_dir = os.path.join(os.getcwd(), 'data/raw/ami-corpus')
ami_processed_csv = os.path.join(os.getcwd(), 'data/processed/ami-corpus.csv')
os.makedirs(os.path.dirname(ami_processed_csv), exist_ok=True)
process_all_files(ami_raw_dir, ami_processed_csv)

# print a sample of the processed transcripts
processed_corpus_df = pd.read_csv(ami_processed_csv)

# Print the first 5 rows
print(processed_corpus_df.head())