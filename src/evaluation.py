import pandas as pd
from rouge import Rouge
from nltk.metrics import edit_distance
from collections import namedtuple
import matplotlib.pyplot as plt

class Evaluation:
    def __init__(self, processed_corpus_df):
        self.processed_corpus_df = processed_corpus_df.copy()
        self.results_df = pd.DataFrame(columns=['Model', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'WER', 'DER'])

    def calculate_rouge(self, reference, hypothesis):
        rouge = Rouge()
        scores = rouge.get_scores(hyps=hypothesis, refs=reference, avg=True)
        return scores

    def calculate_rouge_n(self, reference, hypothesis, n=1):
        # Assuming reference and hypothesis are strings
        rouge = Rouge()
        scores = rouge.get_scores(hyps=hypothesis, refs=reference, avg=True)
        return scores[f'rouge-{n}']['f']

    def calculate_wer(self, reference, hypothesis):
        # Assuming reference and hypothesis are strings
        return edit_distance(reference.split(), hypothesis.split())

    def calculate_der(self, reference_annotation, hypothesis_annotation):
        # Assuming reference_annotation and hypothesis_annotation are lists of namedtuples (e.g., Segment)
        return self.calculate_wer(reference_annotation, hypothesis_annotation)

    def evaluate_model(self, model_name, hypothesis_column):
        reference_summary = self.processed_corpus_df['text'].tolist()
        hypothesis_summary = self.processed_corpus_df[hypothesis_column].tolist()

        # ROUGE calculation
        rouge_1 = self.calculate_rouge_n(reference_summary, hypothesis_summary, n=1)
        rouge_2 = self.calculate_rouge_n(reference_summary, hypothesis_summary, n=2)
        rouge_l = self.calculate_rouge_n(reference_summary, hypothesis_summary, n='l')

        # WER calculation
        wer_scores = [self.calculate_wer(reference, hypothesis) for reference, hypothesis in zip(reference_summary, hypothesis_summary)]
        avg_wer = sum(wer_scores) / len(wer_scores)

        # DER calculation (assuming annotation columns are available in the corpus)
        der_scores = [self.calculate_der(reference_annotation, hypothesis_annotation)
                      for reference_annotation, hypothesis_annotation in zip(self.processed_corpus_df['annotation'], self.processed_corpus_df[f'{model_name.lower()}_annotation'])]
        avg_der = sum(der_scores) / len(der_scores)

        # Save results to results_df
        self.results_df = self.results_df.append({
            'Model': model_name,
            'ROUGE-1': rouge_1,
            'ROUGE-2': rouge_2,
            'ROUGE-L': rouge_l,
            'WER': avg_wer,
            'DER': avg_der
        }, ignore_index=True)

    def save_evaluation_results_to_csv(self, output_file='data/evaluation_results.csv'):
        self.results_df.to_csv(output_file, index=False)
        print(f'Evaluation Results saved to {output_file}')
        
    def plot_wer_rouge(self, output_file='data/wer_rouge_plot.png'):
        plt.scatter(self.results_df['WER'], self.results_df['ROUGE-1'], label='ROUGE-1')
        plt.scatter(self.results_df['WER'], self.results_df['ROUGE-2'], label='ROUGE-2')
        plt.scatter(self.results_df['WER'], self.results_df['ROUGE-L'], label='ROUGE-L')
        plt.xlabel('Word Error Rate (WER)')
        plt.ylabel('ROUGE Score')
        plt.legend()
        plt.title('WER x ROUGE-N')
        plt.savefig(output_file)
        plt.show()

    def plot_der_rouge(self, output_file='data/der_rouge_plot.png'):
        plt.scatter(self.results_df['DER'], self.results_df['ROUGE-1'], label='ROUGE-1')
        plt.scatter(self.results_df['DER'], self.results_df['ROUGE-2'], label='ROUGE-2')
        plt.scatter(self.results_df['DER'], self.results_df['ROUGE-L'], label='ROUGE-L')
        plt.xlabel('Diarization Error Rate (DER)')
        plt.ylabel('ROUGE Score')
        plt.legend()
        plt.title('DER x ROUGE-N')
        plt.savefig(output_file)
        plt.show()

# Define a named tuple for annotation segments
Segment = namedtuple('Segment', ['start', 'end', 'speaker'])

def load_summaries_and_annotations(csv_file):
    df = pd.read_csv(csv_file)
    # Assuming 'start', 'end', and 'speaker' columns are available in the CSV
    annotations = [Segment(row['start'], row['end'], row['speaker']) for _, row in df.iterrows()]
    return df['text'].tolist(), annotations

def load_corpus_csv(csv_file):
    return pd.read_csv(csv_file)

def main():
    # Load your processed corpus CSV file
    processed_corpus_df = load_corpus_csv('data/processed/ami-corpus.csv')

    # Load baseline and model summaries with annotations
    baseline_summaries, baseline_annotations = load_summaries_and_annotations('data/summaries/baseline_summaries.csv')
    xwin_summaries, xwin_annotations = load_summaries_and_annotations('data/summaries/xwin_summaries.csv')
    vicuna_summaries, vicuna_annotations = load_summaries_and_annotations('data/summaries/vicuna_summaries.csv')
    mpt_summaries, mpt_annotations = load_summaries_and_annotations('data/summaries/mpt_summaries.csv')
    llama2_summaries, llama2_annotations = load_summaries_and_annotations('data/summaries/llama2_summaries.csv')

    # Create Evaluation instance
    evaluation = Evaluation(processed_corpus_df)

    # Evaluate baseline and models
    evaluation.evaluate_model('Baseline', 'baseline_summary')
    evaluation.evaluate_model('Xwin', 'xwin_summary')
    evaluation.evaluate_model('Vicuna', 'vicuna_summary')
    evaluation.evaluate_model('MPT', 'mpt_summary')
    evaluation.evaluate_model('Llama2', 'llama2_summary')

    # Save evaluation results to CSV
    evaluation.save_evaluation_results_to_csv()
    
    # Plot WER x ROUGE-N
    evaluation.plot_wer_rouge()

    # Plot DER x ROUGE-N
    evaluation.plot_der_rouge()

if __name__ == "__main__":
    main()
