import os
import pandas as pd
from src.summarization_models import ChatGPTSummarizer, XwinSummarizer, VicunaSummarizer, MPTSummarizer, Llama2Summarizer
from rouge import Rouge

class SummarizationPipeline:
    def __init__(self, corpus_df):
        self.corpus_df = corpus_df.copy()
        self.results_df = pd.DataFrame(columns=['Model', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L'])

    def create_baseline_summaries(self):
        chatgpt4_summarizer = ChatGPTSummarizer()
        self.corpus_df['baseline_summary'] = self.corpus_df['text'].apply(chatgpt4_summarizer.summarize)

    def summarize_with_models(self, model_summarizer):
        model_name = model_summarizer.name
        self.corpus_df[f'{model_name.lower()}_summary'] = self.corpus_df['text'].apply(model_summarizer.summarize)

    def evaluate_summaries(self):
        rouge = Rouge()

        baseline_summary = self.corpus_df['baseline_summary'].tolist()

        for model_name in ['Xwin', 'Vicuna', 'MPT', 'Llama2']:
            model_summary = self.corpus_df[f'{model_name.lower()}_summary'].tolist()
            scores = rouge.get_scores(hyps=model_summary, refs=baseline_summary, avg=True)

            # Save evaluation results to results_df
            self.results_df = self.results_df.append({
                'Model': model_name,
                'ROUGE-1': scores['rouge-1']['f'],
                'ROUGE-2': scores['rouge-2']['f'],
                'ROUGE-L': scores['rouge-l']['f']
            }, ignore_index=True)

    def save_summaries_to_csv(self, output_dir='results/summaries'):
        os.makedirs(output_dir, exist_ok=True)

        for col in self.corpus_df.columns:
            if col.endswith('_summary'):
                model_name = col.replace('_summary', '')
                output_csv = os.path.join(output_dir, f'{model_name}_summaries.csv')
                self.corpus_df[['id', 'text', col]].to_csv(output_csv, index=False)

    def save_evaluation_results_to_csv(self, output_file='results/evaluation_results.csv'):
        self.results_df.to_csv(output_file, index=False)

def load_corpus_csv(csv_file):
    return pd.read_csv(csv_file)

# Example usage
# Load your processed corpus CSV file
corpus_name = 'ami-corpus'
processed_corpus_path = os.path.join(os.getcwd(), 'data', 'processed', f'{corpus_name}.csv')
processed_corpus_df = load_corpus_csv(processed_corpus_path)

# Create SummarizationPipeline
summarization_pipeline = SummarizationPipeline(processed_corpus_df)

# Create baseline summaries using ChatGPT-4
summarization_pipeline.create_baseline_summaries()

# Summarize with Xwin model
xwin_summarizer = XwinSummarizer()
summarization_pipeline.summarize_with_models(xwin_summarizer)

# Summarize with Vicuna model
vicuna_summarizer = VicunaSummarizer()
summarization_pipeline.summarize_with_models(vicuna_summarizer)

# Summarize with MPT model
mpt_summarizer = MPTSummarizer()
summarization_pipeline.summarize_with_models(mpt_summarizer)

# Summarize with Llama2 model
llama2_summarizer = Llama2Summarizer()
summarization_pipeline.summarize_with_models(llama2_summarizer)

# Evaluate the summaries
summarization_pipeline.evaluate_summaries()

# Save summaries to CSV
summarization_pipeline.save_summaries_to_csv()

# Save evaluation results to CSV
summarization_pipeline.save_evaluation_results_to_csv()
