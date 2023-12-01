import os
import pandas as pd
from src.preprocessing import process_all_files, load_corpus_csv
# from src.speech_recognition import SpeechRecognitionPipeline
from src.counterfactual_generation import CounterfactualPipeline
from src.summarization import SummarizationPipeline
from src.evaluation import EvaluationPipeline
from src.summarization_models import XwinSummarizer, VicunaSummarizer, MPTSummarizer, Llama2Summarizer

def main():
    # Create GitHub repository (if not already created)

    # Dataset preprocessing
    corpus_name = 'ami-corpus'  # Decide on the dataset to use
    raw_data_dir = os.path.join('data', 'raw', corpus_name)
    processed_corpus_path = os.path.join('data', 'processed', f'{corpus_name}.csv')
    process_all_files(raw_data_dir, processed_corpus_path)
    processed_corpus_df = load_corpus_csv(processed_corpus_path)

    # # Speech Recognition
    # speech_recognition_pipeline = SpeechRecognitionPipeline(processed_corpus_df)
    # speech_recognition_pipeline.execute_whisper_small()  # Use Whisper (small)
    # speech_recognition_pipeline.calculate_wer()

    # Counterfactual Generation
    counterfactual_pipeline = CounterfactualPipeline(processed_corpus_df)
    counterfactual_pipeline.generate_random_speaker_changes()
    counterfactual_pipeline.measure_differences_by_der()

    # Summarization
    summarization_pipeline = SummarizationPipeline(processed_corpus_df)
    summarization_pipeline.create_baseline_summaries()

    # Decide models to use for summarization
    xwin_summarizer = XwinSummarizer()
    vicuna_summarizer = VicunaSummarizer()
    mpt_summarizer = MPTSummarizer()
    llama2_summarizer = Llama2Summarizer()

    summarization_pipeline.summarize_with_models(xwin_summarizer)
    summarization_pipeline.summarize_with_models(vicuna_summarizer)
    summarization_pipeline.summarize_with_models(mpt_summarizer)
    summarization_pipeline.summarize_with_models(llama2_summarizer)

    # Evaluation
    evaluation_pipeline = EvaluationPipeline(summarization_pipeline.results_df)
    evaluation_pipeline.calculate_rouge_n()
    
    # Save results
    summarization_pipeline.save_summaries_to_csv()
    summarization_pipeline.save_evaluation_results_to_csv()
    evaluation_pipeline.save_rouge_n_results_to_csv()
    
    # Save evaluation results to CSV
    evaluation_pipeline.save_evaluation_results_to_csv()
    evaluation_pipeline.plot_wer_rouge()  # Plot WER x ROUGE-N
    evaluation_pipeline.plot_der_rouge() # Plot DER x ROUGE-N
    
if __name__ == "__main__":
    main()


# import os
# import yaml
# from src.data_preparation import DataPreparation
# from src.error_injection import ErrorInjection
# from src.summarization import Summarization
# from src.evaluation import Evaluation
# from src.error_correction import ErrorCorrection

# # Set OpenAI API key as an environment variable
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
# os.environ["HF_DATASETS_CACHE"] = os.path.join('data', 'cache')

# # Function to load experiment configurations from YAML files
# def load_experiment_config(config_path):
#     with open(config_path, 'r') as file:
#         config = yaml.load(file, Loader=yaml.FullLoader)
#     return config

# # Function to run an experiment based on the provided configuration
# def run_experiment(config):
#     # Data Preparation
#     cache_dir = os.path.join('data', 'cache')  
#     data_prep = DataPreparation(corpus_name=config['corpus_name'], cache_dir=cache_dir)
#     transcripts_df = data_prep.load_transcripts()
#     baseline_transcripts_df = data_prep.create_baseline_transcripts(transcripts_df, sample_size=config['sample_size'])

#     # Error Injection
#     error_injector = ErrorInjection(config['error_types'])
#     transcripts_with_errors = [error_injector.inject_errors(t, degree=config['degree_of_error']) for t in baseline_transcripts_df['transcript']]

#     # Summarization
#     summarizer = Summarization(model_name=config['model_name'])
#     summaries = [summarizer.summarize(t) for t in transcripts_with_errors]

#     # Evaluation
#     evaluation = Evaluation()
#     references = baseline_transcripts_df['reference_summary']

#     # Calculate and store ROUGE scores
#     rouge_scores = []
#     for i in range(len(transcripts_with_errors)):
#         rouge_scores.append(evaluation.calculate_rouge(references[i], summaries[i]))

#     # Store results (adjust the storage method based on your needs)
#     result_path = f'results/{config["experiment_name"]}_without_correction_results.yaml'
#     with open(result_path, 'w') as result_file:
#         yaml.dump({'ROUGE_scores': rouge_scores}, result_file)

#     if config['error_correction_model']:
#         # Run with error correction
#         error_correction = ErrorCorrection(model=config['error_correction_model'])
#         transcripts_with_errors = [error_correction.correct_errors(t) for t in transcripts_with_errors]
#         summaries_with_correction = [summarizer.summarize(t) for t in transcripts_with_errors]

#         # Calculate and store ROUGE scores with error correction
#         rouge_scores_correction = []
#         for i in range(len(transcripts_with_errors)):
#             rouge_scores_correction.append(evaluation.calculate_rouge(references[i], summaries_with_correction[i]))

#         # Store results with error correction
#         result_path_correction = f'results/{config["experiment_name"]}_with_correction_results.yaml'
#         with open(result_path_correction, 'w') as result_file_correction:
#             yaml.dump({'ROUGE_scores': rouge_scores_correction}, result_file_correction)

# # Create a directory for result files
# os.makedirs('results', exist_ok=True)

# # Run experiments for all generated configurations
# config_files = [f for f in os.listdir('configs') if f.endswith('.yaml')]
# for config_file in config_files:
#     config_path = os.path.join('configs', config_file)
#     config = load_experiment_config(config_path)

#     # Run without error correction
#     run_experiment(config)

#     # Run with error correction if the model is specified
#     if config['error_correction_model']:
#         run_experiment(config)
