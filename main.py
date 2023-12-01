import os
import yaml
from src.data_preparation import DataPreparation
from src.error_injection import ErrorInjection
from src.summarization import Summarization
from src.evaluation import Evaluation
from src.error_correction import ErrorCorrection

# Set OpenAI API key as an environment variable
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
os.environ["HF_DATASETS_CACHE"] = os.path.join('data', 'cache')

# Function to load experiment configurations from YAML files
def load_experiment_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

# Function to run an experiment based on the provided configuration
def run_experiment(config):
    # Data Preparation
    cache_dir = os.path.join('data', 'cache')  
    data_prep = DataPreparation(corpus_name=config['corpus_name'], cache_dir=cache_dir)
    transcripts_df = data_prep.load_transcripts()
    baseline_transcripts_df = data_prep.create_baseline_transcripts(transcripts_df, sample_size=config['sample_size'])

    # Error Injection
    error_injector = ErrorInjection(config['error_types'])
    transcripts_with_errors = [error_injector.inject_errors(t, degree=config['degree_of_error']) for t in baseline_transcripts_df['transcript']]

    # Summarization
    summarizer = Summarization(model_name=config['model_name'])
    summaries = [summarizer.summarize(t) for t in transcripts_with_errors]

    # Evaluation
    evaluation = Evaluation()
    references = baseline_transcripts_df['reference_summary']

    # Calculate and store ROUGE scores
    rouge_scores = []
    for i in range(len(transcripts_with_errors)):
        rouge_scores.append(evaluation.calculate_rouge(references[i], summaries[i]))

    # Store results (adjust the storage method based on your needs)
    result_path = f'results/{config["experiment_name"]}_without_correction_results.yaml'
    with open(result_path, 'w') as result_file:
        yaml.dump({'ROUGE_scores': rouge_scores}, result_file)

    if config['error_correction_model']:
        # Run with error correction
        error_correction = ErrorCorrection(model=config['error_correction_model'])
        transcripts_with_errors = [error_correction.correct_errors(t) for t in transcripts_with_errors]
        summaries_with_correction = [summarizer.summarize(t) for t in transcripts_with_errors]

        # Calculate and store ROUGE scores with error correction
        rouge_scores_correction = []
        for i in range(len(transcripts_with_errors)):
            rouge_scores_correction.append(evaluation.calculate_rouge(references[i], summaries_with_correction[i]))

        # Store results with error correction
        result_path_correction = f'results/{config["experiment_name"]}_with_correction_results.yaml'
        with open(result_path_correction, 'w') as result_file_correction:
            yaml.dump({'ROUGE_scores': rouge_scores_correction}, result_file_correction)

# Create a directory for result files
os.makedirs('results', exist_ok=True)

# Run experiments for all generated configurations
config_files = [f for f in os.listdir('configs') if f.endswith('.yaml')]
for config_file in config_files:
    config_path = os.path.join('configs', config_file)
    config = load_experiment_config(config_path)

    # Run without error correction
    run_experiment(config)

    # Run with error correction if the model is specified
    if config['error_correction_model']:
        run_experiment(config)
