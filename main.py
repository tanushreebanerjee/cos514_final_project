# main.py
# Add these imports to main.py
from collections import namedtuple
from src.speech_recognition import ASRPipeline
import json
import os
import pandas as pd
from src.preprocessing import process_all_files, load_corpus
from src.counterfactual_generation import CounterfactualGenerator, compute_der
from src.error_correction import ErrorCorrection
from src.summarization_models import (
    ChatGPTSummarizer, XwinSummarizer, VicunaSummarizer, MPTSummarizer, Llama2Summarizer
)
from src.summarization import SummarizationPipeline
from src.evaluation import EvaluationPipeline

# Define ASRResult namedtuple
ASRResult = namedtuple('ASRResult', ['id', 'reference', 'hypothesis'])


def run_pipeline(config):
    # Load configuration from JSON file
    with open(config, 'r') as f:
        config_data = json.load(f)


    # Example: Accessing configuration parameters
    dataset = config_data.get('dataset', 'ami')  # Default to 'ami' if not specified
    summarization_model = config_data.get('summarization_model', 'gpt4')  # Default to 'gpt4' if not specified
    correct_errors = config_data.get('correct_errors', False)
    llm_model = config_data.get('llm_model', 'gpt3.5')  # Default to 'gpt3.5' if not specified
    counterfactual_type = config_data.get('counterfactual_type', 'change_speakers')  # Default to 'change_speakers' if not specified

    # Your actual pipeline logic goes here based on the provided configuration
    print(f"Running pipeline with config: {config}")

    # Step 1: Preprocessing
    raw_data_dir = os.path.join(os.getcwd(), f'data/raw/{dataset}-corpus')
    processed_csv_path = os.path.join(os.getcwd(), f'data/processed/{dataset}-corpus.csv')
    os.makedirs(os.path.dirname(processed_csv_path), exist_ok=True)
    process_all_files(raw_data_dir, processed_csv_path)

    # Step 2: Speech Recognition (ASR)
    asr_model_name = config_data.get('asr_model', 'whisper-large')  # Default to 'whisper-large' if not specified
    asr_audio_directory = config_data.get('asr_audio_directory', 'path_to_your_audio_files')  # Replace with the actual path
    asr_output_csv = os.path.join(os.getcwd(), 'results/asr_results.csv')
    asr_results_df = run_asr_pipeline(asr_model_name, asr_audio_directory, asr_output_csv)

    # Step 3: Counterfactual Generation
    counterfactual_generator = CounterfactualGenerator(asr_results_df)
    if counterfactual_type == 'change_speakers':
        counterfactual_generator.change_speakers_randomly()
    elif counterfactual_type == 'change_sentences':
        counterfactual_generator.change_sentences()
    counterfactual_csv_path = os.path.join(os.getcwd(), f'data/counterfactual/{dataset}-counterfactual-{counterfactual_type}.csv')
    os.makedirs(os.path.dirname(counterfactual_csv_path), exist_ok=True)
    counterfactual_generator.save_counterfactual_to_csv(counterfactual_csv_path)

    # Step 4: Error Correction
    if correct_errors:
        error_correction = ErrorCorrection(model=llm_model)
        corrected_results_df = run_error_correction(asr_results_df, error_correction)
    else:
        corrected_results_df = asr_results_df.copy()

    # Step 5: Summarization
    summarization_pipeline = SummarizationPipeline(corrected_results_df)
    summarizer = get_summarizer_instance(summarization_model)
    summarization_pipeline.summarize_with_model(summarizer)

    # Step 6: Evaluation
    evaluation_pipeline = EvaluationPipeline(summarization_pipeline.results_df)
    evaluation_pipeline.calculate_rouge_n()

    # Step 7: Save Results
    summarization_pipeline.save_summaries_to_csv()
    summarization_pipeline.save_evaluation_results_to_csv()
    evaluation_pipeline.save_rouge_n_results_to_csv()

    print("Pipeline completed successfully.")

def run_asr_pipeline(model_name, audio_directory, output_csv):
    # Your ASR pipeline logic goes here
    print("Running ASR pipeline...")
    # Example: You can use the ASRPipeline class to transcribe audio
    asr_pipeline = ASRPipeline(model_name)
    asr_results = []

    for audio_file in os.listdir(audio_directory):
        audio_path = os.path.join(audio_directory, audio_file)
        transcription = asr_pipeline.transcribe_audio(audio_path)
        asr_results.append(ASRResult(id=audio_file, reference="TODO", hypothesis=" ".join(transcription)))

    asr_df = pd.DataFrame(asr_results)
    asr_df.to_csv(output_csv, index=False)
    print(f'ASR Results saved to {output_csv}')

    return asr_df

def run_error_correction(df, error_correction):
    # Your error correction pipeline logic goes here
    print("Running Error Correction pipeline...")
    # Example: You can use the ErrorCorrection class to correct errors in the text
    df['corrected_text'] = df['text'].apply(error_correction.correct_errors)
    return df

def get_summarizer_instance(model_name):
    if model_name.lower() == 'gpt4':
        return ChatGPTSummarizer(model_name="gpt-4")
    elif model_name.lower() == 'chatgpt':
        return ChatGPTSummarizer()
    elif model_name.lower() == 'xwin':
        return XwinSummarizer()
    elif model_name.lower() == 'vicuna':
        return VicunaSummarizer()
    elif model_name.lower() == 'mpt':
        return MPTSummarizer()
    elif model_name.lower() == 'llama2':
        return Llama2Summarizer()
    else:
        raise ValueError(f"Unsupported summarization model: {model_name}")
