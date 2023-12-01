# generate_configs.py
import itertools

def generate_configs():
    datasets = ['ami', 'icsi', 'whisper']
    summarization_models = ['gpt4', 'gpt-3.5-turbo', 'Xwin', 'vicuna', 'mpt', 'Llama2']
    error_correction_options = [False, True]  # Whether to correct errors or not
    llm_models = ['gpt3.5', 'gpt4']  # If correcting errors, choose LLM model
    counterfactual_types = ['change_speakers', 'change_sentences']  # Types of counterfactuals

    # Generate all possible combinations
    all_configs = list(itertools.product(datasets, summarization_models, error_correction_options, llm_models, counterfactual_types))

    # Map the combinations to a dictionary for better readability
    configs_dict_list = []
    for config in all_configs:
        config_dict = {
            'dataset': config[0],
            'summarization_model': config[1],
            'correct_errors': config[2],
            'llm_model': config[3],
            'counterfactual_type': config[4]
        }
        configs_dict_list.append(config_dict)

    return configs_dict_list

def main():
    generated_configs = generate_configs()

    # Print or save the generated configurations
    for idx, config in enumerate(generated_configs, start=1):
        print(f"Config {idx}: {config}")

if __name__ == "__main__":
    main()

# import yaml
# import os
# from itertools import product

# # Define model names, error types, and degrees
# models = ["Xwin-LM-7B-v0.1", "LLAMA-2-7B-Chat", "MPT-7B-chat", "vicuna-7b-v1.3"]
# error_types = ['speaker_identification', 'speech_recognition']
# degrees = [0.1, 0.2, 0.3]  # Adjust as needed

# # Define other parameters for experiments
# sample_sizes = [100, 200, 300]  # Adjust as needed
# corpus_names = ['ami', 'icsi']  # Adjust as needed
# error_correction_models = ["gpt-3.5-turbo", "gpt-4", None]
# # Generate all combinations of parameters
# experiments = list(product(models, error_types, degrees, sample_sizes, corpus_names, error_correction_models))

# # Function to generate a configuration file for each experiment
# def generate_config_file(experiment):
#     model, error_type, degree, sample_size, corpus_name, error_correction_model = experiment
#     config = {
#         'experiment_name': f'{model}_{error_type}_error_{degree}_sample_{sample_size}_{corpus_name}_correction_{error_correction_model}',
#         'model_name': model,
#         'error_types': [error_type],
#         'degree_of_error': degree,
#         'sample_size': sample_size,
#         'corpus_name': corpus_name,
#         'error_correction_model': error_correction_model if error_correction_model else 'None'
#         # Add other experiment-specific parameters as needed
#     }

#     # Write the configuration to a YAML file
#     with open(f'configs/{config["experiment_name"]}.yaml', 'w') as file:
#         yaml.dump(config, file)

# # Create a directory for configuration files
# import os
# os.makedirs('configs', exist_ok=True)

# # Generate configuration files for all experiments
# for experiment in experiments:
#     generate_config_file(experiment)
