import yaml
import os
from itertools import product

# Define model names, error types, and degrees
models = ["Xwin-LM-7B-v0.1", "LLAMA-2-7B-Chat", "MPT-7B-chat", "vicuna-7b-v1.3"]
error_types = ['speaker_identification', 'speech_recognition']
degrees = [0.1, 0.2, 0.3]  # Adjust as needed

# Define other parameters for experiments
sample_sizes = [100, 200, 300]  # Adjust as needed
corpus_names = ['ami', 'isci']  # Adjust as needed

# Generate all combinations of parameters
experiments = list(product(models, error_types, degrees, sample_sizes, corpus_names))

# Function to generate a configuration file for each experiment
def generate_config_file(experiment):
    model, error_type, degree, sample_size, corpus_name = experiment
    config = {
        'experiment_name': f'{model}_{error_type}_error_{degree}_sample_{sample_size}_{corpus_name}',
        'model_name': model,
        'error_types': [error_type],
        'degree_of_error': degree,
        'sample_size': sample_size,
        'corpus_name': corpus_name,
        # Add other experiment-specific parameters as needed
    }

    # Write the configuration to a YAML file
    with open(f'configs/{config["experiment_name"]}.yaml', 'w') as file:
        yaml.dump(config, file)

# Create a directory for configuration files
import os
os.makedirs('configs', exist_ok=True)

# Generate configuration files for all experiments
for experiment in experiments:
    generate_config_file(experiment)
