# run_experiments.py
import os
import subprocess
import argparse

def run_experiment(config_path, results_output_path):
    # Run the pipeline with the specified config file
    command = f"python run_config.py --config {config_path} --results {results_output_path}"
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running the pipeline for {config_path}. Error: {e}")

def run_all_experiments(configs_directory, results_output_path):
    # Iterate over all files in the configs directory
    for filename in os.listdir(configs_directory):
        if filename.endswith(".json"):  # Assuming your config files have a .json extension
            config_path = os.path.join(configs_directory, filename)
            run_experiment(config_path, results_output_path)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run experiments with multiple configuration files.")
    parser.add_argument('--configs', required=True, help='Path to the directory containing configuration files.')
    parser.add_argument('--results', required=True, help='Path to the results output directory.')
    args = parser.parse_args()

    configs_directory = args.configs
    results_output_path = args.results

    # Check if the provided directory exists
    if not os.path.exists(configs_directory):
        print(f"Error: Configuration directory '{configs_directory}' not found.")
        sys.exit(1)

    # Check if the provided results output path exists
    if not os.path.exists(results_output_path):
        os.makedirs(results_output_path)

    # Run experiments with all config files in the specified directory
    run_all_experiments(configs_directory, results_output_path)

if __name__ == "__main__":
    main()
