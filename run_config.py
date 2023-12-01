# run_config.py
import subprocess
import argparse
import os

def run_pipeline(config_file):
    # Example command to run the main pipeline using the provided config file
    command = f"python main.py --config {config_file}"
    
    try:
        # Run the pipeline using subprocess
        subprocess.run(command, shell=True, check=True)
        print(f"Pipeline completed successfully for {config_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error running the pipeline for {config_file}. Error: {e}")
        sys.exit(1)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run pipeline with specified configuration file.")
    parser.add_argument('--config', required=True, help='Path to the configuration file.')
    args = parser.parse_args()

    config_file = args.config

    # Check if the provided file exists
    if not os.path.isfile(config_file):
        print(f"Error: Configuration file '{config_file}' not found.")
        sys.exit(1)

    # Run the main pipeline with the provided configuration file
    run_pipeline(config_file)

if __name__ == "__main__":
    main()
