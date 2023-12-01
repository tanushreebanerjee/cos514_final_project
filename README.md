# Counterfactual Analysis for Spoken Dialog Summarization

COS514 Final Project, Fall 2023, Princeton University
Authors: Tanushree Banerjee, Kiyosu Maeda
Instructor: Prof. Sanjeev Arora

This project aims to analyze the impact of speaker diarization and speech recognition errors on the quality of text summarization in spoken dialogue. By injecting artificial errors into transcripts and leveraging large language models (LLMs), we aim to understand how errors in upstream tasks affect downstream summarization tasks.

## Project Structure
```css
project-root/
├── data/: Contains raw and processed data.
    ├── raw/: Original data from ICSI or AMI corpus.
    └── processed/: Processed data for the project.
├── src/: Python source code for different stages.
    ├── data_preparation.py: Script for loading and preparing data.
    ├── error_injection.py: Script for injecting errors into transcripts.
    ├── summarization.py: Script for summarizing transcripts.
    └── evaluation.py: Script for evaluating the summarization.
├── requirements.txt: List of Python dependencies for easy installation.
├── LICENSE: License information for your project.
└── README.md: Project documentation.
```

## Getting Started

1. Clone this repository.
```bash
    git clone REPO_URL
```

3. Set up a conda environment and install dependencies from the `requirements.txt` file:
```bash
    conda create --name cos514 --file requirements.txt
```

Alternatively, you may use the `install.sh` script to create the environment and install the dependencies:
```bash
    bash scripts/install.sh
```

3. Activate the conda environment:
```bash
    conda activate cos514
```
4. Set OpenAI API key as the environment variable.
```bash
    export OPENAI_API_KEY=YOUR_API_KEY
```

5. Run the main script:
```bash
    python main.py
```

## License
This project is licensed under the [MIT License](LICENSE).
