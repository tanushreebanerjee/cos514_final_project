# Counterfactual Analysis for Spoken Dialog Summarization Tasks
COS514 Final Project, Fall 2023, Princeton University

Tanushree Banerjee, Kiyosu Maeda

Instructor: Prof. Sanjeev Arora

## Overview

This project aims to analyze the impact of speaker diarization and speech recognition errors on the quality of text summarization in spoken dialogs. By injecting artificial errors into transcripts and leveraging large language models (LLMs), we aim to understand how errors in upstream tasks affect downstream summarization tasks.

## Table of Contents

- [Background](#background)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [License](#license)

## Background

Long-form spoken dialogs, such as podcasts, interviews, and meetings, often require summarization for better understanding. This project investigates the interplay between speaker identification, ASR accuracy, and summarization quality.

## Methodology

We conduct a technical experiment using the ICSI and AMI corpus, injecting errors into transcripts and summarizing them with various LLMs. The impact of errors is then evaluated using ROUGE-N metrics.

## Project Structure
```css
project-root/
│
├── data/: Contains raw and processed data.
    ├── raw/: Original data from ICSI or AMI corpus.
    ├── processed/: Processed data for the project.
├── src/: Python source code for different stages.
    ├── data_preparation.py: Script for loading and preparing data.
    ├── error_injection.py: Script for injecting errors into transcripts.
    ├── summarization.py: Script for summarizing transcripts.
    ├── evaluation.py: Script for evaluating the summarization.
├── models/: Pre-trained language models. Each model has its own folder containing necessary files.
├── requirements.txt: List of Python dependencies for easy installation.
├── LICENSE: License information for your project.
└── README.md: Project documentation.
```

See [project_structure.txt](project_structure.txt) for a more detailed description of project structure.

## Getting Started

1. Clone this repository.
    ```bash
    git clone REPO_URL
    ```

2. Set up a conda environment and install dependencies:
   ```bash
   bash scripts/install.sh

3. Activate the conda environment:
    ```bash
    conda activate cos514
    ```

4. Run the main script:
    ```bash
    python main.py
    ```

## License
This project is licensed under the MIT License.
