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

We conduct a technical experiment using the ICSI or AMI corpus, injecting errors into transcripts and summarizing them with various LLMs. The impact of errors is then evaluated using ROUGE-N metrics.

For detailed information, refer to the [Methodology](notebooks/methodology.ipynb) notebook.

## Project Structure
```css
project-root/
│
├── data/
├── notebooks/
├── src/
├── models/
├── requirements.txt
└── README.md
```

For a detailed explanation, see [Project Structure](#project-structure).

## Getting Started

1. Clone this repository.
    ```bash
    git clone REPO_URL
    ```

2. Set up a conda environment and install dependencies:
   ```bash
   bash scripts/install.sh


## License
This project is licensed under the MIT License.
