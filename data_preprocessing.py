import json
from pathlib import Path

import pandas as pd

from src.counterfactual_generation import CounterfactualGenerator


def main():
    project_root Path('WRITE THE ABSOLUTE PATH TO THE PROJECT')
    data_dir = project_root / 'data'
    ami_dir = data_dir / 'AMI'

    segments_annotation = 'segments_wer_0_der_0'
    abstractive_annotation = 'abstractive_annotation'
    extractive_annotation = 'extractive_annotation'

    counterfactual_generator = CounterfactualGenerator()

    meeting_id_list = [item.name for item in ami_dir.iterdir() if item.is_dir()]
    for meeting_id in meeting_id_list:
        meeting_dir = ami_dir / meeting_id

        # save segment annotation
        with open(meeting_dir / f'{segments_annotation}.json', 'r') as f:
            segments = json.load(f)
        with open(meeting_dir / f'{segments_annotation}.txt', 'w') as f:
            for segment in segments:
                speaker = segment['speaker']
                text = segment['text']
                f.write(f'{speaker}: {text}\n')
        
        # save counterfactual segment annotation
        # DER
        for i in range(1, 9):
            der = i * 5
            counterfactual_segments = counterfactual_generator.change_speaker_names(segments, der/100)
            file_name = f'segments_wer_0_der_{der}.txt'
            with open(meeting_dir / file_name, 'w') as f:
                for segment in counterfactual_segments:
                    speaker = segment['speaker']
                    text = segment['text']
                    f.write(f'{speaker}: {text}\n')

        # WER
        for i in range(1, 9):
            wer = i * 5
            counterfactual_segments = counterfactual_generator.change_utterances(segments, wer/100)
            file_name = f'segments_wer_{wer}_der_0.txt'
            with open(meeting_dir / file_name, 'w') as f:
                for segment in counterfactual_segments:
                    speaker = segment['speaker']
                    text = segment['text']
                    f.write(f'{speaker}: {text}\n')
        
        # save abstractive summary
        with open(meeting_dir / f'{abstractive_annotation}.json', 'r') as f:
            abstractive_summary = json.load(f)
        with open(meeting_dir / f'{abstractive_annotation}.txt', 'w') as f:
            for summary in abstractive_summary:
                text = summary['text']
                f.write(f'{text}\n')

        # save extractive summary
        with open(meeting_dir / f'{extractive_annotation}.json', 'r') as f:
            extractive_summary = json.load(f)
        with open(meeting_dir / f'{extractive_annotation}.txt', 'w') as f:
            for summary in extractive_summary:
                text = summary['text']
                f.write(f'{text}\n')


if __name__ == '__main__':
    main()
