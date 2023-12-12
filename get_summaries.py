from pathlib import Path

from src.summarization_models import SummarizationModel


def main():
    project_root = Path('WRITE THE ABSOLUTE PATH TO THE PROJECT')
    data_dir = project_root / 'data'
    ami_dir = data_dir / 'AMI'

    model_name = xwin
    model_root = Path('WRITE THE ABSOLUTE PATH TO THE MODEL DIR')
    model_hash = 'Xwin-LM-7B-V0.1'
    model_dir = model_root / model_hash
    summarization_model = SummarizationModel(model_dir)

    meeting_id_list = [item.name for item in ami_dir.iterdir() if item.is_dir()]
    for meeting_id in meeting_id_list:
        print(meeting_id)
        meeting_dir = ami_dir / meeting_id
        
        for i in range(9):
            wer = i * 5
            error_condition = f'wer_{wer}_der_0'
            input_file_name = f'segments_{error_condition}.txt'
            dialogue = ''
            with open(meeting_dir / input_file_name, 'r') as f:
                for line in f:
                    dialogue += line

            # abstractive, unaware
            outputs = summarization_model(
                dialogue,
                is_abstractive=True,
                error_aware=False,
            )
            print(outputs)
            output_file_name = f'abstractive_unaware_{error_condition}_{model_name}.txt'
            with open(output_file_name, 'w') as f:
                f.write(outputs)
            
            # abstractive, aware
            outputs = summarization_model(
                dialogue,
                is_abstractive=True,
                error_aware=True,
            )
            print(outputs)
            output_file_name = f'abstractive_aware_{error_condition}_{model_name}.txt'
            with open(output_file_name, 'w') as f:
                f.write(outputs)
            
            # extractive, unaware
            outputs = summarization_model(
                dialogue,
                is_abstractive=False,
                error_aware=False,
            )
            print(outputs)
            output_file_name = f'extractive_unaware_{error_condition}_{model_name}.txt'
            with open(output_file_name, 'w') as f:
                f.write(outputs)
            
            # extractive, aware
            outputs = summarization_model(
                dialogue,
                is_abstractive=False,
                error_aware=True,
            )
            print(outputs)
            output_file_name = f'extractive_aware_{error_condition}_{model_name}.txt'
            with open(output_file_name, 'w') as f:
                f.write(outputs)

        for i in range(1, 9):
            der = i * 5
            error_condition = f'wer_0_der_{der}'
            input_file_name = f'segments_{error_condition}.txt'
            dialogue = ''
            with open(meeting_dir / input_file_name, 'r') as f:
                for line in f:
                    dialogue += line

            # abstractive, unaware
            outputs = summarization_model(
                dialogue,
                is_abstractive=True,
                error_aware=False,
            )
            print(outputs)
            output_file_name = f'abstractive_unaware_{error_condition}_{model_name}.txt'
            with open(output_file_name, 'w') as f:
                f.write(outputs)
            
            # abstractive, aware
            outputs = summarization_model(
                dialogue,
                is_abstractive=True,
                error_aware=True,
            )
            print(outputs)
            output_file_name = f'abstractive_aware_{error_condition}_{model_name}.txt'
            with open(output_file_name, 'w') as f:
                f.write(outputs)
            
            # extractive, unaware
            outputs = summarization_model(
                dialogue,
                is_abstractive=False,
                error_aware=False,
            )
            print(outputs)
            output_file_name = f'extractive_unaware_{error_condition}_{model_name}.txt'
            with open(output_file_name, 'w') as f:
                f.write(outputs)
            
            # extractive, aware
            outputs = summarization_model(
                dialogue,
                is_abstractive=False,
                error_aware=True,
            )
            print(outputs)
            output_file_name = f'extractive_aware_{error_condition}_{model_name}.txt'
            with open(output_file_name, 'w') as f:
                f.write(outputs)


if __name__ == '__main__':
    main()
