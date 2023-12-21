import copy
import random

import numpy as np
import pandas as pd


class CounterfactualGenerator:

    @staticmethod
    def get_total_word_num(segments):
        total_word_num = 0
        for segment in segments:
            text = segment['text']
            text = text.split(' ')
            total_word_num += len(text)
        return total_word_num

    def change_speaker_names(self, segments, error_rate):
        utterance_num = len(segments)
        speaker_name_error_num = int(utterance_num * error_rate)
        counterfactual_segments = copy.deepcopy(segments)
        random_index_list = random.sample(range(utterance_num), k=speaker_name_error_num)
        for random_index in random_index_list:
            counterfactual_segments[random_index]['speaker'] = 'unknown'
        return counterfactual_segments


    def change_utterances(self, segments, error_rate):
        utterance_num = len(segments)
        total_word_num = self.get_total_word_num(segments)
        word_error_num = int(total_word_num * error_rate)
        counterfactual_segments = copy.deepcopy(segments)
        random_index_list = random.sample(range(total_word_num), k=word_error_num)
        for random_index in random_index_list:
            current_word_index = 0
            current_utterance_id = 0
            while current_word_index < random_index:
                text = counterfactual_segments[current_utterance_id]['text']
                text = text.split(' ')
                word_num = len(text)
                current_word_index += word_num
                if current_word_index >= random_index:
                    text[-1 + random_index - current_word_index] = '...'
                counterfactual_segments[current_utterance_id]['text'] = ' '.join(text)
                current_utterance_id += 1
        return counterfactual_segments
