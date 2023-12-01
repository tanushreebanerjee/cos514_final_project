import random

class ErrorInjection:
    def __init__(self, error_types):
        self.error_types = error_types

    def _inject_speaker_identification_error(self, transcript, error_degree):
        # Logic to inject speaker identification errors
        # For example, randomly change the speaker of a sentence
        tokens = transcript.split()
        num_tokens = len(tokens)
        num_errors = int(num_tokens * error_degree)

        for _ in range(num_errors):
            index_to_change = random.randint(0, num_tokens - 1)
            tokens[index_to_change] = 'SPEAKER_ERROR'

        return ' '.join(tokens)

    def _inject_speech_recognition_error(self, transcript, error_degree):
        # Logic to inject speech recognition errors
        # For example, randomly change a word in the transcript
        words = transcript.split()
        num_words = len(words)
        num_errors = int(num_words * error_degree)

        for _ in range(num_errors):
            index_to_change = random.randint(0, num_words - 1)
            words[index_to_change] = 'RECOGNITION_ERROR'

        return ' '.join(words)

    def inject_errors(self, transcript, degree):
        # Logic to inject errors based on error types and degree
        for error_type in self.error_types:
            if error_type == 'speaker_identification':
                transcript = self._inject_speaker_identification_error(transcript, degree)
            elif error_type == 'speech_recognition':
                transcript = self._inject_speech_recognition_error(transcript, degree)
            # Add more error types if needed

        return transcript
