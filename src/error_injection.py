# Implement functions to inject errors into transcripts
def inject_speaker_errors(transcript, error_degree):
    # Logic to inject speaker identification errors
    # ...

def inject_asr_errors(transcript, error_degree):
    # Logic to inject speech recognition errors
    # ...

# Apply error injection to transcripts
transcripts_with_errors = [inject_speaker_errors(t, degree) for t in transcripts]
transcripts_with_errors = [inject_asr_errors(t, degree) for t in transcripts_with_errors]
