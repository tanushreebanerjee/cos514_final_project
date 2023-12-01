import openai
import backoff

class ErrorCorrection:
    def __init__(self, api_key, model="text-davinci-003"):
        openai.api_key = api_key
        self.model = model

    @backoff.on_exception(backoff.expo, openai.OpenAIError, max_tries=3, factor=2)
    def correct_errors(self, text):
        prompt = f"Correct the following text: '{text}'"

        response = openai.Completion.create(
            engine=self.model,
            prompt=prompt,
            temperature=0.7,
            max_tokens=150
        )

        corrected_text = response['choices'][0]['text'].strip()
        return corrected_text

# Example usage:
api_key = "YOUR_OPENAI_API_KEY"
error_correction = ErrorCorrection(api_key)

# Example text with errors
text_with_errors = "This is an example sentece with speling mistakes."

# Correct errors using ChatGPT with exponential backoff
corrected_text = error_correction.correct_errors(text_with_errors)

# Print the result
print("Original Text:")
print(text_with_errors)
print("\nCorrected Text:")
print(corrected_text)
