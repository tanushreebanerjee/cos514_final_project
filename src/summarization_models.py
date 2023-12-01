import os
import openai
import backoff
from transformers import pipeline

class ChatGPTSummarizer:
    def __init__(self, model_name="gpt-3.5-turbo", api_key=None):
        openai.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model_name = model_name

    @backoff.on_exception(backoff.expo, openai.OpenAIAPIError, max_time=60)
    def summarize(self, text, api_key=None):
        openai.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        prompt = f"Summarize the following text:\n{text}\n\nSummary:"
        response = openai.Completion.create(
            engine=self.model_name,
            prompt=prompt,
            max_tokens=150,
            temperature=0.7,
            stop=None
        )
        return response['choices'][0]['text'].strip()

# Placeholder for XwinSummarizer
class XwinSummarizer:
    def __init__(self, model_name="TheBloke/Xwin-LM-7B-V0.1-GGUF"):
        self.summarizer = pipeline("summarization", model=model_name)

    def summarize(self, text):
        # Use Xwin summarizer to generate a summary
        summary = self.summarizer(text, max_length=150, min_length=50, length_penalty=2.0, num_beams=4)
        return summary[0]['summary_text']

# Placeholder for VicunaSummarizer
class VicunaSummarizer:
    def __init__(self, model_name="lmsys/vicuna-13b-v1.3"):
        self.summarizer = pipeline("summarization", model=model_name)

    def summarize(self, text):
        # Use Vicuna summarizer to generate a summary
        summary = self.summarizer(text, max_length=150, min_length=50, length_penalty=2.0, num_beams=4)
        return summary[0]['summary_text']

# Placeholder for MPTSummarizer
class MPTSummarizer:
    def __init__(self, model_name="mosaicml/mpt-7b"):
        self.summarizer = pipeline("summarization", model=model_name)

    def summarize(self, text):
        # Use MPT summarizer to generate a summary
        summary = self.summarizer(text, max_length=150, min_length=50, length_penalty=2.0, num_beams=4)
        return summary[0]['summary_text']

# Placeholder for Llama2Summarizer
class Llama2Summarizer:
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf"):
        self.summarizer = pipeline("summarization", model=model_name)

    def summarize(self, text):
        # Use Llama2 summarizer to generate a summary
        summary = self.summarizer(text, max_length=150, min_length=50, length_penalty=2.0, num_beams=4)
        return summary[0]['summary_text']

# Example usage
example_text = "The project manager introduced the upcoming project to the team members and then the team members participated in an exercise..."

# ChatGPT Summary
chatgpt_summarizer = ChatGPTSummarizer(model_name="gpt-4")
chatgpt_summary = chatgpt_summarizer.summarize(example_text)
print("ChatGPT Summary:", chatgpt_summary)

# Xwin Summary
xwin_summarizer = XwinSummarizer()
xwin_summary = xwin_summarizer.summarize(example_text)
print("Xwin Summary:", xwin_summary)

# Vicuna Summary
vicuna_summarizer = VicunaSummarizer()
vicuna_summary = vicuna_summarizer.summarize(example_text)
print("Vicuna Summary:", vicuna_summary)

# MPT Summary
mpt_summarizer = MPTSummarizer(model_name="mosaicml/mpt-7b")
mpt_summary = mpt_summarizer.summarize(example_text)
print("MPT Summary:", mpt_summary)

# Llama2 Summary
llama2_summarizer = Llama2Summarizer(model_name="meta-llama/Llama-2-7b-hf")
llama2_summary = llama2_summarizer.summarize(example_text)
print("Llama2 Summary:", llama2_summary)
