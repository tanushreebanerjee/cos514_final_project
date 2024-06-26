from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer

class Summarization:
    def __init__(self, model_name):
        self.model = XLMRobertaForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)

    def summarize(self, transcript):
        # Tokenize and summarize the transcript using the LLM
        inputs = self.tokenizer(transcript, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = self.model.generate(**inputs)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

# # Example usage:
# llm_model_name = "Xwin-LM-7B-v0.1"
# summarizer = Summarization(llm_model_name)

# # Example transcript
# example_transcript = "This is a sample transcript for testing summarization."

# # Summarize the transcript
# summary = summarizer.summarize(example_transcript)

# # Print the result
# print("Original Transcript:")
# print(example_transcript)
# print("\nSummarized Result:")
# print(summary)
