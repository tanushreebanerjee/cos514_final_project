import backoff
import openai

from transformers import AutoModelForCausalLM, AutoTokenizer


# class ChatGPTSummarizer:
#     def __init__(self, model_name="gpt-3.5-turbo", api_key=None):
#         openai.api_key = api_key or os.environ.get("OPENAI_API_KEY")
#         self.model_name = model_name

#     @backoff.on_exception(backoff.expo, openai.OpenAIAPIError, max_time=60)
#     def summarize(self, text, api_key=None):
#         openai.api_key = api_key or os.environ.get("OPENAI_API_KEY")
#         prompt = f"Summarize the following text:\n{text}\n\nSummary:"
#         response = openai.Completion.create(
#             engine=self.model_name,
#             prompt=prompt,
#             max_tokens=150,
#             temperature=0.7,
#             stop=None
#         )
#         return response['choices'][0]['text'].strip()


class SummarizationModel:
    
    def __init__(self, model_name):
        """Summarization model.

        Args:
            model_name (Path or str): Model directory or model name. Choose from the following list.
                ['Xwin-LM/Xwin-LM-7B-V0.1', 'lmsys/vicuna-13b-v1.3', 'mosaicml/mpt-7b', 'meta-llama/Llama-2-7b-hf']
        """
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __call__(self, dialogue, is_abstractive=True, error_aware=True):
        """Summarization

        Args:
            dialogue (str): Dialogue.
            is_abstractive (bool): Whether the model generates abstractive summaries or not.
                default = True
            error_aware (bool): Whether the model is aware of upstream task errors or not.
                default = True

        Returns:
            (str): Summary.
        """
        summary_type = 'abstractive'
        if is_abstractive is not True:
            summary_type = 'extractive'
        prompt = f'Please generate an ${summary_type} summary for the following dialogue'
        if error_aware:
            prompt += '. Note that this dialogue might contain errors in speaker names and transcripts: '
        else:
            prompt += ': '
        # prompt += dialogue

        inputs = self.tokenizer(prompt, return_tensors='pt')
        samples = self.model.generate(**inputs, max_new_tokens=4096, temperature=0.7)
        output = self.tokenizer.decode(samples[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return output
