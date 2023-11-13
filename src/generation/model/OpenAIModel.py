import openai
import pandas as pd
from model.LanguageModel import LanguageModel


class OpenAIModel(LanguageModel):
    def __init__(self, model_name, batch_size=100, max_length=1000, temperature=1.0, repeat_count=1):
        super().__init__(model_name=model_name, max_length=max_length, batch_size=batch_size,
                         delay=5)
        self.temerature = temperature
        self.repeat_count = repeat_count

    def _generate_text(self, prompt):
        output_texts = self._generate_by_model(prompt)

        output_texts = [output_text.replace(
            prompt, '') for output_text in output_texts]

        self._add_generated_text(prompt, output_texts)

        return output_texts

    def _add_generated_text(self, prompt, generated_texts):
        for generated_text in generated_texts:
            self.df = pd.concat([
                self.df,
                pd.DataFrame([
                    {
                        'prompt': prompt,
                        'generated_text': generated_text
                    }
                ]).set_index('prompt')
            ])
            self.save_csv()

    def _generate_by_model(self, prompt):
        if self.model_name == 'gpt-3.5-turbo' or self.model_name == 'gpt-4':
            '''
            Limitation for ChatGPT and GPT-4 API:

            There is a limit of request per minute so it is necessary to add delay for generation using gpt-3.5-turbo and gpt-4.
            '''
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temerature,
                frequency_penalty=0.5,
                presence_penalty=1.10,
                max_tokens=self.max_length,
                top_p=0.95,
                n=self.repeat_count
            )

            length = len(response['choices'])

            return [
                response['choices'][i]['message']['content'] for i in range(length)
            ]
        else:
            response = openai.Completion.create(
                model=self.model_name,
                prompt=prompt,
                temperature=self.temerature,
                max_tokens=self.max_length,
                top_p=0.95,
                frequency_penalty=0.5,
                presence_penalty=1.10,
                stop=["Post 10:"],
                n=self.repeat_count
            )

            length = len(response['choices'])
            return [
                response['choices'][i]['text'] for i in range(length)
            ]
