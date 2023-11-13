import logging
import pandas as pd
import time
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('LanguageModel')


class LanguageModel:
    def __init__(self, model_name, max_length=256, batch_size=100, delay=None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        name = model_name.replace('/', '-')
        self.csv_path = f'../../../data/{name}.csv'
        self.delay = delay
        self._load_csv()

    def _load_csv(self):
        try:
            self.df = pd.read_csv(self.csv_path).set_index('prompt')
        except:
            self.df = pd.DataFrame()

    def save_csv(self):
        self.df.to_csv(self.csv_path)

    def generate(self, texts, repeat=1):
        logger.info('Generating texts...')
        is_list = True
        if isinstance(texts, str):
            is_list = False
            texts = [texts]

        index = 0
        generated_texts = []
        while texts[index * self.batch_size:(index + 1) * self.batch_size]:
            generated_texts.extend(self._generate_batch(
                texts[index * self.batch_size:(index + 1) * self.batch_size], repeat))
            index += 1

        if is_list:
            return generated_texts

        return generated_texts[0]

    def _generate_batch(self, texts, repeat=1):
        print(f'Generating batch of {len(texts)} texts... repeat {repeat}')
        generated_texts = []

        for i in range(repeat):
            for text in tqdm(texts, total=len(texts)):
                generated_texts.append(self._generate_text(text))
                if self.delay:
                    time.sleep(self.delay)

        return generated_texts

    def _generate_text(self, text):
        raise NotImplementedError(
            'Function _generate_text is not implemented!')

    def _add_generated_text(self, prompt, generated_text):
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
