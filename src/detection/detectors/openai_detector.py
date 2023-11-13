from sklearn.metrics import classification_report as cr
from detectors.AbstractDetector import AbstractDetector
from utils import load_data
from detectors.utils import change_label, get_label
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
import torch
from tqdm import tqdm
import numpy as np


class OpenAIDetector(AbstractDetector):

    def __init__(self):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            "roberta-large-openai-detector")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "roberta-large-openai-detector")
        self.model.eval().cuda()

    def predict(self, txt):

        tokens = self.tokenizer.encode(txt, truncation=True, max_length=512)
        tokens = torch.Tensor(tokens)

        tokens = tokens.unsqueeze(0).cuda().long()
        mask = torch.ones_like(tokens).cuda().long()
        logits = self.model(tokens, attention_mask=mask)
        probs = logits[0].softmax(dim=-1)
        probs = probs.detach().cpu().flatten().numpy()

        return [probs[1], probs[0]]


def openai_det(data):

    print('Running...')
    det = OpenAIDetector()

    pred = []
    probabilities = []

    for i in tqdm(list(data['Generation'])):
        p = det.predict(str(i))
        result = np.argmax(p)
        result = get_label(result)

        pred.append(result)
        probabilities.append(p)

    data['pred'] = pred
    data['probabilities'] = probabilities
    y_true = change_label(list(data['label']))
    y_pred = change_label(pred)
    print(cr(y_true=y_true, y_pred=y_pred, digits=4))
    data.to_csv(
        '../../data/results/roberta-large-openai-detector.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenAI GPT-2 Detector')
    opt = parser.parse_args()

    text = 'OpenAI GPT-2 detector'

    DATA = load_data()
    DATA = DATA.sample(frac=1).reset_index(drop=True)
    openai_det(data=DATA)
