from sklearn.metrics import classification_report as cr
from detectors.utils import change_label, get_label
from detectors.AbstractDetector import AbstractDetector
from utils import load_data
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
import numpy as np
from tqdm import tqdm


class SimpleAIDetector(AbstractDetector):

    def __init__(self):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            "Hello-SimpleAI/chatgpt-detector-roberta")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "Hello-SimpleAI/chatgpt-detector-roberta")
        self.model.eval().cuda()


def simpleai_det(data):

    print('Running...')
    det = SimpleAIDetector()

    pred = []
    probabilities = []

    for i in tqdm(list(data['Generation'])):
        p = det.predict(str(i))
        result = np.argmax(p)
        result = get_label(result)

        probabilities.append(p)
        pred.append(result)

    data['pred'] = pred
    data['probabilities'] = probabilities
    y_true = change_label(list(data['label']))
    y_pred = change_label(pred)
    print(cr(y_true=y_true, y_pred=y_pred, digits=4))
    data.to_csv('../../data/results/simpleai-detector.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SimpleAI Detector')
    opt = parser.parse_args()

    text = 'SimpleAI detector'

    DATA = load_data()
    DATA = DATA.sample(frac=1).reset_index(drop=True)
    simpleai_det(data=DATA)
