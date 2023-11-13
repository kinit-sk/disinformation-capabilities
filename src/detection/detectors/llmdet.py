from sklearn.metrics import classification_report as cr
from detectors.AbstractDetector import AbstractDetector
from detectors.utils import change_label, get_label
from utils import load_data
import argparse
import llmdet
from tqdm import tqdm
import numpy as np


class LLMDetDetector(AbstractDetector):

    def __init__(self):
        super().__init__()
        llmdet.load_probability()

    def predict(self, txt):
        results = llmdet.detect(txt)
        result = results[0]

        human_prob = result['Human_write']
        return [human_prob, 1 - human_prob]


def llmdet_det(data):

    print('Running...')
    det = LLMDetDetector()

    pred = []
    probabilities = []

    for i in tqdm(list(data['Generation'])):
        probs = det.predict(str(i))
        result = np.argmax(probs)
        result = get_label(result)

        probabilities.append(probs)
        pred.append(result)

    data['pred'] = pred
    data['probabilities'] = probabilities
    y_true = change_label(list(data['label']))
    y_pred = change_label(pred)
    print(cr(y_true=y_true, y_pred=y_pred, digits=4))
    data.to_csv('../../data/results/llmdet.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLMDet Detector')
    opt = parser.parse_args()

    text = 'LLMDet detector'

    DATA = load_data()
    DATA = DATA.sample(frac=1).reset_index(drop=True)
    llmdet_det(data=DATA)
