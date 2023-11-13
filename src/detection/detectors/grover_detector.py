from sklearn.metrics import classification_report as cr
import numpy as np
from detectors.AbstractDetector import AbstractDetector
from detectors.utils import change_label, get_label
from utils import load_data
import argparse
import requests
import json
from tqdm import tqdm
import time


class GroverDetector(AbstractDetector):

    def __init__(self):
        super().__init__()
        self.url = 'https://discriminate.grover.allenai.org/api/disc'

    def predict(self, txt):
        payload = json.dumps({
            "article": txt,
            "domain": "",
            "date": "",
            "authors": "",
            "title": "",
            "target": "discrimination"
        })

        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.request(
            "POST", self.url, headers=headers, data=payload)
        if response.status_code != 200:
            # try one more time
            time.sleep(60)
            response = requests.request(
                "POST", self.url, headers=headers, data=payload)
            if response.status_code != 200:
                print(response.json())
                raise Exception('Error: ', response.status_code)

        prob = response.json()['groverprob']
        probs = [1 - prob, prob]
        return probs


def grover_det(data):

    print('Running...')
    det = GroverDetector()

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
    data.to_csv('../../data/results/grover.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Grover Detector')
    opt = parser.parse_args()

    text = 'Grover Detector'

    DATA = load_data()
    DATA = DATA.sample(frac=1).reset_index(drop=True)
    grover_det(data=DATA)
