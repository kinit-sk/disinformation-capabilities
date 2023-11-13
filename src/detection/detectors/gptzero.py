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
import pandas as pd


class GPTZero(AbstractDetector):

    COOKIE = 'COOKIE'  # insert your cookie her

    def __init__(self):
        super().__init__()
        self.url = 'https://api.gptzero.me/v2/predict/text'

    def predict(self, txt):
        payload = json.dumps({
            "document": txt,
        })

        headers = {
            'Content-Type': 'application/json',
            'Cookie': self.COOKIE
        }

        response = requests.request(
            "POST", self.url, headers=headers, data=payload)
        print(response.status_code)
        if response.status_code != 200:
            time.sleep(3600)
            response = requests.request(
                "POST", self.url, headers=headers, data=payload)
            if response.status_code != 200:
                print(response.json())
                raise Exception('Error: ', response.status_code)

        prob = response.json()['documents'][0]['completely_generated_prob']
        probs = [1 - prob, prob]
        return probs


def gptzero_det(data):

    print('Running...')
    det = GPTZero()

    pred = []

    for idx, i in tqdm(enumerate(list(data['Generation']))):
        # check if already predicted
        if not pd.isnull(data.at[idx, 'pred']):
            pred.append(data.at[idx, 'pred'])
            continue
        p = det.predict(str(i))

        result = np.argmax(p)
        result = get_label(result)

        data.at[idx, 'pred'] = result
        data.at[idx, 'probabilities'] = [p]
        data.to_csv('../../data/results/gptzero.csv', index=False)

        pred.append(result)

    y_true = change_label(list(data['label']))
    y_pred = change_label(pred)
    print(cr(y_true=y_true, y_pred=y_pred, digits=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPTZero Detector')
    opt = parser.parse_args()

    text = 'GPTZero Detector'

    DATA = load_data()
    DATA = DATA.sample(frac=1).reset_index(drop=True)
    gptzero_det(data=DATA)
