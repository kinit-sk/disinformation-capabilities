from sklearn.metrics import classification_report as cr
from detectors.utils import change_label, get_label
import numpy as np
from utils import load_data
import argparse
import requests
import json
from tqdm import tqdm
import time


class ZeroGPT(object):

    COOKIE = 'COOKIE'  # insert your cookie here
    AUTHORIZATION = 'AUTHORIZATION'  # insert your authorization here

    def __init__(self):

        print('Initializing Detector...')
        self.url = 'https://api.zerogpt.com/api/detect/detectText'

    def predict(self, txt):
        payload = json.dumps({
            "input_text": txt,
        })

        headers = {
            'Authorization': self.AUTHORIZATION,
            'Content-Type': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
            'Cookie': self.COOKIE,
            'Origin': 'https://www.zerogpt.com'
        }

        response = requests.request(
            "POST", self.url, headers=headers, data=payload)
        print(response.status_code)
        if response.status_code != 200:
            # try one more time
            time.sleep(3600)
            response = requests.request(
                "POST", self.url, headers=headers, data=payload)
            if response.status_code != 200:
                print(response.json())
                raise Exception('Error: ', response.status_code)

        prob = response.json()['data']['fakePercentage'] / 100
        probs = [1 - prob, prob]
        return probs


def zerogpt_det(data):
    print('Running...')
    det = ZeroGPT()

    pred = []
    probabilities = []

    count = len(list(data['Generation']))

    for i in tqdm(list(data['Generation']), total=count):
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
    data.to_csv('../../data/results/zerogpt.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Grover Detector')
    opt = parser.parse_args()

    text = 'ZeroGPT Detector'

    DATA = load_data()
    DATA = DATA.sample(frac=1).reset_index(drop=True)
    zerogpt_det(data=DATA)
