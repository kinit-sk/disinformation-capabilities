from sklearn.metrics import classification_report as cr
from detectors.AbstractDetector import AbstractDetector
from detectors.utils import change_label, get_label
from utils import load_data
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
import numpy as np
from tqdm import tqdm


class Detector(AbstractDetector):

    def __init__(self, model_name, model_path):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            f'{model_path}/{model_name}')
        self.model = AutoModelForSequenceClassification.from_pretrained(
            f'{model_path}/{model_name}')
        self.model.eval().cuda()


def detector_det(data, name, model_path, output_name=None):
    print(f'Running: {name}')
    det = Detector(name, model_path)

    pred = []
    probabilities = []

    for i in tqdm(list(data['Generation'])):
        p = det.predict(str(i))
        result = np.argmax(p)
        probabilities.append(p)
        result = get_label(result)

        pred.append(result)

    data['pred'] = pred
    data['probabilities'] = probabilities
    y_true = change_label(list(data['label']))
    y_pred = change_label(pred)
    print(cr(y_true=y_true, y_pred=y_pred, digits=4))
    data.to_csv(output_name, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detector')
    parser.add_argument('--name', type=str,
                        help='name of the model', required=True)
    parser.add_argument('--model_path', type=str,
                        default='../../data/models', help='path to the model')
    opt = parser.parse_args()

    text = f'{opt.name} Detector'

    DATA = load_data()
    DATA = DATA.sample(frac=1).reset_index(drop=True)
    detector_det(data=DATA, name=opt.model, model_path=opt.model_path,
                 output_name=f'../../data/{opt.name}.csv')
