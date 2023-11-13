# https://github.com/yafuly/DeepfakeTextDetect
from sklearn.metrics import classification_report as cr
from detectors.AbstractDetector import AbstractDetector
from detectors.utils import change_label
from utils import load_data
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
from detectors.longformer_utils import preprocess, detect
from tqdm import tqdm


class LongoformerDetector(AbstractDetector):

    def __init__(self):
        super().__init__()

        model_dir = "nealcly/detection-longformer"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir)
        self.model.eval().cuda()

    def predict(self, txt):

        txt = preprocess(txt)

        result = detect(txt, self.tokenizer, self.model)

        return result


def longformer_det(data):

    print('Running...')
    det = LongoformerDetector()

    pred = []

    for i in tqdm(list(data['Generation'])):
        result = det.predict(str(i))

        if result == 'human-written':
            result = 'human'
        else:
            result = 'machine'

        pred.append(result)

    data['pred'] = pred
    y_true = change_label(list(data['label']))
    y_pred = change_label(pred)
    print(cr(y_true=y_true, y_pred=y_pred, digits=4))
    data.to_csv('../../data/results/longformer.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LongFormer Detector')
    opt = parser.parse_args()

    text = 'LongFormer detector'

    DATA = load_data()
    DATA = DATA.sample(frac=1).reset_index(drop=True)
    longformer_det(data=DATA)
