import pandas as pd


def load_data(path='../../data/mgt_detection.csv'):
    data = pd.read_csv(path)

    return data


def change_label(data):
    for i in range(len(data)):
        if data[i] == 'human':
            data[i] = 0
        else:
            data[i] = 1
    return data


def get_label(result):
    if result == 0:
        result = 'human'
    else:
        result = 'machine'

    return result
