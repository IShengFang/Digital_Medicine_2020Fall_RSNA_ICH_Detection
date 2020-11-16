import os
import json
import utils
from sklearn.model_selection import train_test_split


def split_data(path='dataset/TrainingData', split_file_dir='split_file', train_size=0.7):
    label = 0
    label_mapping = {}
    train_data = []
    valid_data = []

    cases = sorted(os.listdir(path))
    for case in cases:
        data = []
        for filename in os.listdir(f'{path}/{case}'):
            data.append((f'{path}/{case}/{filename}', label))
        train, valid = train_test_split(data, train_size=train_size)
        train_data += train
        valid_data += valid
        label_mapping[label] = case
        label += 1

    utils.check_and_make_dir(split_file_dir)
    with open('{}/train.txt'.format(split_file_dir), 'w', encoding='utf-8') as fp:
        for data in train_data:
            fp.write(f'{data[0]} {data[1]}\n')

    with open('{}/valid.txt'.format(split_file_dir), 'w', encoding='utf-8') as fp:
        for data in valid_data:
            fp.write(f'{data[0]} {data[1]}\n')

    json.dump(label_mapping, open('{}/label.json'.format(split_file_dir), 'w', encoding='utf-8'), indent=2)
