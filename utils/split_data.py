# -*- coding: utf-8 -*-
import os
import json
from sklearn.model_selection import train_test_split

def split(train_size=0.7):
    path = 'TrainingData'
    label = 0
    label_mapping = {}
    train_data = []
    valid_data = []

    for case in os.listdir(path):
        data = []
        for filename in os.listdir(f'{path}/{case}'):
            data.append((f'{path}/{case}/{filename}', label))
        train, valid = train_test_split(data, train_size=train_size)
        train_data += train
        valid_data += valid
        label_mapping[label] = case
        label += 1

    with open('config/train.txt', 'w', encoding='utf-8') as fp:
        for data in train_data:
            fp.write(f'{data[0]} {data[1]}\n')

    with open('config/valid.txt', 'w', encoding='utf-8') as fp:
        for data in valid_data:
            fp.write(f'{data[0]} {data[1]}\n')

    json.dump(label_mapping, open('label.json', 'w', encoding='utf-8'), indent=2)