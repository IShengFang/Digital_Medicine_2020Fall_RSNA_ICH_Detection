# -*- coding: utf-8 -*-
import os
import cv2
import json
import pydicom
import numpy as np

from utils import split_data
from utils.preprocess import meta_window, brain_window, bsb_window

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

'''
# of different patient ID
0 -> 硬膜外出血 epidural: 223
1 -> 健康 healthy: 977
2 -> 腦實質性出血 intraparenchymal: 807
3 -> 腦室內出血 intraventricular: 712
4 -> 蛛網膜下腔出血 subarachnoid: 717
5 -> 硬膜下出血 subdural: 794
'''


class IHDataset(Dataset):
    def __init__(self, path, train=True):
        self.train = train
        self.files = []
        self.kernel = 'bsb'
        if path.endswith('.txt'):
            with open(path, 'r') as fp:
                self.files = fp.readlines()
        else:
            for filename in os.listdir(path):
                self.files.append(f'{path}/{filename}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        seg = self.files[idx].split(' ')
        filename = seg[0]
        label = int(seg[1]) if len(seg)>1 else None
        dcm = pydicom.dcmread(filename)

        if self.kernel == 'meta':
            img = meta_window(dcm)
        elif self.kernel == 'brain':
            img = brain_window(dcm)
        elif self.kernel == 'bsb':
            img = bsb_window(dcm)

        img = cv2.resize(img, (512, 512)).astype(np.float32)
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img = transform(img)

        if label is None:
            return img
        else:
            return img, label


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    for batch_idx, (imgs, labels) in enumerate(dataloader):
        imgs, labels = imgs.to(device), labels.to(device)
        output = model(imgs)
        _, pred = output.data.max(1)
        correct += (labels == pred).sum().item()
    print(f'acc on val: {correct/len(dataloader.dataset):.5f}')


def train(model, train_loader, valid_loader, optimizer, criterion, epochs, device):
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx%20 == 0:
                print(f'epoch: {epoch+1:>2}/{epochs}, batch: {batch_idx+1:>3}/{len(train_loader)}, loss: {loss.item():.5f}')

        running_loss /= len(train_loader)
        print('----------------------------')
        print(f'epoch: {epoch+1:>2}/{epochs}, avg. loss: {running_loss:.5f}')
        evaluate(model, valid_loader, device)
        print('============================')


def test(model, dataloader, device):
    res = []
    for batch_idx, imgs in enumerate(dataloader):
        imgs = imgs.to(device)
        output = model(imgs)
        _, pred = output.data.max(1)
        res += pred.tolist()
    return res


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr = 0.001
    epochs = 20
    batch_size = 16
    num_classes = 6
    criterion = nn.CrossEntropyLoss()

    split_data.split(0.7)
    class_dict = json.load(open('label.json', 'r'))
    train_set = IHDataset('config/train.txt')
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=8)
    valid_set = IHDataset('config/valid.txt')
    valid_loader = DataLoader(valid_set, batch_size=8, shuffle=True, num_workers=8)

    print('with resnet18 pretrained weights:')
    model = models.resnet18(pretrained=True)
    set_parameter_requires_grad(model, True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
    train(model, train_loader, valid_loader, optimizer, criterion, epochs, device)

    print('\nwithout resnet18 pretrained weights:')
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
    train(model, train_loader, valid_loader, optimizer, criterion, epochs, device)
