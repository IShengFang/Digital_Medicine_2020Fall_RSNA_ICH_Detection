import os
import pydicom
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from data import preprocess


class ICHDataset(Dataset):
    def __init__(self, path, img_size, kernel='bsb', transform=None):
        '''
        kernel mode: bsb(defualt),meta, brain, all_channel_window, rainbow_window
        '''
        self.transform = transform
        self.files = []
        self.img_size = img_size
        self.kernel = kernel
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

        if self.kernel == 'bsb':
            img = preprocess.bsb_window(dcm)
        elif self.kernel == 'meta':
            img = preprocess.meta_window(dcm)
            img = np.array([img, img, img])
        elif self.kernel == 'brain':
            img = preprocess.brain_window(dcm)
            img = np.array([img, img, img])
        elif self.kernel == 'all_channel_window':
            img = preprocess.all_channel_window(dcm)
        elif self.kernel == 'rainbow_window':
            img = preprocess.rainbow_window(dcm)

        img = torch.tensor(img, dtype=torch.float)
        img = TF.resize(img, (self.img_size, self.img_size))

        if self.transform is not None:
            img = self.transform(img)

        if label is None:
            return img
        else:
            return img, label
