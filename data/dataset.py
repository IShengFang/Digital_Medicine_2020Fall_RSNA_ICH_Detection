from data import preprocess

import pydicom

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF



class ICHDataset(Dataset):
    def __init__(self, path, kernel='bsb', transform=None):
        '''
        kernel mode: meta, brain, bsb
        '''
        self.transform = transform
        self.files = []
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

        if self.kernel == 'meta':
            img = preprocess.meta_window(dcm)
        elif self.kernel == 'brain':
            img = preprocess.brain_window(dcm)
        elif self.kernel == 'bsb':
            img = preprocess.bsb_window(dcm)
        img = torch.tensor(img)
        img = TF.resize(img, (512, 512))

        if self.transform is None:
            img = transform(img)
        else:
            img = self.transform(img)

        if label is None:
            return img
        else:
            return img, label