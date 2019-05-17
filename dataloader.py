#!/usr/bin/env python
# coding: utf-8


import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import glob
import os
import numpy as np


from skimage import io


class LFWDataset(Dataset):
    def __init__(self, path="data", transform=None):
        self.path = os.path.join(".", path, "*", "*")
        self.data = glob.glob(self.path)
        self.classes = list({s.split(os.path.sep)[-2] for s in self.data})
        self.labels = [self.classes.index(
            l.split(os.path.sep)[-2]) for l in self.data]
        self.transform = transform

    def __getitem__(self, index):
        image = io.imread(self.data[index])
        label = np.array(self.labels[index])

        if self.transform:
            image = self.transform(image)

        return image, torch.from_numpy(label)

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":

    t = torchvision.transforms.ToTensor()
    mydataset = LFWDataset(transform=t)
    loader = DataLoader(dataset=mydataset, batch_size=10,
                        num_workers=0, shuffle=False)

    for data in loader:
        print(data)
        break
