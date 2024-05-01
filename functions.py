from typing import Sequence
import random
from torch.utils.data import Dataset
import torch
import torchvision.transforms.functional as TF
from collections import OrderedDict
import torchvision.transforms as T 
import numpy as np
import math


TRAIN_BATCH_SIZE = 128
LEARNING_RATE = 0.0001
EPOCHS = 300

hashPREFIX2SOURCE = {'DEPTH':'image','RGB':'image','MS':'image','SAR':'image','SPECTRO':'image','MNIST':'mnist','FULL':'hyper','HALF':'hyper', "THERMAL":"thermal"}

class MyRotateTransform():
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

angle = [0, 90, 180, 270]
transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomApply([MyRotateTransform(angles=angle)], p=0.5)
    ])

class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transform:
            if np.random.uniform() > .5:
                x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)
