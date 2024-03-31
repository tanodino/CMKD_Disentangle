from typing import Sequence
import random
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms.functional as TF
from collections import OrderedDict
import torchvision.transforms as T 
import numpy as np
import math






TRAIN_BATCH_SIZE = 128#512#128#16#512#1024#512
VALID_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
LEARNING_RATE = 0.0001
MOMENTUM_EMA = .95
EPOCHS = 300
TH_FIXMATCH = .95
WARM_UP_EPOCH_EMA = 50

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
    T.RandomApply([MyRotateTransform(angles=angle)], p=0.5)#,
    #T.RandomApply([T.ColorJitter()], p=0.5)
    ])



def cumulate_EMA(model, ema_weights, alpha):
    current_weights = OrderedDict()
    current_weights_npy = OrderedDict()
    state_dict = model.state_dict()
    for k in state_dict:
        current_weights_npy[k] = state_dict[k].cpu().detach().numpy()

    if ema_weights is not None:
        for k in state_dict:
            current_weights_npy[k] = alpha * ema_weights[k].cpu().detach().numpy() + (1-alpha) * current_weights_npy[k]

    for k in state_dict:
        current_weights[k] = torch.tensor( current_weights_npy[k] )

    return current_weights

def modify_weights(model, ema_weights, alpha):
    current_weights = OrderedDict()
    current_weights_npy = OrderedDict()
    state_dict = model.state_dict()
    
    for k in state_dict:
        current_weights_npy[k] = state_dict[k].cpu().detach().numpy()

    if ema_weights is not None:
        for k in state_dict:
            current_weights_npy[k] = alpha * ema_weights[k] + (1-alpha) * current_weights_npy[k]
    
    for k in state_dict:
        current_weights[k] = torch.tensor( current_weights_npy[k] )
    
    return current_weights, current_weights_npy




class MyDataset_Unl(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]        
        x_transform = self.transform(self.data[index])
        
        return x, x_transform
    
    def __len__(self):
        return len(self.data)

#TO CHECK IF IT WORKS ...
class MyDatasetMM(Dataset):
    def __init__(self, data1, data2, targets, transform=None):
        self.data1 = data1
        self.data2 = data2
        self.targets = torch.LongTensor(targets)
        self.transform = transform
        
    def __getitem__(self, index):
        x1 = self.data1[index]
        x2 = self.data2[index]
        dim1 = x1.shape[0]
        y = self.targets[index]
        x = torch.cat([x1,x2],dim=0)

        if self.transform:
            if np.random.uniform() > .5:
                x = self.transform(x)
        
        x1 = x[0:dim1,:,:]
        x2 = x[dim1::,:,:]
        return x1, x2, y

    def __len__(self):
        return len(self.data1)




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


class CosineDecay(object):
    def __init__(self,
                max_value,
                min_value,
                num_loops):
        self._max_value = max_value
        self._min_value = min_value
        self._num_loops = num_loops

    def get_value(self, i):
        if i < 0:
            i = 0
        if i >= self._num_loops:
            i = self._num_loops
        value = (math.cos(i * math.pi / self._num_loops) + 1.0) * 0.5
        value = value * (self._max_value - self._min_value) + self._min_value
        return value
