from typing import Sequence
import random
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms.functional as TF
from collections import OrderedDict
import torchvision.transforms as T 
import numpy as np

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
    T.RandomApply([MyRotateTransform(angles=angle)], p=0.5),
    T.RandomApply([T.ColorJitter()], p=0.5)
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
