### To implemente the Transformer Framework I used the code from this website : https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch

import torch
import torch.nn as nn
import sys
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sys
from sklearn.utils import shuffle
#from model_transformer import TransformerEncoder
from model_pytorch import MonoSourceModel, ModelHYPER
import time
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from functions import TRAIN_BATCH_SIZE, LEARNING_RATE, EPOCHS, transform, MyDataset, hashPREFIX2SOURCE, VALID_BATCH_SIZE, TEST_BATCH_SIZE
import os


def createDataLoader(x, y, tobeshuffled, BATCH_SIZE, type_data='none'):
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.int64)
    dataset = None
    if type_data == 'RGB' or type_data=='MS' or type_data=='MNIST' or type_data=='SAR' or type_data=='DEPTH' or type_data=='THERMAL':
        dataset = MyDataset(x_tensor, y_tensor, transform=transform)
    else:
        dataset = TensorDataset(x_tensor, y_tensor)
        
    dataloader = DataLoader(dataset, shuffle=tobeshuffled, batch_size=BATCH_SIZE)
    return dataloader

def evaluation(model, dataloader, device):
    model.eval()
    tot_pred = []
    tot_labels = []
    for x_batch_f, y_batch in dataloader:
        x_batch_f = x_batch_f.to(device)
        y_batch = y_batch.to(device)
        pred = model(x_batch_f)
        pred_npy = np.argmax(pred.cpu().detach().numpy(), axis=1)
        tot_pred.append( pred_npy )
        tot_labels.append( y_batch.cpu().detach().numpy())
    tot_pred = np.concatenate(tot_pred)
    tot_labels = np.concatenate(tot_labels)
    return tot_pred, tot_labels

def dataAugRotate(data, labels, axis):
    new_data = []
    new_label = []
    for idx, el in enumerate(data):
        for k in range(4):
            new_data.append( np.rot90(el, k, axis) )
            new_label.append( labels[idx] )
    return np.array(new_data), np.array(new_label)


dir_ = sys.argv[1]
prefix = sys.argv[2]
run_id = int(sys.argv[3])


dir_name = dir_+"/MONO_%s"%prefix
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

output_file = dir_name+"/%s.pth"%(run_id)

first_data = np.load("%s/%s_data_normalized.npy"%(dir_,prefix))
labels = np.load("%s/labels.npy"%dir_)

print("FIRST DATA ",first_data.shape)

train_idx = np.load("%s/train_idx_%d.npy"%(dir_,run_id))
valid_idx = np.load("%s/valid_idx_%d.npy"%(dir_,run_id)) 
test_idx = np.load("%s/test_idx_%d.npy"%(dir_,run_id))

train_f_data = first_data[train_idx]
valid_f_data = first_data[valid_idx]
test_f_data = first_data[test_idx]

train_labels = labels[train_idx]
valid_labels = labels[valid_idx]
test_labels = labels[test_idx]


n_classes = len(np.unique(labels))

train_f_data, train_labels = shuffle(train_f_data, train_labels)

#DATALOADER TRAIN
dataloader_train = createDataLoader(train_f_data, train_labels, True, TRAIN_BATCH_SIZE, type_data=first_data)

#DATALOADER VALID
dataloader_valid = createDataLoader(valid_f_data, valid_labels, False, VALID_BATCH_SIZE)

#DATALOADER TEST
dataloader_test = createDataLoader(test_f_data, test_labels, False, TEST_BATCH_SIZE)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = None
#if dir_ == "PAVIA_UNIVERSITY":
#    model = ModelHYPER(num_classes=n_classes)
#else:
model = MonoSourceModel(input_channel_first=first_data.shape[1], encoder=hashPREFIX2SOURCE[prefix], num_classes=n_classes)

model = model.to(device)


learning_rate = 0.0001
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

# Loop through the data
global_valid = 0
ema_weights = None
valid_f1 = 0.0
for epoch in range(EPOCHS):
    start = time.time()
    model.train()
    tot_loss = 0.0
    den = 0
    for x_batch_f, y_batch in dataloader_train:
        optimizer.zero_grad()
        x_batch_f = x_batch_f.to(device)
        y_batch = y_batch.to(device)

        pred = model(x_batch_f)        
        loss_pred = loss_fn(pred, y_batch)

        loss = loss_pred
        loss.backward() # backward pass: backpropagate the prediction loss
        optimizer.step() # gradient descent: adjust the parameters by the gradients collected in the backward pass
        tot_loss+= loss.cpu().detach().numpy()
        den+=1.

    end = time.time()
    pred_valid, labels_valid = evaluation(model, dataloader_valid, device)
    f1_val = f1_score(labels_valid, pred_valid, average="weighted")

    pred_test, labels_test = evaluation(model, dataloader_test, device)
    f1_test = f1_score(labels_test, pred_test, average="weighted")
    
    if f1_val > global_valid:
        global_valid =f1_val
        print("TRAIN LOSS at Epoch %d: %.4f with BEST ACC on TEST SET %.2f with training time %d"%(epoch, tot_loss/den, 100*f1_test,  (end-start)))
        torch.save(model.state_dict(), output_file)
    else:
        print("TRAIN LOSS at Epoch %d: %.4f with F1 %.2f training time %d"%(epoch, tot_loss/den, 100*f1_test, (end-start)))        
    sys.stdout.flush()