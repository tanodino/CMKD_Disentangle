### To implemente the Transformer Framework I used the code from this website : https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch

import torch
import torch.nn as nn
import sys
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sys
from sklearn.utils import shuffle
#from model_transformer import TransformerEncoder
from model_pytorch import MultiSourceModel
import time
from sklearn.metrics import f1_score
from torchvision.models import resnet18
from sklearn.model_selection import train_test_split
from functions import TRAIN_BATCH_SIZE, LEARNING_RATE, EPOCHS, WARM_UP_EPOCH_EMA, cumulate_EMA, MOMENTUM_EMA, transform, MyDataset, hashPREFIX2SOURCE
import os


def createDataLoader(x_ms, x_sar, y, tobeshuffled, BATCH_SIZE):
    #DATALOADER TRAIN
    x_ms_tensor = torch.tensor(x_ms, dtype=torch.float32)
    x_sar_tensor = torch.tensor(x_sar, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.int64)

    dataset = TensorDataset(x_ms_tensor, x_sar_tensor, y_tensor)
    dataloader = DataLoader(dataset, shuffle=tobeshuffled, batch_size=BATCH_SIZE)
    return dataloader

def evaluation(model, dataloader, device):
    model.eval()
    tot_pred = []
    tot_labels = []
    for x_batch_f, x_batch_s, y_batch in dataloader:
        x_batch_f = x_batch_f.to(device)
        x_batch_s = x_batch_s.to(device)
        y_batch = y_batch.to(device)
        pred = model([x_batch_f, x_batch_s])
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
first_prefix = sys.argv[2]
second_prefix = sys.argv[3]
run_id = int(sys.argv[4])
fusion_type = sys.argv[5]

first_enc = hashPREFIX2SOURCE[first_prefix]
second_enc = hashPREFIX2SOURCE[second_prefix]


dir_name = dir_+"/TEACHER_%s"%fusion_type
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

output_file = dir_name+"/%s.pth"%(run_id)

first_data = np.load("%s/%s_data_normalized.npy"%(dir_,first_prefix))
second_data = np.load("%s/%s_data_normalized.npy"%(dir_,second_prefix))
labels = np.load("%s/labels.npy"%dir_)

print("FIRST DATA ",first_data.shape)
print("SECOND DATA ",second_data.shape)

train_idx = np.load("%s/train_idx_%d.npy"%(dir_,run_id))
valid_idx = np.load("%s/valid_idx_%d.npy"%(dir_,run_id)) 
test_idx = np.load("%s/test_idx_%d.npy"%(dir_,run_id))

train_f_data = first_data[train_idx]
valid_f_data = first_data[valid_idx]
test_f_data = first_data[test_idx]

train_s_data = second_data[train_idx]
valid_s_data = second_data[valid_idx]
test_s_data = second_data[test_idx]

train_labels = labels[train_idx]
valid_labels = labels[valid_idx]
test_labels = labels[test_idx]


n_classes = len(np.unique(labels))

train_f_data, train_s_data, train_labels = shuffle(train_f_data, train_s_data, train_labels)

#DATALOADER TRAIN
dataloader_train = createDataLoader(train_f_data, train_s_data, train_labels, True, TRAIN_BATCH_SIZE)

#DATALOADER VALID
dataloader_valid = createDataLoader(valid_f_data, valid_s_data, valid_labels, False, TRAIN_BATCH_SIZE)

#DATALOADER TEST
dataloader_test = createDataLoader(test_f_data, test_s_data, test_labels, False, TRAIN_BATCH_SIZE)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MultiSourceModel(input_channel_first=first_data.shape[1], input_channel_second=second_data.shape[1], f_encoder=first_enc, s_encoder=second_enc, fusion_type=fusion_type, num_classes=n_classes)
#model = MultiSourceModel(input_channel_first=first_data.shape[1], input_channel_second=second_data.shape[1], fusion_type=fusion_type, num_classes=n_classes)
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
    for x_batch_f, x_batch_s, y_batch in dataloader_train:
        optimizer.zero_grad()
        x_batch_f = x_batch_f.to(device)
        x_batch_s = x_batch_s.to(device)
        y_batch = y_batch.to(device)

        pred = model([x_batch_f, x_batch_s])        
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

    f1_test_ema = 0.0
    f1_val_ema = 0.0
    if epoch >= WARM_UP_EPOCH_EMA:
        ema_weights = cumulate_EMA(model, ema_weights, MOMENTUM_EMA)
        current_state_dict = model.state_dict()
        model.load_state_dict(ema_weights)

        pred_valid_ema, labels_valid_ema = evaluation(model, dataloader_valid, device)
        f1_val_ema = f1_score(labels_valid_ema, pred_valid_ema, average="weighted")

        pred_test_ema, labels_test_ema = evaluation(model, dataloader_test, device)
        f1_test_ema = f1_score(labels_test_ema, pred_test_ema, average="weighted")

        model.load_state_dict(current_state_dict)
    
    #print("current best %.2f EMA %.2f ORIG %.2f on the validation set"%(global_valid, f1_val_ema, f1_val))
    if f1_val > global_valid or f1_val_ema > global_valid:
        global_valid = max(f1_val, f1_val_ema)
        if f1_val > f1_val_ema:
            print("TRAIN LOSS at Epoch %d: %.4f with BEST ACC on TEST SET %.2f with training time %d"%(epoch, tot_loss/den, 100*f1_test,  (end-start)))
            torch.save(model.state_dict(), output_file)
        else:
            print("TRAIN LOSS at Epoch %d: %.4f with BEST ACC (from EMA) on TEST SET %.2f with training time %d"%(epoch, tot_loss/den, 100*f1_test_ema, (end-start)))    
            current_state_dict = model.state_dict()
            model.load_state_dict(ema_weights)
            torch.save(model.state_dict(), output_file)
            model.load_state_dict(current_state_dict)
    else:
        print("TRAIN LOSS at Epoch %d: %.4f with EMA ACC %.2f training time %d"%(epoch, tot_loss/den, 100*f1_test_ema, (end-start)))        
    sys.stdout.flush()