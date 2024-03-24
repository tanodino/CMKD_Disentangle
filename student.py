### To implemente the Transformer Framework I used the code from this website : https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch

import torch
import torch.nn as nn
import sys
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sys
from sklearn.utils import shuffle
#from model_transformer import TransformerEncoder
from model_pytorch import MultiSourceModel, ModelHYPER, MonoSourceModel
import time
from sklearn.metrics import f1_score
from torchvision.models import resnet18
from sklearn.model_selection import train_test_split
from functions import TRAIN_BATCH_SIZE, LEARNING_RATE, EPOCHS, WARM_UP_EPOCH_EMA, cumulate_EMA, MOMENTUM_EMA, hashPREFIX2SOURCE
import os
from kd_losses import kd_loss, dkd_loss, mlkd_loss
import warnings 
warnings.filterwarnings("ignore")


def getTeacherLogits(dir_, first_prefix, second_prefix, fusion_type, run_id):
    folder_name = dir_+"/TEACHER_%s"%fusion_type
    first_data = np.load("%s/%s_data_normalized.npy"%(dir_,first_prefix))
    train_idx = np.load("%s/train_idx_%d.npy"%(dir_,run_id))
    second_data = None
    train_s_data = None
    if dir_ != "PAVIA_UNIVERSITY":
        second_data = np.load("%s/%s_data_normalized.npy"%(dir_,second_prefix))
        train_s_data = second_data[train_idx]
    
    labels = np.load("%s/labels.npy"%dir_)

    train_f_data = first_data[train_idx]    
    train_labels = labels[train_idx]
    n_classes = len(np.unique(labels))

    #DATALOADER TRAIN
    dataloader_train = None
    
    model = None
    if dir_ != "PAVIA_UNIVERSITY":
        dataloader_train = createDataLoaderTrain(train_f_data, train_s_data, train_labels, False, TRAIN_BATCH_SIZE)
        model = MultiSourceModel(input_channel_first=first_data.shape[1], input_channel_second=second_data.shape[1], f_encoder=hashPREFIX2SOURCE[first_prefix], s_encoder=hashPREFIX2SOURCE[second_prefix], fusion_type=fusion_type, num_classes=n_classes)
    else:
        dataloader_train = createDataLoader(train_f_data, train_labels, False, TRAIN_BATCH_SIZE)
        model = ModelHYPER(num_classes=n_classes)
    
    model = model.to(device)
    model.load_state_dict(torch.load(folder_name+"/%d.pth"%run_id))
    
    if dir_ != "PAVIA_UNIVERSITY":
        return getLogits(model, dataloader_train, device)
    else:
        return getLogitsHYPER(model, dataloader_train, device)


def getLogitsHYPER(model, dataloader, device):
    model.eval()
    logits = []
    for x_batch_f, y_batch in dataloader:
        x_batch_f = x_batch_f.to(device)
        y_batch = y_batch.to(device)
        pred = model(x_batch_f)
        logits.append( pred.cpu().detach().numpy() )
    return np.concatenate(logits, axis=0)


def getLogits(model, dataloader, device):
    model.eval()
    logits = []
    for x_batch_f, x_batch_s, y_batch in dataloader:
        x_batch_f = x_batch_f.to(device)
        x_batch_s = x_batch_s.to(device)
        y_batch = y_batch.to(device)
        pred = model([x_batch_f, x_batch_s])
        logits.append( pred.cpu().detach().numpy() )
    return np.concatenate(logits, axis=0)

def createDataLoaderTrain(data, teacher_logits, labels, tobeshuffled, BATCH_SIZE):
    x_tensor = torch.tensor(data, dtype=torch.float32)
    x_teacher_logits = torch.tensor(teacher_logits, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.int64)

    dataset = TensorDataset(x_tensor, x_teacher_logits, y_tensor)
    dataloader = DataLoader(dataset, shuffle=tobeshuffled, batch_size=BATCH_SIZE)
    return dataloader

def createDataLoader(x, y, tobeshuffled, BATCH_SIZE):
    #DATALOADER TRAIN
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.int64)

    dataset = TensorDataset(x_tensor, y_tensor)
    dataloader = DataLoader(dataset, shuffle=tobeshuffled, batch_size=BATCH_SIZE)
    return dataloader

def evaluationStudent(model, dataloader, device):
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


#python student.py SUNRGBD RGB DEPTH SUM [RGB|DEPTH] [KD|DKD|MLKD] 0
#python student.py EUROSAT MS SAR SUM [MS|SAR] [KD|DKD|MLKD] 0
#python student.py PAVIA_UNIVERSITY FULL FULL FULL HALF [KD|DKD|MLKD] 0

dir_ = sys.argv[1]

first_prefix = sys.argv[2]
second_prefix = sys.argv[3]
fusion_type = sys.argv[4]

student_prefix = sys.argv[5]
kd_loss_name = sys.argv[6]

run_id = int(sys.argv[7])


#CREATE FOLDER TO STORE RESULTS
dir_name = dir_+"/STUDENT_%s_%s"%(student_prefix, kd_loss_name)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

output_file = dir_name+"/%s.pth"%(run_id)


#RETRIEVING TEACHER LOGIT
device = 'cuda' if torch.cuda.is_available() else 'cpu'
teacher_logits = getTeacherLogits(dir_, first_prefix, second_prefix, fusion_type, run_id)

first_data = np.load("%s/%s_data_normalized.npy"%(dir_,student_prefix))
labels = np.load("%s/labels.npy"%dir_)

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

train_f_data, teacher_logits, train_labels = shuffle(train_f_data, teacher_logits, train_labels)

#DATALOADER TRAIN
dataloader_train = createDataLoaderTrain(train_f_data, teacher_logits, train_labels, True, TRAIN_BATCH_SIZE)

#DATALOADER VALID
dataloader_valid = createDataLoader(valid_f_data, valid_labels, False, TRAIN_BATCH_SIZE)

#DATALOADER TEST
dataloader_test = createDataLoader(test_f_data, test_labels, False, TRAIN_BATCH_SIZE)

model = None
#if dir_ == "PAVIA_UNIVERSITY":
#    model = ModelHYPER(num_classes=n_classes)
#else:
model = MonoSourceModel(input_channel_first=first_data.shape[1], encoder=hashPREFIX2SOURCE[first_prefix], num_classes=n_classes)

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
    for x_batch_f, x_teacher_logit, y_batch in dataloader_train:
        optimizer.zero_grad()
        x_batch_f = x_batch_f.to(device)
        x_teacher_logit = x_teacher_logit.to(device)
        y_batch = y_batch.to(device)

        pred = model(x_batch_f)   
        loss_pred = loss_fn(pred, y_batch)  

        loss_kd = None
        loss = None
        if kd_loss_name == "KD": # ORIGINAL
            loss_kd = kd_loss(pred, x_teacher_logit)
            #weighting strategy taken by "Multi-level Logit Distillation" CVPR 2023
            loss = .1 * loss_pred + .9*loss_kd
        if kd_loss_name == "DKD": # 2022
            loss_kd = dkd_loss(pred, x_teacher_logit, y_batch)
            #weighting strategy taken by "Multi-level Logit Distillation" CVPR 2023
            loss = loss_pred + min(epoch / 20., 1.) * loss_kd
        if kd_loss_name == "MLKD":# 2023
            loss_kd = mlkd_loss(pred, x_teacher_logit)
            #weighting strategy taken by "Multi-level Logit Distillation" CVPR 2023
            loss = loss_pred + loss_kd
        
        loss.backward() # backward pass: backpropagate the prediction loss
        optimizer.step() # gradient descent: adjust the parameters by the gradients collected in the backward pass
        tot_loss+= loss.cpu().detach().numpy()
        den+=1.

    end = time.time()
    pred_valid, labels_valid = evaluationStudent(model, dataloader_valid, device)
    f1_val = f1_score(labels_valid, pred_valid, average="weighted")

    pred_test, labels_test = evaluationStudent(model, dataloader_test, device)
    f1_test = f1_score(labels_test, pred_test, average="weighted")

    f1_test_ema = 0.0
    f1_val_ema = 0.0
    if epoch >= WARM_UP_EPOCH_EMA:
        ema_weights = cumulate_EMA(model, ema_weights, MOMENTUM_EMA)
        current_state_dict = model.state_dict()
        model.load_state_dict(ema_weights)

        pred_valid_ema, labels_valid_ema = evaluationStudent(model, dataloader_valid, device)
        f1_val_ema = f1_score(labels_valid_ema, pred_valid_ema, average="weighted")

        pred_test_ema, labels_test_ema = evaluationStudent(model, dataloader_test, device)
        f1_test_ema = f1_score(labels_test_ema, pred_test_ema, average="weighted")

        model.load_state_dict(current_state_dict)
    
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