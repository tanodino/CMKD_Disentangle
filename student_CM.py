### To Implement -> https://arxiv.org/abs/2403.01427


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
from functions import TRAIN_BATCH_SIZE, LEARNING_RATE, EPOCHS, hashPREFIX2SOURCE, CosineDecay, transform, MyDatasetMM
import os
from kd_losses import kd_loss, dkd_loss, mlkd_loss, normalizeLogit
from temp_global import Global_T
import warnings 
warnings.filterwarnings("ignore")



def getTeacherModel(dir_, prefix, labels, run_id, hashPREFIX2SOURCE):
    folder_name = dir_+"/MONO_%s"%prefix
    first_data = np.load("%s/%s_data_normalized.npy"%(dir_,prefix))
    n_classes = len(np.unique(labels))
    
    model = MonoSourceModel(input_channel_first=first_data.shape[1], encoder=hashPREFIX2SOURCE[prefix], num_classes=n_classes)
        
    model = model.to(device)
    model.load_state_dict(torch.load(folder_name+"/%d.pth"%run_id))

    return model, first_data

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

def createDataLoaderTrain(x_ms, x_sar, y, tobeshuffled, BATCH_SIZE, transform=None):
    #DATALOADER TRAIN
    x_ms_tensor = torch.tensor(x_ms, dtype=torch.float32)
    x_sar_tensor = torch.tensor(x_sar, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.int64)

    dataest = None
    if transform is None:
        dataset = TensorDataset(x_ms_tensor, x_sar_tensor, y_tensor)
    else:
        dataset = MyDatasetMM(x_ms_tensor, x_sar_tensor, y_tensor, transform=transform)
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

dir_ = sys.argv[1]  #SUNRGBD | TRISTAR | EUROSAT

first_prefix = sys.argv[2] # RGB | DEPTH | MS | SAR | THERMAL
second_prefix = sys.argv[3] # RGB | DEPTH | MS | SAR | THERMAL
#fusion_type = sys.argv[4] # SUM

#student_prefix = sys.argv[5] # MS | SAR | RGB | DEPTH | THERMAL
kd_loss_name = sys.argv[4] # KD1 | KD2 | DKD | MLKD | CTKD
z_score_norm = int(sys.argv[5]) # 0 | 1

run_id = int(sys.argv[6])

#CREATE FOLDER TO STORE RESULTS
dir_name = None
if z_score_norm:
    dir_name = dir_+"/STUDENT_CM_%s_%s_%s_ZNORM"%(first_prefix, second_prefix, kd_loss_name)
else:
    dir_name = dir_+"/STUDENT_CM_%s_%s_%s"%(first_prefix, second_prefix, kd_loss_name)

print("dir_name %s"%dir_name)


if not os.path.exists(dir_name):
    os.mkdir(dir_name)

output_file = dir_name+"/%s.pth"%(run_id)


#RETRIEVING TEACHER LOGIT
device = 'cuda' if torch.cuda.is_available() else 'cpu'
labels = np.load("%s/labels.npy"%dir_)
teacher_model, first_data = getTeacherModel(dir_, first_prefix, labels, run_id, hashPREFIX2SOURCE)
second_data = np.load("%s/%s_data_normalized.npy"%(dir_,second_prefix))

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

dataloader_train = createDataLoaderTrain(train_f_data, train_s_data, train_labels, True, TRAIN_BATCH_SIZE, transform=transform)

dataloader_valid = None
dataloader_test = None
model = None

dataloader_valid = createDataLoader(valid_s_data, valid_labels, False, TRAIN_BATCH_SIZE)
#DATALOADER TEST
dataloader_test = createDataLoader(test_s_data, test_labels, False, TRAIN_BATCH_SIZE)
model = MonoSourceModel(input_channel_first=second_data.shape[1], encoder=hashPREFIX2SOURCE[second_prefix], num_classes=n_classes)
model = model.to(device)


module_list = nn.ModuleList([])
module_list.append( model )

global_mlp = None
gradient_decay = None
t_start = 1.0
t_range = 20

if kd_loss_name == 'CTKD':
    gradient_decay = CosineDecay(max_value=0, min_value=-1., num_loops=5.0)
    global_mlp = Global_T()
    global_mlp = global_mlp.to(device)
    module_list.append( global_mlp )


learning_rate = 0.0001
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=module_list.parameters(), lr=LEARNING_RATE)

# Loop through the data
global_valid = 0
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

        pred = None
        pred = model(x_batch_s)
        
        x_teacher_logit = teacher_model(x_batch_f)
        x_teacher_logit = x_teacher_logit.detach()
           
        loss_pred = loss_fn(pred, y_batch)  

        if z_score_norm:
            pred = normalizeLogit(pred)
            x_teacher_logit = normalizeLogit(x_teacher_logit)

        loss_kd = None
        loss = None

        if kd_loss_name == "KD1": # ORIGINAL with alpha = 0
            loss_kd = kd_loss(pred, x_teacher_logit)
            alpha = 0.
            loss = alpha * loss_pred + (1-alpha)*loss_kd
        
        if kd_loss_name == "KD2": # ORIGINAL with alpha = 0.5
            loss_kd = kd_loss(pred, x_teacher_logit)
            alpha = .5
            loss = alpha * loss_pred + (1-alpha)*loss_kd
        
        if kd_loss_name == "DKD": # 2022
            loss_kd = dkd_loss(pred, x_teacher_logit, y_batch)
            #weighting strategy taken by "Multi-level Logit Distillation" CVPR 2023
            loss = loss_pred + min(epoch / 20., 1.) * loss_kd
        
        if kd_loss_name == "MLKD":# 2023
            loss_kd = mlkd_loss(pred, x_teacher_logit)
            #weighting strategy taken by "Multi-level Logit Distillation" CVPR 2023
            loss = loss_pred + loss_kd
        if kd_loss_name == 'CTKD':
            decay_value = gradient_decay.get_value(epoch)
            temp = global_mlp(decay_value)  # (teacher_output, student_output)
            temp = t_start + t_range * torch.sigmoid(temp)
            temp = temp.cuda()
            temp = temp[0]
            loss_kd = kd_loss(pred, x_teacher_logit, temperature=temp)
            alpha = .1
            loss = alpha*loss_pred + (1-alpha)*loss_kd

        loss.backward() # backward pass: backpropagate the prediction loss
        optimizer.step() # gradient descent: adjust the parameters by the gradients collected in the backward pass
        tot_loss+= loss.cpu().detach().numpy()
        den+=1.

    end = time.time()
    pred_valid, labels_valid = evaluationStudent(model, dataloader_valid, device)
    f1_val = f1_score(labels_valid, pred_valid, average="weighted")

    pred_test, labels_test = evaluationStudent(model, dataloader_test, device)
    f1_test = f1_score(labels_test, pred_test, average="weighted")
    
    if f1_val > global_valid :
        global_valid = f1_val
        print("TRAIN LOSS at Epoch %d: %.4f with BEST ACC on TEST SET %.2f with training time %d"%(epoch, tot_loss/den, 100*f1_test,  (end-start)))
        torch.save(model.state_dict(), output_file)
    else:
        print("TRAIN LOSS at Epoch %d: %.4f with F1 %.2f training time %d"%(epoch, tot_loss/den, 100*f1_test, (end-start)))        
    sys.stdout.flush()