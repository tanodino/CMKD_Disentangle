#COMPETITORS:
#   + Classical KD -> KL
#   + Decoupled KD -> http://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.html
#   + Multi-Level KD -> http://openaccess.thecvf.com/content/CVPR2023/html/Jin_Multi-Level_Logit_Distillation_CVPR_2023_paper.html
#   - DML            -> https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Deep_Mutual_Learning_CVPR_2018_paper.pdf


import torch
import torch.nn as nn
import sys
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sys
from sklearn.utils import shuffle
#from model_transformer import TransformerEncoder
from model_pytorch import CrossSourceModelV2, SupervisedContrastiveLoss
import time
from sklearn.metrics import f1_score
from functions import TRAIN_BATCH_SIZE, LEARNING_RATE, EPOCHS, WARM_UP_EPOCH_EMA, cumulate_EMA, MOMENTUM_EMA, transform, MyDataset, hashPREFIX2SOURCE
import os

def createDataLoader2(x, y, tobeshuffled, transform , BATCH_SIZE, type_data='RGB'):
    #DATALOADER TRAIN
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.int64)
    dataest = None
    #'DEPTH','RGB','MS','SAR','SPECTRO','MNIST','FULL','HALF', "THERMAL"}
    if type_data == 'RGB' or type_data=='MS' or type_data=='MNIST':
        dataset = MyDataset(x_tensor, y_tensor, transform=transform)
    else:
        dataset = TensorDataset(x_tensor, y_tensor)
    dataloader = DataLoader(dataset, shuffle=tobeshuffled, batch_size=BATCH_SIZE)
    return dataloader



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
        pred = model.pred_firstEnc(x_batch_f)
        #_,_,_,_,_,_,pred,_ = model([x_batch_f, x_batch_s])
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
method = sys.argv[5]
#target_prefix = sys.argv[2]
#nsamples = sys.argv[3]
#nsplit = sys.argv[4]

#python main_cs.py PAVIA_UNIVERSITY HALF FULL 0 ORTHO

#CREATE FOLDER TO STORE RESULTS
dir_name = dir_+"/OUR-PROJH_%s_%s"%(first_prefix, method)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

output_file = dir_name+"/%s.pth"%(run_id)

first_enc = hashPREFIX2SOURCE[first_prefix]
second_enc = hashPREFIX2SOURCE[second_prefix]

print("first_enc %s"%first_enc)
print("second_enc %s"%second_enc)

'''
first_enc = None
second_enc = None
if dir_ == "PAVIA_UNIVERSITY":
    first_enc = 'hyper'
    second_enc = 'hyper'
elif dir_ == 'SUNRGBD' or dir_ == 'EUROSAT':
    first_enc = 'image'
    second_enc = 'image'
'''


#sar_data = np.load("%s/SAR_data_normalized.npy"%dir_)
#ms_data = np.load("%s/MS_data_normalized.npy"%dir_)
#labels = np.load("%s/labels.npy"%dir_)

#TRICK TO EASILY SWITCH BETWEEN OPT AND SAR AS SOURCES
first_data = np.load("%s/%s_data_normalized.npy"%(dir_,first_prefix))
second_data = np.load("%s/%s_data_normalized.npy"%(dir_,second_prefix))
labels = np.load("%s/labels.npy"%dir_)

print("FIRST DATA ",first_data.shape)
print("SECOND DATA ",second_data.shape)

#TRICK TO EASILY SWITCH BETWEEN OPT AND SAR AS SOURCES

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
#dataloader_train = createDataLoader(train_ms_data, train_sar_data, train_labels, True, TRAIN_BATCH_SIZE)
dataloader_train_f = createDataLoader2(train_f_data, train_labels, True, transform, TRAIN_BATCH_SIZE*2, type_data=first_prefix)
dataloader_train_s = createDataLoader2(train_s_data, train_labels, True, transform, TRAIN_BATCH_SIZE*2, type_data=second_prefix)

#DATALOADER VALID
dataloader_valid = createDataLoader(valid_f_data, valid_s_data, valid_labels, False, 256)

#DATALOADER TEST
dataloader_test = createDataLoader(test_f_data, test_s_data, test_labels, False, 256)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

#model = CrossSourceModel(input_channel_first=ms_data.shape[1], input_channel_second=sar_data.shape[1])
model = CrossSourceModelV2(input_channel_first=first_data.shape[1], input_channel_second=second_data.shape[1],  num_classes=n_classes, f_encoder=first_enc, s_encoder=second_enc)
model = model.to(device)


learning_rate = 0.0001
loss_fn = nn.CrossEntropyLoss()
scl = SupervisedContrastiveLoss()
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
    for x_batch_f, y_batch_f in dataloader_train_f:
        x_batch_s, y_batch_s = next(iter(dataloader_train_s))

        optimizer.zero_grad()
        x_batch_f = x_batch_f.to(device)
        x_batch_s = x_batch_s.to(device)
        y_batch_f = y_batch_f.to(device)
        y_batch_s = y_batch_s.to(device)

        f_emb_inv, \
        f_emb_spec, \
        s_emb_inv, \
        s_emb_spec, \
        pred_dom_f, \
        pred_dom_s, \
        pred_f, \
        pred_s =  model([x_batch_f, x_batch_s])
        
        tot_pred = torch.cat([pred_f, pred_s])   
        loss_pred = loss_fn(tot_pred, torch.cat([y_batch_f, y_batch_s]) )

        emb_inv = nn.functional.normalize( torch.cat([f_emb_inv, s_emb_inv]) )
        emb_spec = nn.functional.normalize( torch.cat([f_emb_spec, s_emb_spec]) )
        
        #emb_inv = nn.functional.normalize( torch.cat([f_emb_inv, s_emb_inv, f_emb_inv, s_emb_inv]) )
        #emb_spec = nn.functional.normalize( torch.cat([f_emb_spec, s_emb_spec, s_emb_spec, f_emb_spec ]) )
        loss_ortho = torch.mean( torch.sum(emb_inv * emb_spec, dim=1) )

        tot_pred_dom = torch.cat([pred_dom_f, pred_dom_s])
        y_dom = torch.cat([ torch.zeros_like(pred_dom_f), torch.ones_like(pred_dom_s)] )
        loss_pred_dom =loss_fn(tot_pred_dom, y_dom)


        #scl
        emb_scl = nn.functional.normalize( torch.cat([f_emb_inv, s_emb_inv, f_emb_spec, s_emb_spec]) )
        #emb_scl = nn.functional.normalize( torch.cat([f_emb_inv, s_emb_inv]) )
        #y_scl = torch.cat([y_batch_opt, y_batch_sar])
        y_scl = torch.cat([y_batch_f, y_batch_s, torch.ones_like(y_batch_f)*n_classes, torch.ones_like(y_batch_s)*(n_classes+1)  ])
        #y_scl = torch.cat([y_batch_opt, y_batch_sar + n_classes, torch.ones_like(y_batch_opt)*(2*n_classes), torch.ones_like(y_batch_sar)*(2*n_classes+1)  ])
        loss_contra = scl( emb_scl , y_scl )

        loss = loss_pred + loss_pred_dom
        if method == "CONTRA":
            loss = loss + loss_contra #loss_ortho #+ loss_contra#+ loss_contra #  #
        elif method == "ORTHO":
            loss = loss + loss_ortho
        
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
    
    print("current best %.2f EMA %.2f ORIG %.2f on the validation set"%(global_valid, f1_val_ema, f1_val))
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