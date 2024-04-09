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
from model_pytorch import CrossSourceModelGRLv2, SupervisedContrastiveLoss, CrossSourceModelGRL, CrossSourceModelGRLv3
import time
from sklearn.metrics import f1_score
from functions import TRAIN_BATCH_SIZE, LEARNING_RATE, EPOCHS, WARM_UP_EPOCH_EMA, cumulate_EMA, MOMENTUM_EMA, transform, MyDataset, hashPREFIX2SOURCE, CosineDecay, MyDataset4
import os
import warnings
warnings.filterwarnings('ignore')


def createDataLoader4(x1, x2, y1, y2, tobeshuffled, transform, BATCH_SIZE, type_data1='RGB', type_data2='RGB'):
    x1_tensor = torch.tensor(x1, dtype=torch.float32)
    y1_tensor = torch.tensor(y1, dtype=torch.int64)
    x2_tensor = torch.tensor(x2, dtype=torch.float32)
    y2_tensor = torch.tensor(y2, dtype=torch.int64)

    dataset = None
    transform1 = None
    transform2 = None

    #'DEPTH','RGB','MS','SAR','SPECTRO','MNIST',"THERMAL"}
    if type_data1 == 'RGB' or type_data1=='MS' or type_data1=='MNIST' or type_data1=='SAR' or type_data1=='DEPTH' or type_data1=='THERMAL':
        transform1 = transform
    
    if type_data2 == 'RGB' or type_data2=='MS' or type_data2=='MNIST' or type_data2=='SAR' or type_data2=='DEPTH' or type_data2=='THERMAL':
        transform2 = transform
    
    dataset = MyDataset4(x1_tensor, y1_tensor, x2_tensor, y2_tensor, transform1=transform1, transform2=transform2)    
    dataloader = DataLoader(dataset, shuffle=tobeshuffled, batch_size=BATCH_SIZE)
    return dataloader



def createDataLoader2(x, y, tobeshuffled, transform , BATCH_SIZE, type_data='RGB'):
    #DATALOADER TRAIN
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.int64)
    dataset = None
    #'DEPTH','RGB','MS','SAR','SPECTRO','MNIST',"THERMAL"}
    if type_data == 'RGB' or type_data=='MS' or type_data=='MNIST' or type_data=='SAR' or type_data=='DEPTH' or type_data=='THERMAL':
        dataset = MyDataset(x_tensor, y_tensor, transform=transform)
    else:
        dataset = TensorDataset(x_tensor, y_tensor)
    dataloader = DataLoader(dataset, shuffle=tobeshuffled, batch_size=BATCH_SIZE)
    return dataloader

def createDataSet(x, y, transform , type_data='RGB'):
    #DATALOADER TRAIN
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.int64)
    dataset = None
    #'DEPTH','RGB','MS','SAR','SPECTRO','MNIST',"THERMAL"}
    if type_data == 'RGB' or type_data=='MS' or type_data=='MNIST' or type_data=='SAR' or type_data=='DEPTH' or type_data=='THERMAL':
        dataset = MyDataset(x_tensor, y_tensor, transform=transform)
    else:
        dataset = TensorDataset(x_tensor, y_tensor)
    #dataloader = DataLoader(dataset, shuffle=tobeshuffled, batch_size=BATCH_SIZE)
    #return dataloader
    return dataset




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
    tot_pred_f = []
    tot_pred_s = []
    tot_labels = []
    for x_batch_f, x_batch_s, y_batch in dataloader:
        x_batch_f = x_batch_f.to(device)
        x_batch_s = x_batch_s.to(device)
        y_batch = y_batch.to(device)
        pred_f = model.pred_firstEnc(x_batch_f)
        pred_s = model.pred_secondEnc(x_batch_s)
        #_,_,_,_,_,_,pred,_ = model([x_batch_f, x_batch_s])
        pred_npy_f = np.argmax(pred_f.cpu().detach().numpy(), axis=1)
        tot_pred_f.append( pred_npy_f )

        pred_npy_s = np.argmax(pred_s.cpu().detach().numpy(), axis=1)
        tot_pred_s.append( pred_npy_s )

        tot_labels.append( y_batch.cpu().detach().numpy())
    tot_pred_f = np.concatenate(tot_pred_f)
    tot_pred_s = np.concatenate(tot_pred_s)
    tot_labels = np.concatenate(tot_labels)
    return tot_pred_f, tot_pred_s, tot_labels

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

prefix_method = "v5"
#CREATE FOLDER TO STORE RESULTS
dir_name_f = dir_+"/OUR-%s_%s_%s"%(prefix_method, first_prefix, method)
if not os.path.exists(dir_name_f):
    os.mkdir(dir_name_f)

output_file_f = dir_name_f+"/%s.pth"%(run_id)

dir_name_s = dir_+"/OUR-%s_%s_%s"%(prefix_method, second_prefix, method)
if not os.path.exists(dir_name_s):
    os.mkdir(dir_name_s)

output_file_s = dir_name_s+"/%s.pth"%(run_id)



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

train_f_data, train_label_f = shuffle(train_f_data, train_labels)
train_s_data, train_label_s = shuffle(train_s_data, train_labels)
#train_s_data, train_labels = shuffle(train_f_data, train_s_data, train_labels)


#train_f_data, train_labels = shuffle(train_f_data, train_labels)
#train_s_data, train_labels = shuffle(train_s_data, train_labels)


#DATALOADER TRAIN
#dataloader_train = createDataLoader(train_ms_data, train_sar_data, train_labels, True, TRAIN_BATCH_SIZE)

dataloader_train_f = createDataLoader2(train_f_data, train_label_f, True, transform, TRAIN_BATCH_SIZE, type_data=first_prefix)
dataloader_train_s = createDataLoader2(train_s_data, train_label_s, True, transform, TRAIN_BATCH_SIZE, type_data=second_prefix)

#dataloader_train = createDataLoader4(train_f_data, train_s_data, train_label_f, train_label_s, True, transform, TRAIN_BATCH_SIZE, type_data1=first_prefix, type_data2=second_prefix)

#train_f_data, train_s_data, train_labels = shuffle(train_f_data, train_s_data, train_labels)
#train_f_data, train_labels = shuffle(train_f_data, train_labels)
#train_s_data, train_labels = shuffle(train_s_data, train_labels)

#dataset_train_f = createDataSet(train_f_data, train_labels, transform, type_data=first_prefix)
#dataset_train_s = createDataSet(train_s_data, train_labels, transform, type_data=second_prefix)

#dataset = torch.utils.data.TensorDataset(dataset_train_f, dataset_train_s)
#train_dataloader = DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

#DATALOADER VALID
dataloader_valid = createDataLoader(valid_f_data, valid_s_data, valid_labels, False, 512)

#DATALOADER TEST
dataloader_test = createDataLoader(test_f_data, test_s_data, test_labels, False, 512)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

#model = CrossSourceModel(input_channel_first=ms_data.shape[1], input_channel_second=sar_data.shape[1])
#model = CrossSourceModelGRLv2(input_channel_first=first_data.shape[1], input_channel_second=second_data.shape[1],  num_classes=n_classes, f_encoder=first_enc, s_encoder=second_enc)
model = CrossSourceModelGRLv3(input_channel_first=first_data.shape[1], input_channel_second=second_data.shape[1],  num_classes=n_classes, f_encoder=first_enc, s_encoder=second_enc)
#model = CrossSourceModelGRL(input_channel_first=first_data.shape[1], input_channel_second=second_data.shape[1],  num_classes=n_classes, f_encoder=first_enc, s_encoder=second_enc)
model = model.to(device)


learning_rate = 0.0001
loss_fn = nn.CrossEntropyLoss()
loss_fn_2 = nn.CrossEntropyLoss(reduction='none')
scl = SupervisedContrastiveLoss(temperature=1.)
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

gradient_decay = CosineDecay(max_value=0, min_value=1., num_loops=5.0)
# Loop through the data
global_valid_f = 0
global_valid_s = 0
ema_weights = None
ridg_init = torch.ones(n_classes, device='cuda')
ridg_rational_bank = torch.zeros(n_classes, n_classes, 512, device='cuda')
#ridg_rational_bank = torch.zeros(n_classes, n_classes, 256, device='cuda')
ridg_momentum = 0.1


for epoch in range(EPOCHS):
    start = time.time()
    model.train()
    tot_loss = 0.0
    den = 0
    lambda_ = 1.0
    '''
    train_f_data, train_label_f = shuffle(train_f_data, train_label_f)
    train_s_data, train_label_s = shuffle(train_s_data, train_label_s)
    #dataloader_train_f = createDataLoader2(train_f_data, train_label_f, True, transform, TRAIN_BATCH_SIZE, type_data=first_prefix)
    #dataloader_train_s = createDataLoader2(train_s_data, train_label_s, True, transform, TRAIN_BATCH_SIZE, type_data=second_prefix)
    dataloader_train = createDataLoader4(train_f_data, train_s_data, train_label_f, train_label_s, True, transform, TRAIN_BATCH_SIZE, type_data1=first_prefix, type_data2=second_prefix)
    '''
    #for x_batch_f, x_batch_s, y_batch_f, y_batch_s in dataloader_train:    
    #for xy_s, xy_f in zip(dataloader_train_s, dataloader_train_f):
    #    x_batch_f, y_batch_f = xy_f
    #    x_batch_s, y_batch_s = xy_s
    #for x_batch_s, y_batch_s in dataloader_train_s:
    #    x_batch_f, y_batch_f = next(iter(dataloader_train_f))
    #train_s_data, train_label_s = shuffle(train_s_data, train_label_s)
    #dataloader_train_s = createDataLoader2(train_s_data, train_label_s, True, transform, TRAIN_BATCH_SIZE, type_data=second_prefix)
    #iteration_dataloader_train_s = iter(dataloader_train_s)
    
    for xy_f, xy_s in zip(dataloader_train_f,dataloader_train_s):
    #for x_batch_f, y_batch_f in dataloader_train_f:
    #    x_batch_s, y_batch_s = next(iteration_dataloader_train_s)
        x_batch_f, y_batch_f = xy_f
        x_batch_s, y_batch_s = xy_s
        optimizer.zero_grad()
        x_batch_f = x_batch_f.to(device)
        x_batch_s = x_batch_s.to(device)
        y_batch_f = y_batch_f.to(device)
        y_batch_s = y_batch_s.to(device)
        
        f_shared_discr, \
        s_shared_discr, \
        f_domain_discr, \
        f_domain_useless, \
        s_domain_discr, \
        s_domain_useless, \
        pred_dom_f, \
        pred_dom_s, \
        pred_f, \
        pred_s, \
        discr_f, \
        discr_s, \
        pred_f_shared, \
        pred_s_shared =  model([x_batch_f, x_batch_s], lambda_val=lambda_)

        tot_pred = torch.cat([pred_f, pred_s])   
        loss_pred = loss_fn(tot_pred, torch.cat([y_batch_f, y_batch_s]) )

        tot_pred_shared = torch.cat([pred_f_shared, pred_s_shared]) 
        loss_pred_shared = loss_fn(tot_pred_shared, torch.cat([y_batch_f, y_batch_s]) )

        emb_dom_info = nn.functional.normalize( torch.cat([f_domain_discr, s_domain_discr]) )
        emb_dom_uninfo = nn.functional.normalize( torch.cat([f_domain_useless, s_domain_useless]) )
        emb_shared = nn.functional.normalize( torch.cat([f_shared_discr, s_shared_discr]) )

        #loss_ortho = torch.mean( torch.sum(emb_dom_info * emb_dom_uninfo, dim=1) ) \
        #             + torch.mean( torch.sum(emb_dom_info * emb_shared, dim=1) ) \

        loss_ortho = torch.mean( torch.sum(emb_dom_info * emb_dom_uninfo, dim=1) ) \
                     + torch.mean( torch.sum(torch.cat([emb_dom_info, emb_dom_uninfo],dim=1) * emb_shared, dim=1) )


        tot_pred_dom = torch.cat([pred_dom_f, pred_dom_s])
        y_dom = torch.cat([ torch.ones_like(pred_dom_f), torch.zeros_like(pred_dom_s)] )
        loss_pred_dom =loss_fn(tot_pred_dom, y_dom)


        #scl
        #emb_scl = nn.functional.normalize( torch.cat([f_shared_discr, s_shared_discr, f_domain_discr, s_domain_discr]) )
        #emb_scl = nn.functional.normalize( torch.cat([f_emb_inv, s_emb_inv]) )
        #y_scl = torch.cat([y_batch_f, y_batch_s, torch.ones_like(y_batch_f)*n_classes, torch.ones_like(y_batch_s)*(n_classes+1)  ])
        loss_contra = 0#scl( emb_scl , y_scl )

        
        #loss_contra1 = scl( emb_scl , y_scl )
        #f_emb_discr = torch.cat([f_domain_discr, f_domain_discr],dim=1)
        #s_emb_discr = torch.cat([s_domain_discr, s_domain_discr],dim=1)
        
        emb_scl_sel = nn.functional.normalize( torch.cat([f_domain_discr, s_domain_discr]) )
        #emb_scl_sel = nn.functional.normalize( torch.cat([f_emb_discr, s_emb_discr]) )
        
        y_scl_sel = torch.cat([y_batch_f, y_batch_s])
        loss_contra_sel = scl( emb_scl_sel , y_scl_sel )

        #emb_scl = nn.functional.normalize( torch.cat([f_emb_spec, s_emb_spec]) )
        #y_scl = torch.cat([torch.zeros_like(y_batch_f), torch.ones_like(y_batch_s)])
        #loss_contra2 = scl(emb_scl,y_scl)

        #loss_contra = (loss_contra1 + loss_contra2)/2
        #DANN GRL
        
        tot_pred_adv = torch.cat([discr_f, discr_s])
        y_dom = torch.cat([ torch.ones_like(discr_f), torch.zeros_like(discr_s)] )
        loss_adv_dann = loss_fn( tot_pred_adv, y_dom )    
        
        #emb_scl_cl = nn.functional.softmax( tot_pred, dim=1 )
        emb_scl_cl = nn.functional.normalize(tot_pred)
        loss_contra_cl = scl( emb_scl_cl , y_scl_sel )
        
        #loss = loss_pred  + loss_pred_dom + loss_adv_dann + loss_ortho

        loss = loss_pred + loss_pred_shared + loss_pred_dom  + loss_adv_dann  + loss_ortho #+ loss_contra_sel#+ loss_contra_sel#+ loss_adv_dann + loss_adv_dann #+ # + loss_contra_cl #+ loss_adv_dann#

        
        
        '''
        if method == "CONTRA":
            loss = loss + loss_contra  #loss_ortho #+ loss_contra#+ loss_contra #  #
        elif method == "ORTHO":
            loss = loss + loss_ortho 
        '''
        
        #### LOSS RATIONALE DOMAIN GENERALIZATION #############
        #### ICCV 2023 - Domain Generalization via Rationale Invariance
        '''
        emb_inv = torch.cat([f_emb_inv, s_emb_inv],dim=0)
        all_y = torch.cat([y_batch_f,y_batch_s],dim=0)
        rational = torch.zeros(n_classes, x_batch_f.shape[0]+x_batch_s.shape[0], f_emb_inv.shape[1], device=device)
        for i in range(n_classes):
            rational[i] = model.task_cl.weight[i] * emb_inv
        
        classes = torch.unique(all_y)
        loss_rational = 0
        for i in range(classes.shape[0]):
            rational_mean = rational[:, all_y==classes[i]].mean(dim=1)
            if ridg_init[classes[i]]:
                ridg_rational_bank[classes[i]] = rational_mean
                ridg_init[classes[i]] = False
            else:
                ridg_rational_bank[classes[i]] = (1 - ridg_momentum) * ridg_rational_bank[classes[i]] + ridg_momentum * rational_mean
            loss_rational += ((rational[:, all_y==classes[i]] - (ridg_rational_bank[classes[i]].unsqueeze(1)).detach())**2).sum(dim=2).mean()
            #loss_rational += torch.abs((rational[:, all_y==classes[i]] - (ridg_rational_bank[classes[i]].unsqueeze(1)).detach()) ).sum(dim=2).mean()
        #loss = F.cross_entropy(logits, all_y)

        loss = loss + 1. * loss_rational
        '''
        
        ############################################################ 
        
        loss.backward() # backward pass: backpropagate the prediction loss
        optimizer.step() # gradient descent: adjust the parameters by the gradients collected in the backward pass
        tot_loss+= loss.cpu().detach().numpy()
        den+=1.

    end = time.time()
    pred_valid_f, pred_valid_s, labels_valid = evaluation(model, dataloader_valid, device)
    f1_val_f = f1_score(labels_valid, pred_valid_f, average="weighted")
    f1_val_s = f1_score(labels_valid, pred_valid_s, average="weighted")

    pred_test_f, pred_test_s, labels_test = evaluation(model, dataloader_test, device)
    f1_test_f = f1_score(labels_test, pred_test_f, average="weighted")
    f1_test_s = f1_score(labels_test, pred_test_s, average="weighted")

    final_string = "TRAIN LOSS at Epoch %d: %.4f"%(epoch, tot_loss/den)
    if f1_val_f > global_valid_f:
        global_valid_f = f1_val_f
        torch.save(model.state_dict(), output_file_f)
        final_string = final_string+" BEST on %s F1 on TEST SET %.2f"%(first_prefix, 100*f1_test_f)
    
    if f1_val_s > global_valid_s:
        global_valid_s = f1_val_s
        torch.save(model.state_dict(), output_file_s)
        final_string = final_string+" BEST on %s F1 on TEST SET %.2f"%(second_prefix, 100*f1_test_s)
    
    final_string = final_string+" with training time %d"%((end-start))
    print(final_string)
    sys.stdout.flush()
    torch.cuda.empty_cache()