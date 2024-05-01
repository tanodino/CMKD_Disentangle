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
from model_pytorch import DisCoMKD
import time
from sklearn.metrics import f1_score
from functions import TRAIN_BATCH_SIZE, LEARNING_RATE, EPOCHS, transform, MyDataset, hashPREFIX2SOURCE
import os
import warnings
warnings.filterwarnings('ignore')


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
        pred_npy_f = np.argmax(pred_f.cpu().detach().numpy(), axis=1)
        tot_pred_f.append( pred_npy_f )

        pred_npy_s = np.argmax(pred_s.cpu().detach().numpy(), axis=1)
        tot_pred_s.append( pred_npy_s )

        tot_labels.append( y_batch.cpu().detach().numpy())
    tot_pred_f = np.concatenate(tot_pred_f)
    tot_pred_s = np.concatenate(tot_pred_s)
    tot_labels = np.concatenate(tot_labels)
    return tot_pred_f, tot_pred_s, tot_labels

def main():
    # DIRECTORY WHERE DATA ARE STORED
    dir_ = sys.argv[1]

    # PREFIX OF THE FIRST MODALITY
    first_prefix = sys.argv[2]

    # PREFIX OF THE SECOND MODALITY
    second_prefix = sys.argv[3]

    prefix_method = "DisCoM-KD"
    #CREATE FOLDER TO STORE RESULTS
    dir_name_f = dir_+"/OUR-%s_%s"%(prefix_method, first_prefix)
    if not os.path.exists(dir_name_f):
        os.mkdir(dir_name_f)

    output_file_f = dir_name_f+"/WEIGHTS.pth"

    dir_name_s = dir_+"/OUR-%s_%s"%(prefix_method, second_prefix)
    if not os.path.exists(dir_name_s):
        os.mkdir(dir_name_s)

    output_file_s = dir_name_s+"/WEIGHTS.pth"

    first_enc = hashPREFIX2SOURCE[first_prefix]
    second_enc = hashPREFIX2SOURCE[second_prefix]

    # WE HAVE PRE NORMALIZED THE DATA IN ORDER TO AVOID THIS STEP ONLINE
    # first_data HAS A DIMENSION S x C1 x H1 x W1 where S number of samples, C1 number of channels of the first modality, H1 height of the images of the first modality, W1 width of the images of the first modality
    # second_data HAS A DIMENSION S x C2 x H2 x W2 where S number of samples, C2 number of channels of the second modality, H2 height of the images of the second modality, W2 width of the images of the second modality

    first_data = np.load("%s/%s_data_normalized.npy"%(dir_,first_prefix))
    second_data = np.load("%s/%s_data_normalized.npy"%(dir_,second_prefix))

    # labels HAS A DIMENSION  S where S number of samples

    labels = np.load("%s/labels.npy"%dir_)

    # THESE ARE THE IDX (ROW NUMBER) CORRESPONDING TO THE TRAINING, VALIDATION AND TEST SET
    train_idx = np.load("%s/train_idx.npy"%(dir_))
    valid_idx = np.load("%s/valid_idx.npy"%(dir_)) 
    test_idx = np.load("%s/test_idx.npy"%(dir_))

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

    dataloader_train_f = createDataLoader2(train_f_data, train_label_f, True, transform, TRAIN_BATCH_SIZE, type_data=first_prefix)
    dataloader_train_s = createDataLoader2(train_s_data, train_label_s, True, transform, TRAIN_BATCH_SIZE, type_data=second_prefix)

    #DATALOADER VALID
    dataloader_valid = createDataLoader(valid_f_data, valid_s_data, valid_labels, False, 512)

    #DATALOADER TEST
    dataloader_test = createDataLoader(test_f_data, test_s_data, test_labels, False, 512)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #DisCoM-KD
    model = DisCoMKD(input_channel_first=first_data.shape[1], input_channel_second=second_data.shape[1],  num_classes=n_classes, f_encoder=first_enc, s_encoder=second_enc)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    global_valid_f = 0
    global_valid_s = 0

    for epoch in range(EPOCHS):
        start = time.time()
        model.train()
        tot_loss = 0.0
        den = 0
        lambda_ = 1.0

        for xy_f, xy_s in zip(dataloader_train_f,dataloader_train_s):
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
            pred_s_shared, \
            pred_task_f_domain_discr, \
            pred_task_s_domain_discr =  model([x_batch_f, x_batch_s], lambda_val=lambda_)

            # L_{CL}
            tot_pred = torch.cat([pred_f, pred_s])   
            loss_pred = loss_fn(tot_pred, torch.cat([y_batch_f, y_batch_s]) )

            # L_{ADV}
            tot_pred_adv = torch.cat([discr_f, discr_s])
            y_dom = torch.cat([ torch.ones_like(discr_f), torch.zeros_like(discr_s)] )
            loss_adv_dann = loss_fn( tot_pred_adv, y_dom )    

            # L_{MOD}
            tot_pred_dom = torch.cat([pred_dom_f, pred_dom_s])
            y_dom = torch.cat([ torch.ones_like(pred_dom_f), torch.zeros_like(pred_dom_s)] )
            loss_pred_dom =loss_fn(tot_pred_dom, y_dom)

            # L_{AUX}
            loss_pred_domain_discr = loss_fn( torch.cat([pred_task_f_domain_discr, pred_task_s_domain_discr]), torch.cat([y_batch_f, y_batch_s]) )
            tot_pred_shared = torch.cat([pred_f_shared, pred_s_shared]) 
            loss_pred_shared = loss_fn(tot_pred_shared, torch.cat([y_batch_f, y_batch_s]) )
            
            # L_{\perp}
            emb_dom_info = nn.functional.normalize( torch.cat([f_domain_discr, s_domain_discr]) )
            emb_dom_uninfo = nn.functional.normalize( torch.cat([f_domain_useless, s_domain_useless]) )
            emb_shared = nn.functional.normalize( torch.cat([f_shared_discr, s_shared_discr]) )

            loss_ortho = torch.mean( torch.sum(emb_dom_info * emb_dom_uninfo, dim=1) ) \
                        + torch.mean( torch.sum(emb_dom_info * emb_shared, dim=1) ) \

            loss = loss_pred + loss_pred_shared + loss_pred_domain_discr + loss_pred_dom  + loss_adv_dann  + loss_ortho #+ loss_contra_sel#+ loss_contra_sel#+ loss_adv_dann + loss_adv_dann #+ # + loss_contra_cl #+ loss_adv_dann#
            
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


if __name__ == "__main__":
    main()