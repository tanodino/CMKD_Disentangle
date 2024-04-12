import torch
import sys
import numpy as np
import sys
from torch.utils.data import TensorDataset, DataLoader
from model_pytorch import CrossSourceModelGRLv10
from sklearn.metrics import f1_score, accuracy_score
from functions import hashPREFIX2SOURCE
import os

def getPerf(model, model_weights_fileName, test_data, test_labels, first=True):
    model.load_state_dict(torch.load(model_weights_fileName))
    dataloader_test = createDataLoader(test_data, test_labels, False, 512)
    pred_test, labels_test = evaluation(model, dataloader_test, device, first=first)
    f1_test = f1_score(labels_test, pred_test, average="weighted")
    acc_test = accuracy_score(labels_test, pred_test)
    return f1_test, acc_test



def evaluation(model, dataloader, device, first=True):
    model.eval()
    tot_pred = []
    tot_labels = []
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        pred = None
        if first:
            pred = model.pred_firstEnc(x_batch)
        else:
            pred = model.pred_secondEnc(x_batch)
        pred_npy = np.argmax(pred.cpu().detach().numpy(), axis=1)
        tot_pred.append( pred_npy )
        tot_labels.append( y_batch.cpu().detach().numpy())
    
    tot_pred = np.concatenate(tot_pred)
    tot_labels = np.concatenate(tot_labels)
    return tot_pred, tot_labels


def createDataLoader(first_data, y, tobeshuffled, BATCH_SIZE):
    #DATALOADER TRAIN
    first_tensor = torch.tensor(first_data, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.int64)
    dataset = TensorDataset(first_tensor, y_tensor)
    
    dataloader = DataLoader(dataset, shuffle=tobeshuffled, batch_size=BATCH_SIZE)
    return dataloader

#python restore_direct.py PAVIA_UNIVERSITY HALF [KD|DKD|MLKD]
#python restore_direct.py SUNRGBD RGB 
#python restore_direct.py SUNRGBD DEPTH
#python restore_direct.py EUROSAT MS 
#python restore_direct.py EUROSAT SAR


dir_ = sys.argv[1]
first_prefix = sys.argv[2]
seocnd_prefix = sys.argv[3]
method = sys.argv[4]

folder_name_f = dir_+"/OUR-v10_%s_%s"%(first_prefix, method)
folder_name_s = dir_+"/OUR-v10_%s_%s"%(seocnd_prefix, method)

first_enc = hashPREFIX2SOURCE[first_prefix]
second_enc = hashPREFIX2SOURCE[seocnd_prefix]

first_data = np.load("%s/%s_data_normalized.npy"%(dir_,first_prefix))
second_data = np.load("%s/%s_data_normalized.npy"%(dir_,seocnd_prefix))
labels = np.load("%s/labels.npy"%dir_)
n_classes = len(np.unique(labels))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = CrossSourceModelGRLv10(input_channel_first=first_data.shape[1], input_channel_second=second_data.shape[1], f_encoder=first_enc, s_encoder=second_enc)

model = model.to(device)


tot_f1_f = []
tot_accuracy_f = []
tot_f1_s = []
tot_accuracy_s = []
for i in range(5):
    model_weights_fileName_f = folder_name_f+"/%d.pth"%i
    model_weights_fileName_s = folder_name_s+"/%d.pth"%i
    print("model_weights_fileName %s"%model_weights_fileName_f)
    if not os.path.exists(model_weights_fileName_f):
        continue

    test_idx = np.load("%s/test_idx_%d.npy"%(dir_,i))

    test_f_data = first_data[test_idx]
    test_s_data = second_data[test_idx]
    test_labels = labels[test_idx]

    f1_test_f, acc_test_f = getPerf(model, model_weights_fileName_f, test_f_data, test_labels, first=True)
    print("F1 %.2f and ACC %.2f"%(f1_test_f*100, acc_test_f*100))
    tot_f1_f.append(f1_test_f)
    tot_accuracy_f.append(acc_test_f)

    f1_test_s, acc_test_s = getPerf(model, model_weights_fileName_s, test_s_data, test_labels, first=False)
    print("F1 %.2f and ACC %.2f"%(f1_test_s*100, acc_test_s*100))
    tot_f1_s.append(f1_test_s)
    tot_accuracy_s.append(acc_test_s)


tot_f1_f = np.array(tot_f1_f)
tot_f1_s = np.array(tot_f1_s)
tot_accuracy_f = np.array(tot_accuracy_f)
tot_accuracy_s = np.array(tot_accuracy_s)

print("%.2f $\pm$ %.2f & %.2f $\pm$ %.2f"%(np.mean(tot_f1_f)*100, np.std(tot_f1_f)*100, np.mean(tot_accuracy_f)*100, np.std(tot_accuracy_f)*100 ))
print("%.2f $\pm$ %.2f & %.2f $\pm$ %.2f"%(np.mean(tot_f1_s)*100, np.std(tot_f1_s)*100, np.mean(tot_accuracy_s)*100, np.std(tot_accuracy_s)*100 ))

