import torch
import sys
import numpy as np
import sys
from torch.utils.data import TensorDataset, DataLoader
#from model_transformer import TransformerEncoder
from model_pytorch import CrossSourceModel
from sklearn.metrics import f1_score, accuracy_score
from functions import TRAIN_BATCH_SIZE, LEARNING_RATE, EPOCHS, WARM_UP_EPOCH_EMA, cumulate_EMA, MOMENTUM_EMA, transform, MyDataset
import os


def evaluation(model, dataloader, device):
    model.eval()
    tot_pred = []
    tot_labels = []
    for x_batch_f, y_batch in dataloader:
        x_batch_f = x_batch_f.to(device)
        y_batch = y_batch.to(device)
        pred = model.pred_firstEnc(x_batch_f)
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

folder_name = None
folder_name = dir_name = dir_+"/OUR_%s_%s"%(first_prefix, method)

first_enc = None
second_enc = None
if dir_ == "PAVIA_UNIVERSITY":
    first_enc = 'hyper'
    second_enc = 'hyper'
elif dir_ == 'SUNRGBD' or dir_ == 'EUROSAT':
    first_enc = 'image'
    second_enc = 'image'

first_data = np.load("%s/%s_data_normalized.npy"%(dir_,first_prefix))
second_data = np.load("%s/%s_data_normalized.npy"%(dir_,seocnd_prefix))
labels = np.load("%s/labels.npy"%dir_)
n_classes = len(np.unique(labels))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = model = CrossSourceModel(input_channel_first=first_data.shape[1], input_channel_second=second_data.shape[1], f_encoder=first_enc, s_encoder=second_enc)

model = model.to(device)


tot_f1 = []
tot_accuracy = []
for i in range(10):
    model_weights_fileName = folder_name+"/%d.pth"%i
    print("model_weights_fileName %s"%model_weights_fileName)
    if not os.path.exists(model_weights_fileName):
        continue

    test_idx = np.load("%s/test_idx_%d.npy"%(dir_,i))

    test_f_data = first_data[test_idx]
    test_labels = labels[test_idx]

    model.load_state_dict(torch.load(model_weights_fileName))

    print("test_f_data ",test_f_data.shape)
    print("test_labels ",test_labels.shape)

    dataloader_test = createDataLoader(test_f_data, test_labels, False, 512)

    pred_test, labels_test = evaluation(model, dataloader_test, device)
    f1_test = f1_score(labels_test, pred_test, average="weighted")
    acc_test = accuracy_score(labels_test, pred_test)
    tot_f1.append(f1_test)
    tot_accuracy.append(acc_test)

tot_f1 = np.array(tot_f1)
tot_accuracy = np.array(tot_accuracy)

print("%.2f $\pm$ %.2f & %.2f $\pm$ %.2f"%(np.mean(tot_f1)*100, np.std(tot_f1)*100, np.mean(tot_accuracy)*100, np.std(tot_accuracy)*100 ))

