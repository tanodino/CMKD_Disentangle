import torch
import sys
import numpy as np
import sys
from torch.utils.data import TensorDataset, DataLoader
#from model_transformer import TransformerEncoder
from model_pytorch import ModelHYPER, MultiSourceModel
from sklearn.metrics import f1_score, accuracy_score
from functions import TRAIN_BATCH_SIZE, LEARNING_RATE, EPOCHS, WARM_UP_EPOCH_EMA, cumulate_EMA, MOMENTUM_EMA, transform, MyDataset, hashPREFIX2SOURCE
import os


def evaluation(model, dataloader, device, dir_):
    model.eval()
    tot_pred = []
    tot_labels = []
    if dir_ == "PAVIA_UNIVERSITY":
        for x_batch_f, y_batch in dataloader:
            x_batch_f = x_batch_f.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch_f)
            pred_npy = np.argmax(pred.cpu().detach().numpy(), axis=1)
            tot_pred.append( pred_npy )
            tot_labels.append( y_batch.cpu().detach().numpy())

    #elif dir_ == "EUROSAT" or dir_ == "SUNRGBD" or dir_ == "HANDS" or dir_ == "AV-MNIST" or dir_ == "TRISTAR":
    else:
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


def createDataLoader(first_data, second_data, y, tobeshuffled, BATCH_SIZE):
    #DATALOADER TRAIN
    first_tensor = torch.tensor(first_data, dtype=torch.float32)
    second_tensor = None
    dataset = None
    y_tensor = torch.tensor(y, dtype=torch.int64)
    if second_data is not None:
        second_tensor = torch.tensor(second_data, dtype=torch.float32)
        dataset = TensorDataset(first_tensor, second_tensor, y_tensor)
    else:
        dataset = TensorDataset(first_tensor, y_tensor)
    
    dataloader = DataLoader(dataset, shuffle=tobeshuffled, batch_size=BATCH_SIZE)
    return dataloader

#python restore_teacher.py PAVIA_UNIVERSITY FULL FULL FULL
#python restore_teacher.py SUNRGBD RGB DEPTH SUM
#python restore_teacher.py SUNRGBD RGB DEPTH CONCAT
#python restore_teacher.py EUROSAT MS SAR SUM
#python restore_teacher.py SUNRGB MS SAR CONCAT


dir_ = sys.argv[1]
first_prefix = sys.argv[2]
second_prefix = sys.argv[3]
fusion_type = sys.argv[4]

folder_name = None
folder_name = dir_+"/TEACHER_%s"%fusion_type

second_data = None

first_data = np.load("%s/%s_data_normalized.npy"%(dir_,first_prefix))
second_data = np.load("%s/%s_data_normalized.npy"%(dir_,second_prefix))
labels = np.load("%s/labels.npy"%dir_)
n_classes = len(np.unique(labels))

#if dir_ != "PAVIA_UNIVERSITY":
#    second_data = np.load("%s/%s_data_normalized.npy"%(dir_,second_prefix))
first_enc = hashPREFIX2SOURCE[first_prefix]
second_enc = hashPREFIX2SOURCE[second_prefix]



device = 'cuda' if torch.cuda.is_available() else 'cpu'

#if dir_ != "PAVIA_UNIVERSITY":
model = MultiSourceModel(input_channel_first=first_data.shape[1], input_channel_second=second_data.shape[1], f_encoder=first_enc, s_encoder=second_enc, fusion_type=fusion_type, num_classes=n_classes)
#else:
#    model = ModelHYPER(num_classes=n_classes)
model = model.to(device)


tot_f1 = []
tot_accuracy = []
for i in range(10):
    print("ITERATION %d"%i)
    model_weights_fileName = folder_name+"/%d.pth"%i
    print("model_weights_fileName %s"%model_weights_fileName)
    if not os.path.exists(model_weights_fileName):
        continue

    test_idx = np.load("%s/test_idx_%d.npy"%(dir_,i))
    test_s_data = None

    if dir_ != "PAVIA_UNIVERSITY":
        test_s_data = second_data[test_idx]

    test_f_data = first_data[test_idx]
    test_labels = labels[test_idx]

    print(test_f_data.shape)
    print(test_labels.shape)
    #DATALOADER TEST
    dataloader_test = createDataLoader(test_f_data, test_s_data, test_labels, False, 512)
    print(len(dataloader_test))

    model.load_state_dict(torch.load(model_weights_fileName))

    pred_test, labels_test = evaluation(model, dataloader_test, device, dir_)
    f1_test = f1_score(labels_test, pred_test, average="weighted")
    acc_test = accuracy_score(labels_test, pred_test)
    print(f1_test)
    tot_f1.append(f1_test)
    tot_accuracy.append(acc_test)

tot_f1 = np.array(tot_f1)
tot_accuracy = np.array(tot_accuracy)

print("%.2f $\pm$ %.2f & %.2f $\pm$ %.2f"%(np.mean(tot_f1)*100, np.std(tot_f1)*100, np.mean(tot_accuracy)*100, np.std(tot_accuracy)*100 ))

