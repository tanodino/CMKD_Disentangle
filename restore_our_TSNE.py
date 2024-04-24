import torch
import sys
import numpy as np
import sys
from torch.utils.data import TensorDataset, DataLoader
from model_pytorch import CrossSourceModelGRLv10
from sklearn.metrics import f1_score, accuracy_score
from functions import hashPREFIX2SOURCE
import os
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import MinMaxScaler

def plotEmb(emb, test_labels, dom_labels, outFileName):

    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'cyan', 'orange', 'magenta', 'teal', 'maroon']
    markers = ["D", "*"]
    colorMapping = [colors[i.astype("int")] for i in test_labels]
    domainMapping = [markers[el.astype("int")] for el in dom_labels]
    scaler = MinMaxScaler()
    emb = scaler.fit_transform(emb)

    #pca = PCA(n_components=50)
    #pca_result = pca.fit_transform(emb)

    #scaler = MinMaxScaler()
    #pca_result = scaler.fit_transform(pca_result)


    #X_embedded = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=30).fit_transform(pca_result)
    X_embedded = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=3).fit_transform(emb)
    half = X_embedded.shape[0] // 2
    
    plt.scatter(X_embedded[0:half,0], X_embedded[0:half,1], c = colorMapping[0:half], marker= "D")#, s=area, c=colors, alpha=0.5)
    plt.scatter(X_embedded[half::,0], X_embedded[half::,1], c = colorMapping[half::], marker= "*")#, s=area, c=colors, alpha=0.5)
    
    plt.savefig(outFileName+".png")
    plt.clf()

def selectRandomSamples(first_data, second_data, labels, n_samples):
    new_labels = []
    new_first_data = []
    new_second_data = []
    for l in np.unique(labels):
        idx = np.where(labels == l)[0]
        idx = shuffle(idx)
        idx = idx[0:n_samples]
        new_labels.append( np.ones( len(idx))*l  )
        new_first_data.append(first_data[idx])
        new_second_data.append(second_data[idx])
    return np.concatenate(new_first_data), np.concatenate(new_second_data), np.concatenate(new_labels)


def getEmbeddings(model, model_weights_fileName, test_data, test_labels, first=True):
    model.load_state_dict(torch.load(model_weights_fileName))
    dataloader_test = createDataLoader(test_data, test_labels, False, 32)
    emb_inv, emb_task, emb_irrelevant = getEmbs(model, dataloader_test, device, first=first)
    return emb_inv, emb_task, emb_irrelevant



def getEmbs(model, dataloader, device, first=True):
    model.eval()
    emb_inv = []
    emb_task = []
    emb_irrelevant = []
    for x_batch, _ in dataloader:
        x_batch = x_batch.to(device)
        temp_emb_inv = None
        temp_emb_task = None
        temp_emb_irrelevant = None
        if first:
            temp_emb_inv, temp_emb_task, temp_emb_irrelevant = model.getEmb_first(x_batch)
        else:
            temp_emb_inv, temp_emb_task, temp_emb_irrelevant = model.getEmb_second(x_batch)
        emb_inv.append( temp_emb_inv.cpu().detach().numpy() )
        emb_task.append( temp_emb_task.cpu().detach().numpy() )
        emb_irrelevant.append( temp_emb_irrelevant.cpu().detach().numpy() )

    return np.concatenate(emb_inv), np.concatenate(emb_task), np.concatenate(emb_irrelevant), 


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
second_prefix = sys.argv[3]
method = sys.argv[4]

folder_name_f = dir_+"/OUR-v10_%s_%s"%(first_prefix, method)
folder_name_s = dir_+"/OUR-v10_%s_%s"%(second_prefix, method)

first_enc = hashPREFIX2SOURCE[first_prefix]
second_enc = hashPREFIX2SOURCE[second_prefix]

first_data = np.load("%s/%s_data_normalized.npy"%(dir_,first_prefix))
second_data = np.load("%s/%s_data_normalized.npy"%(dir_,second_prefix))
labels = np.load("%s/labels.npy"%dir_)

n_samples = 50



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

    test_f_data, test_s_data, test_labels = selectRandomSamples(test_f_data, test_s_data, test_labels, n_samples)
    
    f_emb_inv, f_emb_task, f_emb_irrelevant = getEmbeddings(model, model_weights_fileName_f, test_f_data, test_labels, first=True)
    s_emb_inv, s_emb_task, s_emb_irrelevant = getEmbeddings(model, model_weights_fileName_s, test_s_data, test_labels, first=False)

    emb_inv = np.concatenate([f_emb_inv,s_emb_inv])
    emb_task = np.concatenate([f_emb_task,s_emb_task])
    emb_irrelevant = np.concatenate([f_emb_irrelevant,s_emb_irrelevant])
    extended_test_labels = np.concatenate([test_labels, test_labels])
    dom_labels = np.concatenate([np.zeros(test_labels.shape[0]), np.ones(test_labels.shape[0])])

    plotEmb(emb_inv, extended_test_labels, dom_labels, "%s_%s_inv_%d"%(dir_,first_prefix, i) )
    plotEmb(emb_task, extended_test_labels, dom_labels, "%s_%s_domDiscr_%d"%(dir_,first_prefix, i))
    plotEmb(emb_irrelevant, extended_test_labels, dom_labels, "%s_%s_domIrrelevant_%d"%(dir_,first_prefix, i))



    '''
    plotEmb(emb_inv, test_labels, "%s_%s_inv_%d"%(dir_,second_prefix, i) )
    plotEmb(emb_task, test_labels, "%s_%s_domDiscr_%d"%(dir_,second_prefix, i))
    plotEmb(emb_irrelevant, test_labels, "%s_%s_domIrrelevant_%d"%(dir_,second_prefix, i))
    '''
    exit()