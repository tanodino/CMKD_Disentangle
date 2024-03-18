import numpy as np
from sklearn.utils import shuffle

def rescale(data):
    min_ = np.percentile(data, 2)
    max_ = np.percentile(data, 98)
    return np.clip( (data - min_) / (max_ - min_), 0, 1.)

def getIdxVal(sub_hashCl2idx, val):
    idx = []
    for k in sub_hashCl2idx.keys():
        temp = sub_hashCl2idx[k]
        idx.append(temp[0:val])
    return np.concatenate(idx, axis=0)


def get_idxPerClass(hashCl2idx, max_val):
    sub_hashCl2idx = {}
    for k in hashCl2idx.keys():
        temp = hashCl2idx[k]
        temp = shuffle(temp)
        sub_hashCl2idx[k] = temp[0:max_val]
    return sub_hashCl2idx


def extractWriteTrainIdx(nrepeat, nsample_list, hashCl2idx, prefix):
    max_val = nsample_list[-1]
    for i in range(nrepeat):
        sub_hashCl2idx = get_idxPerClass(hashCl2idx, max_val)
        for val in nsample_list:
            idx = getIdxVal(sub_hashCl2idx, val)
            np.save("%s_%d_%d_train_idx.npy"%(prefix,i,val), idx) 


def getHash2classes(labels):
    hashCl2idx = {}
    for v in np.unique(labels):
        idx = np.where(labels == v)[0]
        idx = shuffle(idx)
        hashCl2idx[v] = idx
    return hashCl2idx
        
def writeFilteredData(prefix, data):
    np.save("%s_data_normalized.npy"%prefix,data)
    np.save("%s_label_normalized.npy"%prefix,label)


prefix_path = "../SSHDA/EuroSAT_OPT_SAR/"
#READ DATA
data_sar = np.load(prefix_path+"SAR_data.npy")
data_opt = np.load(prefix_path+"MS_data.npy").astype("float32")

label = np.load(prefix_path+"labels.npy")

#RESCALE DATA BETWEEN 0 and 1 per band
for i in range(data_sar.shape[-1]):
    data_sar[:,:,:,i] = rescale(data_sar[:,:,:,i])

for i in range(data_opt.shape[-1]):
    data_opt[:,:,:,i] = rescale(data_opt[:,:,:,i])

#SWAP DIMENSION : [N,H,W,C] -> [N,C,H,W]
data_sar = np.moveaxis(data_sar, (0,1,2,3), (0,2,3,1))
data_opt = np.moveaxis(data_opt, (0,1,2,3), (0,2,3,1))

np.save("MS_data_normalized.npy",data_opt)
np.save("SAR_data_normalized.npy",data_sar)
np.save("labels.npy",label)


