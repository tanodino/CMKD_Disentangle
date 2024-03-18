from sklearn.utils import shuffle
import sys
import numpy as np


def createSplit(labels, train_perc, valid_perc, rs):
    train_idx = []
    val_idx = []
    test_idx = []
    for el in np.unique(labels):
        idx = np.where(labels == el)[0]
        idx = shuffle(idx, random_state=rs)
        train_limit = int(len(idx)*train_perc)
        val_limit = int(len(idx)*(train_perc + valid_perc) )
        train_idx.append(  idx[0:train_limit]  )
        val_idx.append(  idx[train_limit:val_limit]  )
        test_idx.append(  idx[val_limit::]  )
    return np.concatenate(train_idx), np.concatenate(val_idx), np.concatenate(test_idx)



dir_ = sys.argv[1]

labels = np.load("%s/labels.npy"%dir_)
train_perc = 0.7
valid_perc = 0.1

for i in range(10):
    train_idx, val_idx, test_idx = createSplit(labels, train_perc, valid_perc, rs=i*1000)
    np.save("%s/train_idx_%d.npy"%(dir_,i), train_idx)
    np.save("%s/valid_idx_%d.npy"%(dir_,i), val_idx)
    np.save("%s/test_idx_%d.npy"%(dir_,i), test_idx)
