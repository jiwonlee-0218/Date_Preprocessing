import numpy as np
import torch
import numpy as np
import keras
from keras.layers import *
from keras.models import *
from keras.optimizers import *
import keras.backend as K

from scipy import stats
from sklearn.model_selection import StratifiedKFold





if __name__ == "__main__":
    tf_data, tf_label, tf_idx = torch.load('/DataCommon/jwlee/data/hcp-task_roi-aal.pth')
    del tf_data[2096]
    del tf_label[2096]

    all_list = []
    for i in range(1043):
        a = np.array([i, 0])
        all_list.append(a)

    for i in range(1084):
        a = np.array([1043 + i, 1])
        all_list.append(a)

    for i in range(1047):
        a = np.array([2127 + i, 2])
        all_list.append(a)

    for i in range(1080):
        a = np.array([3174 + i, 3])
        all_list.append(a)

    for i in range(1040):
        a = np.array([4254 + i, 4])
        all_list.append(a)

    for i in range(1049):
        a = np.array([5294 + i, 5])
        all_list.append(a)

    for i in range(1084):
        a = np.array([6343 + i, 6])
        all_list.append(a)

    demo = np.array(all_list)


    L = 176
    S = 0
    data = np.zeros((demo.shape[0], 1, L, 116, 1))
    label = np.zeros((demo.shape[0],))

    # load all data
    idx = 0
    data_all = None

    for i in range(demo.shape[0]):

        full_sequence = tf_data[i].T

        if full_sequence.shape[1] < S + L:
            continue

        full_sequence = full_sequence[:, S:S + L];
        z_sequence = stats.zscore(full_sequence, axis=1)

        if data_all is None:
            data_all = z_sequence
        else:
            data_all = np.concatenate((data_all, z_sequence), axis=1)

        data[idx, 0, :, :, 0] = np.transpose(z_sequence)
        label[idx] = demo[i, 1]
        idx = idx + 1
        print(i)

    print('-----------------------------------------------------------------------------------------')
    # compute adj matrix
    n_regions = 116
    A = np.zeros((n_regions, n_regions))
    for i in range(n_regions):
        print(i)
        for j in range(i, n_regions):
            if i == j:
                A[i][j] = 1
            else:
                A[i][j] = abs(np.corrcoef(data_all[i, :], data_all[j, :])[0][1])  # get value from corrcoef matrix
                A[j][i] = A[i][j]

    np.save('/home/jwlee/HMM/ST_GCN/data/task_fMRI/adj_matrix.npy', A)

    # split train/test and save data

    data = data[:idx]
    label = label[:idx]
    print(data.shape)
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    fold = 1
    for train_idx, test_idx in skf.split(data, label):
        train_data = data[train_idx]
        train_label = label[train_idx]
        test_data = data[test_idx]
        test_label = label[test_idx]

        filename = '/home/jwlee/HMM/ST_GCN/data/task_fMRI/train_data_' + str(fold) + '.npy'
        np.save(filename, train_data)
        filename = '/home/jwlee/HMM/ST_GCN/data/task_fMRI/train_label_' + str(fold) + '.npy'
        np.save(filename, train_label)
        filename = '/home/jwlee/HMM/ST_GCN/data/task_fMRI/test_data_' + str(fold) + '.npy'
        np.save(filename, test_data)
        filename = '/home/jwlee/HMM/ST_GCN/data/task_fMRI/test_label_' + str(fold) + '.npy'
        np.save(filename, test_label)

        fold = fold + 1