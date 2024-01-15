import torch
import argparse

import os
import numpy as np


from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import scipy.io as sio
import matplotlib.pyplot as plt
import warnings





# GPU Configuration
gpu_id = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)





# LR_path = '/DataCommon/jwlee/AAL_116_RELATIONAL/LR/*'
# RL_path = '/DataCommon/jwlee/AAL_116_RELATIONAL/RL/*'
#
#
# for i in sorted( glob(LR_path) ):
#
#     a = sio.loadmat(i)
#     a = a['ROI']
#     a = np.expand_dims(a, axis=0)
#
#     name = i.split('/')[-1]
#
#     if name == 'AAL_116_150423.mat' or name == 'AAL_116_929464.mat':
#         continue
#
#     if name == 'AAL_116_100206.mat':
#         tfMRI_RELATIONAL_LR = a
#
#
#     else:
#         print(name)
#         tfMRI_RELATIONAL_LR = np.concatenate((tfMRI_RELATIONAL_LR, a))
#
#
#
#
# for j in sorted(glob(RL_path)):
#
#     b = sio.loadmat(j)
#     b = b['ROI']
#     b = np.expand_dims(b, axis=0)
#
#     name = j.split('/')[-1]
#
#     if name == 'AAL_116_100206.mat':
#         tfMRI_RELATIONAL_RL = b
#
#     else:
#         print(name)
#         tfMRI_RELATIONAL_RL = np.concatenate((tfMRI_RELATIONAL_RL, b))
#
#
# np.savez_compressed('/DataCommon/jwlee/RELATIONAL_LR/hcp_relational', tfMRI_RELATIONAL_LR=tfMRI_RELATIONAL_LR, tfMRI_RELATIONAL_RL=tfMRI_RELATIONAL_RL)

##################################################################################################################################
''' only task block, non-fixation block  LR + RL  '''

# data = np.load('/DataCommon/jwlee/RELATIONAL_LR/hcp_relational.npz')
# samples = data['tfMRI_RELATIONAL_LR']
#
# samples_RL = data['tfMRI_RELATIONAL_RL']
#
#
#
#
#
#
#
# list_atlas = [37, 85, 110, 158, 184]
#
# arr = samples[:, 11:11+22, :]
#
# for i in list_atlas:
#     b = samples[:, i:i+22 ,:]
#     arr = np.concatenate((arr, b), axis=1)
#
# cluster_2_hcp_relational_LR = arr   # (1040, 132, 116)
#
#
#
# arr_RL = samples_RL[:, 11:11+22, :]
#
# for j in list_atlas:
#     r = samples_RL[:, j:j+22, :]
#     arr_RL = np.concatenate((arr_RL, r), axis=1)
#
# cluster_2_hcp_relational_RL = arr_RL   # (1080, 132, 116)
#
#
# np.savez_compressed('/DataCommon/jwlee/RELATIONAL_LR/cluster_2_hcp_relational', tfMRI_RELATIONAL_LR=cluster_2_hcp_relational_LR, tfMRI_RELATIONAL_RL=cluster_2_hcp_relational_RL)


##################################################################################################################################
''' only task block, non-fixation block  LR + RL  FC flatten vector'''

import numpy as np
import scipy.io as sio
from os.path import join
from nilearn.connectome import ConnectivityMeasure
from glob import glob
from numpy import zeros, arange, corrcoef


def connectivity(input, type='correlation', vectorization=True, fisher_t=False):
    '''kind{“covariance”, “correlation”, “partial correlation”, “tangent”, “precision”}, optional'''
    measure = ConnectivityMeasure(kind=type, vectorize=vectorization, discard_diagonal=True)
    mt = measure.fit_transform(input)

    if fisher_t == True:
        for i in range(len(mt)):
            mt[i][:] = np.arctanh(mt[i][:])
    return mt


def corr(series, size):
    slide = size # window size = 3, 3 of 176

    idx = arange(0, series.shape[0], slide) # (0, 132, 22) 0부터 132까지 22 단위씩
    corr_mats = zeros((idx.shape[0], 6670)) # (6, 28)


    for w in range(idx.shape[0]):
        p = np.expand_dims(series[0 + idx[w]: size + idx[w],:], 0) # (1, 22, 116) = (1, t, 116)
        # corr_mats[w, :, :] = connectivity(p, type='correlation', vectorization=True, fisher_t=False)
        corr_mats[w, :] = connectivity(p, type='correlation', vectorization=True, fisher_t=False)

    return corr_mats, idx


data = np.load('/DataCommon/jwlee/RELATIONAL_LR/cluster_2_hcp_relational.npz')
sample = data['tfMRI_RELATIONAL_LR'] # (1040, 132, 116)


for i in range(sample.shape[0]):
    if i == 0:
        corr_matx, idx = corr(sample[i], 22) # corr_matx (6, 6670)
        corr_matx_F = np.expand_dims(corr_matx, 0)
        print(corr_matx_F.shape) # (1, 6, 6670)
    else:
        corr_matx, idx = corr(sample[i], 22)
        corr_matx_P = np.expand_dims(corr_matx, 0)
        corr_matx_F = np.concatenate((corr_matx_F, corr_matx_P), axis=0)

print(corr_matx_F.shape)

np.savez_compressed
# np.savez_compressed('/DataCommon/jwlee/RELATIONAL_LR/cluster_2_hcp_relational_FC', tfMRI_RELATIONAL_LR=corr_matx_F)






##################################################################################################################################


##################################################################################################################################
''' original data all timepoints and all areas '''

# data = np.load('/DataCommon/jwlee/EMOTION_LR/hcp_emotion.npz')
# samples = data['tfMRI_EMOTION_LR']
# samples = samples[:1041]  # (1080, 284, 116)
#
# samples_RL = data['tfMRI_EMOTION_RL']
# samples_RL = samples_RL[:1041]



# real_label = data['label_EMOTION_LR']
# real_label = real_label[:1041]
#
# real_label_RL = data['label_EMOTION_RL']
# real_label_RL = real_label[:1041]


##################################################################################################################################################################################


''' emotion label LR + RL '''

## LR ##
# for i in range(1080):
#
#     label_list_RL = [0 for i in range(150)]
#     shape_index_RL = [0, 50, 100]
#     face_index_RL = [25, 75, 125]
#
#     for id in shape_index_RL:
#         label_list_RL[id: id + 25] = [1 for i in range(25)]
#
#     for id in face_index_RL:
#         label_list_RL[id: id + 25] = [2 for i in range(25)]
#
#     label_list_RL = np.array(label_list_RL)
#     label_list_RL = np.expand_dims(label_list_RL, axis=0)
#
#     if i == 0:
#         label_list_EMOTION_RL = label_list_RL
#
#     else:
#         label_list_EMOTION_RL = np.concatenate((label_list_EMOTION_RL, label_list_RL))
#
# cluster_2_emotion_label_RL = label_list_EMOTION_RL  # (1080, 136)
#
# ## RL ##
#
# for i in range(1080):
#
#     label_list = [0 for i in range(150)]
#     shape_index = [0, 50, 100]
#     face_index = [25, 75, 125]
#
#     for id in shape_index:
#         label_list[id: id + 25] = [1 for i in range(25)]
#
#     for id in face_index:
#         label_list[id: id + 25] = [2 for i in range(25)]
#
#     label_list = np.array(label_list)
#     label_list = np.expand_dims(label_list, axis=0)
#
#     if i == 0:
#         label_list_LR = label_list
#
#     else:
#         label_list_LR = np.concatenate((label_list_LR, label_list))
#
# cluster_2_emotion_label_LR = label_list_LR  # (1080, 136)
#
#
#
# np.savez_compressed('/DataCommon/jwlee/EMOTION_LR/cluster_2_hcp_emotion_label', label_list_LR=cluster_2_emotion_label_LR, label_list_RL=cluster_2_emotion_label_RL)


###########################################################################################################################################################################
