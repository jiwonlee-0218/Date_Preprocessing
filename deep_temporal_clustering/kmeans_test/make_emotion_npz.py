import torch
import argparse

import os
import numpy as np


from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import glob
import matplotlib.pyplot as plt
import warnings





# GPU Configuration
gpu_id = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


##################################################################################################
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

####################################################

''' emotion data LR + RL '''

# list_atlas = [34, 63, 92, 121, 151]
#
# arr = samples[:, 4:4+25, :]
#
# for i in list_atlas:
#     b = samples[:, i:i+25 ,:]
#     arr = np.concatenate((arr, b), axis=1)
#
# cluster_2_hcp_emotion_LR = arr   # (1080, 284, 10)
#
#
#
# arr_RL = samples_RL[:, 4:4+25, :]
#
# for j in list_atlas:
#     r = samples_RL[:, j:j+25 ,:]
#     arr_RL = np.concatenate((arr_RL, r), axis=1)
#
# cluster_2_hcp_emotion_RL = arr_RL   # (1080, 136, 10)
#
# np.savez_compressed('/DataCommon/jwlee/EMOTION_LR/cluster_2_hcp_emotion', tfMRI_EMOTION_LR=cluster_2_hcp_emotion_LR, tfMRI_EMOTION_RL=cluster_2_hcp_emotion_RL)


##################################################################################################################################################################################


''' emotion label LR + RL '''

# ## LR ##
# for i in range(1080):
#
#     label_list_RL = [0 for i in range(150)]
#     shape_index_RL = [0, 50, 100]
#     face_index_RL = [25, 75, 125]
#
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

''' emotion label LR + RL  cluster 6 '''


## RL ##
for j in range(1041):

    label_list_RL = [0 for i in range(176)]
    shape_index_RL = [0, 50, 100]
    face_index_RL = [25, 75, 125]

    label_list_RL[0:4] = [0 for i in range(4)]
    label_list_RL[4:29] = [1 for i in range(25)]
    label_list_RL[29:34] = [0 for i in range(5)]
    label_list_RL[34:59] = [2 for i in range(25)]
    label_list_RL[59:63] = [0 for i in range(4)]
    label_list_RL[63:88] = [3 for i in range(25)]
    label_list_RL[88:92] = [0 for i in range(4)]
    label_list_RL[92:117] = [4 for i in range(25)]
    label_list_RL[117:121] = [0 for i in range(4)]
    label_list_RL[121:146] = [5 for i in range(25)]
    label_list_RL[146:151] = [0 for i in range(4)]
    label_list_RL[151:176] = [6 for i in range(25)]

    label_list_RL = np.array(label_list_RL)
    label_list_RL = np.expand_dims(label_list_RL, axis=0)

    if j == 0:
        label_list_EMOTION_RL = label_list_RL

    else:
        label_list_EMOTION_RL = np.concatenate((label_list_EMOTION_RL, label_list_RL))

cluster_7_emotion_label_RL = label_list_EMOTION_RL



## LR ##
for j in range(1041):

    label_list_LR = [0 for i in range(176)]
    shape_index = [0, 50, 100]
    face_index = [25, 75, 125]

    label_list_LR[0:4] = [0 for i in range(4)]
    label_list_LR[4:29] = [1 for i in range(25)]
    label_list_LR[29:34] = [0 for i in range(5)]
    label_list_LR[34:59] = [2 for i in range(25)]
    label_list_LR[59:63] = [0 for i in range(4)]
    label_list_LR[63:88] = [3 for i in range(25)]
    label_list_LR[88:92] = [0 for i in range(4)]
    label_list_LR[92:117] = [4 for i in range(25)]
    label_list_LR[117:121] = [0 for i in range(4)]
    label_list_LR[121:146] = [5 for i in range(25)]
    label_list_LR[146:151] = [0 for i in range(4)]
    label_list_LR[151:176] = [6 for i in range(25)]

    label_list_LR = np.array(label_list_LR)
    label_list_LR = np.expand_dims(label_list_LR, axis=0)

    if j == 0:
        label_list_EMOTION_LR = label_list_LR

    else:
        label_list_EMOTION_LR = np.concatenate((label_list_EMOTION_LR, label_list_LR))

cluster_7_emotion_label_LR = label_list_EMOTION_LR  # (1041, 150)


np.savez_compressed('/DataCommon/jwlee/EMOTION_LR/cluster_7_hcp_emotion_label', label_list_LR=cluster_7_emotion_label_LR, label_list_RL=cluster_7_emotion_label_RL)