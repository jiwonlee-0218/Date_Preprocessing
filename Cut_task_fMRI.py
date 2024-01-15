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

emotion_data = np.load('/DataCommon/jwlee/HCP_7_task_npz/hcp_emotion.npz')
emotion_samples_LR = emotion_data['tfMRI_EMOTION_LR']
emotion_samples_LR = emotion_samples_LR[:1041]  # (1041, 176, 116)

emotion_samples_RL = emotion_data['tfMRI_EMOTION_RL']
emotion_samples_RL = emotion_samples_RL[:1041] # (1041, 176, 116)




gambling_data = np.load('/DataCommon/jwlee/HCP_7_task_npz/hcp_gambling.npz')
gambling_samples_LR = gambling_data['tfMRI_GAMBLING_LR']
gambling_samples_LR = gambling_samples_LR[:1085]

gambling_samples_RL = gambling_data['tfMRI_GAMBLING_RL']
gambling_samples_RL = gambling_samples_RL[:1082]






language_data = np.load('/DataCommon/jwlee/HCP_7_task_npz/hcp_language.npz')
language_samples_LR = language_data['tfMRI_LANGUAGE_LR']
language_samples_LR = language_samples_LR[:1044]

language_samples_RL = language_data['tfMRI_LANGUAGE_RL']
language_samples_RL = language_samples_RL[:1049]





motor_data = np.load('/DataCommon/jwlee/HCP_7_task_npz/hcp_motor.npz')
motor_samples_LR = motor_data['tfMRI_MOTOR_LR']
motor_samples_LR = motor_samples_LR[:1080]  # (1080, 284, 116)

motor_samples_RL = motor_data['tfMRI_MOTOR_RL']
motor_samples_RL = motor_samples_RL[:1080]  # (1080, 284, 116)






relational_data = np.load('/DataCommon/jwlee/HCP_7_task_npz/hcp_relational.npz')
relational_samples_LR = relational_data['tfMRI_RELATIONAL_LR']
relational_samples_LR = relational_samples_LR[:1040]

relational_samples_RL = relational_data['tfMRI_RELATIONAL_RL']
relational_samples_RL = relational_samples_RL[:1040]








social_data = np.load('/DataCommon/jwlee/HCP_7_task_npz/hcp_social.npz')
social_samples_LR = social_data['tfMRI_SOCIAL_LR']
social_samples_LR = social_samples_LR[:1045]

social_samples_RL = social_data['tfMRI_SOCIAL_RL']
social_samples_RL = social_samples_RL[:1051]







wm_data = np.load('/DataCommon/jwlee/HCP_7_task_npz/hcp_wm.npz')
wm_samples_LR = wm_data['tfMRI_WM_LR']
wm_samples_LR = wm_samples_LR[:1082]

wm_samples_RL = wm_data['tfMRI_WM_RL']
wm_samples_RL = wm_samples_RL[:1085]






##############################################################################################################################

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

# np.savez_compressed('/DataCommon/jwlee/EMOTION_LR/cluster_7_hcp_emotion_label', label_list_LR=cluster_7_emotion_label_LR, label_list_RL=cluster_7_emotion_label_RL)


np.savez_compressed('/DataCommon/jwlee/HCP_7_task_npz/hcp_7tasks', tfMRI_EMOTION_LR=emotion_samples_LR, tfMRI_EMOTION_RL=emotion_samples_RL,
                                                                   tfMRI_GAMBLING_LR=gambling_samples_LR, tfMRI_GAMBLING_RL=gambling_samples_RL,
                                                                   tfMRI_LANGUAGE_LR=language_samples_LR, tfMRI_LANGUAGE_RL=language_samples_RL,
                                                                   tfMRI_MOTOR_LR=motor_samples_LR, tfMRI_MOTOR_RL=motor_samples_RL,
                                                                   tfMRI_RELATIONAL_LR=relational_samples_LR, tfMRI_RELATIONAL_RL=relational_samples_RL,
                                                                   tfMRI_SOCIAL_LR=social_samples_LR, tfMRI_SOCIAL_RL=social_samples_RL,
                                                                   tfMRI_WM_LR=wm_samples_LR, tfMRI_WM_RL=wm_samples_RL)
