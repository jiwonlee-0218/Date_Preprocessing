# HCP task fMRI .mat file들을 concatenate해서 .npz파일로 압축 (2)

import nibabel as nib
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
import scipy.io as sio
from glob import glob
import os
import os.path
import numpy as np


save_path = '/media/12T/practice/LR_DICT'


LR_path = '/mnt/hcp/homes/jwlee/HCP_train_data/tfMRI_MOTOR_LR/4d/ho/*'
RL_path = '/mnt/hcp/homes/jwlee/HCP_train_data/tfMRI_MOTOR_RL/4d/ho/*'

list_a = []
list_b = []

for i in sorted(glob(LR_path)):

    print(i)
    a = sio.loadmat(i)
    a = a['ROI']
    a = np.expand_dims(a, axis=0)

    name = i.split('/')[-1]

    if name == '100206.mat':
        tfMRI_MOTOR_LR = a

        list_a.append(name.split('.')[0])

    else:
        print(name)
        tfMRI_MOTOR_LR = np.concatenate((tfMRI_MOTOR_LR, a))
        list_a.append(name.split('.')[0])

tfMRI_MOTOR_LR_label = np.array(list_a)


for j in sorted(glob(RL_path)):

    b = sio.loadmat(j)
    b = b['ROI']
    b = np.expand_dims(b, axis=0)

    name = j.split('/')[-1]

    if name == '689470.mat':
        continue

    elif name == '100206.mat':
        tfMRI_MOTOR_RL = b

        list_b.append(name.split('.')[0])

    else:
        print(name)
        tfMRI_MOTOR_RL = np.concatenate((tfMRI_MOTOR_RL, b))
        list_b.append(name.split('.')[0])

tfMRI_MOTOR_RL_label = np.array(list_b)



print(tfMRI_MOTOR_LR.shape)
print(tfMRI_MOTOR_RL.shape)

data = np.load('/media/12T/practice/LR_DICT/hcp_motor_ho_label.npz')
ho_label_MOTOR_LR = data['label_list_LR']
ho_label_MOTOR_RL = data['label_list_RL']

np.savez_compressed('/media/12T/practice/LR_DICT/ho_hcp_motor', ho_tfMRI_MOTOR_LR=tfMRI_MOTOR_LR, ho_tfMRI_MOTOR_LR_id=tfMRI_MOTOR_LR_label, ho_tfMRI_MOTOR_RL=tfMRI_MOTOR_RL, ho_tfMRI_MOTOR_RL_id=tfMRI_MOTOR_RL_label, ho_label_MOTOR_LR=ho_label_MOTOR_LR, ho_label_MOTOR_RL=ho_label_MOTOR_RL)




