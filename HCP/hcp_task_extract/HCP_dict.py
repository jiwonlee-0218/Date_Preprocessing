import nibabel as nib
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
import scipy.io as sio
from glob import glob
import os
import os.path
import numpy as np

'''
save_path = '/media/12T/practice/LR_DICT/emotion'


LR_path = '/mnt/hcp/homes/jwlee/HCP_train_data/tfMRI_EMOTION_LR/AAL/*'
RL_path = '/mnt/hcp/homes/jwlee/HCP_train_data/tfMRI_EMOTION_RL/AAL/*'

list_a = []
list_b = []

for i in sorted(glob(LR_path)):

    a = sio.loadmat(i)
    a = a['ROI']
    a = np.expand_dims(a, axis=0)

    name = i.split('/')[-1]

    if name == '136126.mat':
        continue

    if name == '100206.mat':
        tfMRI_EMOTION_LR = a

        list_a.append(name.split('.')[0])

    else:
        print(name)
        tfMRI_EMOTION_LR = np.concatenate((tfMRI_EMOTION_LR, a))
        list_a.append(name.split('.')[0])

tfMRI_EMOTION_LR_label = np.array(list_a)




for j in sorted(glob(RL_path)):

    b = sio.loadmat(j)
    b = b['ROI']
    b = np.expand_dims(b, axis=0)

    name = j.split('/')[-1]



    if name == '100206.mat':
        tfMRI_EMOTION_RL = b

        list_b.append(name.split('.')[0])

    else:
        print(name)
        tfMRI_EMOTION_RL = np.concatenate((tfMRI_EMOTION_RL, b))
        list_b.append(name.split('.')[0])

tfMRI_EMOTION_RL_label = np.array(list_b)



print(tfMRI_EMOTION_LR.shape)
print(tfMRI_EMOTION_RL.shape)

data = np.load('/media/12T/practice/LR_DICT/emotion/hcp_emotion_label.npz')
label_EMOTION_LR = data['label_list_LR']
label_EMOTION_RL = data['label_list_RL']

np.savez_compressed('/media/12T/practice/LR_DICT/hcp_emotion', tfMRI_EMOTION_LR=tfMRI_EMOTION_LR, tfMRI_EMOTION_LR_id=tfMRI_EMOTION_LR_label, tfMRI_EMOTION_RL=tfMRI_EMOTION_RL, tfMRI_EMOTION_RL_id=tfMRI_EMOTION_RL_label, label_EMOTION_LR=label_EMOTION_LR, label_EMOTION_RL=label_EMOTION_RL)


'''















save_path = '/media/12T/practice/LR_DICT'


LR_path = '/media/12T/practice/AAL_116_WM/LR/*'
RL_path = '/media/12T/practice/AAL_116_WM/RL/*'

list_a = []
list_b = []

for i in sorted(glob(LR_path)):
    try:

        a = sio.loadmat(i)
        a = a['ROI']
        a = np.expand_dims(a, axis=0)

        name = i.split('/')[-1]


        if name == '100206.mat':
            tfMRI_WM_LR = a

            list_a.append(name.split('.')[0])

        else:
            print(name)
            tfMRI_WM_LR = np.concatenate((tfMRI_WM_LR, a))
            list_a.append(name.split('.')[0])

    except:
        print(name, a.shape)
        continue


tfMRI_WM_LR_label = np.array(list_a)




for j in sorted(glob(RL_path)):
    try:

        b = sio.loadmat(j)
        b = b['ROI']
        b = np.expand_dims(b, axis=0)

        name = j.split('/')[-1]


        if name == '100206.mat':
            tfMRI_WM_RL = b

            list_b.append(name.split('.')[0])

        else:
            print(name)
            tfMRI_WM_RL = np.concatenate((tfMRI_WM_RL, b))
            list_b.append(name.split('.')[0])

    except:
        continue

tfMRI_WM_RL_label = np.array(list_b)



print(tfMRI_WM_LR.shape)
print(tfMRI_WM_RL.shape)


np.savez_compressed('/media/12T/practice/LR_DICT/hcp_wm', tfMRI_WM_LR=tfMRI_WM_LR, tfMRI_WM_LR_id=tfMRI_WM_LR_label, tfMRI_WM_RL=tfMRI_WM_RL, tfMRI_WM_RL_id=tfMRI_WM_RL_label)

