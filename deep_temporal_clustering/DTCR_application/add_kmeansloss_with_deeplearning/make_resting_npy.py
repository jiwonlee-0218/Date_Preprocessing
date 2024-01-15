import scipy.io as sio
from glob import glob
import os
import os.path
import numpy as np


save_path = '/DataCommon/jwlee/RESTING_LR'


LR_path = '/DataCommon/jwlee/AAL_resting/LR/*'
RL_path = '/DataCommon/jwlee/AAL_resting/RL/*'

list_a = []
list_b = []

for i in sorted(glob(LR_path)):

    a = sio.loadmat(i)
    a = a['ROI']
    a = np.expand_dims(a, axis=0)

    name = i.split('/')[-1]

    if a.shape[1] != 1200:
        print(name + 'except because ... '+ str(a.shape))
        continue

    if name == '100206.mat':
        print(name)
        HCP_RESTING_LR = a

        list_a.append(name.split('.')[0])

    else:
        print(name)
        HCP_RESTING_LR = np.concatenate((HCP_RESTING_LR, a))
        list_a.append(name.split('.')[0])

HCP_RESTING_LR_label = np.array(list_a)

print()
print('RL start...')


for j in sorted(glob(RL_path)):

    b = sio.loadmat(j)
    b = b['ROI']
    b = np.expand_dims(b, axis=0)

    name = j.split('/')[-1]


    if b.shape[1] != 1200:
        print(name + ' except because ... '+ str(b.shape))
        continue

    if name == '100307.mat':
        print(name)
        HCP_RESTING_RL = b

        list_b.append(name.split('.')[0])

    else:
        print(name)
        HCP_RESTING_RL = np.concatenate((HCP_RESTING_RL, b))
        list_b.append(name.split('.')[0])

HCP_RESTING_RL_label = np.array(list_b)



print(HCP_RESTING_LR.shape)
print(HCP_RESTING_RL.shape)


np.savez_compressed(save_path, HCP_RESTING_LR=HCP_RESTING_LR, HCP_RESTING_LR_id=HCP_RESTING_LR_label, HCP_RESTING_RL=HCP_RESTING_RL, tfMRI_EMOTION_RL_id=HCP_RESTING_RL_label)




