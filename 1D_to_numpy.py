import numpy as np
import nibabel as nib
import os
from scipy.io import savemat, loadmat
import pandas as pd

# 1d to ndarray (ROI)
# check file extension
name='/media/12T/ABIDE/CASE1/case1_default/case1_default_output/output/cpac_cpac-default-pipeline/0051456_session_1/func/sub-0051456_session_1_task-func-1_atlas-AALspace-MNI152NLin6res-1x1x1_desc-Mean-1_timeseries.1D'
# print(os.path.splitext(name)[1])

# load .1D file
a = np.loadtxt('/media/12T/ABIDE/CASE1/case1_default/case1_default_output/output/cpac_cpac-default-pipeline/0051456_session_1/func/sub-0051456_session_1_task-func-1_atlas-AALspace-MNI152NLin6res-1x1x1_desc-Mean-1_timeseries.1D', delimiter=',')
# len(a) = (150,116)
# print(np.ndim(a)) = 2d

mdic = {'ROI': a}
savemat('/media/12T/ABIDE/hello.mat',mdic)


b = loadmat('/media/12T/ABIDE/hello.mat')
# print(b['ROI'].shape) = (150,116)
bb = b['ROI']


# csv to ndarray (functional correlation)
c = pd.read_csv('/media/12T/ABIDE/CASE1/case1_default/case1_default_output/output/cpac_cpac-default-pipeline/0051456_session_1/func/sub-0051456_session_1_task-func-1_atlas-aalmaskpad_desc-ndmg-1_correlations.csv')
cc = c.iloc[:,0:116]
cp = cc.to_numpy()
np.save('/media/12T/ABIDE/fc.npy',cp)

# a = np.load('/media/12T/ABIDE/fc.npy')
