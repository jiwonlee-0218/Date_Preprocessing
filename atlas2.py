import nibabel as nib
from nilearn.input_data import NiftiLabelsMasker
import scipy.io as sio
import glob
import os
import matplotlib.pyplot as plt

# path = '/media/12T/practice/atlas/aal_roi_atlas.nii.gz'
#
# atlas = nib.load(path)
#
# masker = NiftiLabelsMasker(labels_img=atlas, standardize=True)
#
# path2 = '/media/12T/ccc.nii.gz'
# n1 = nib.load(path2)
#
# sub = path2.split('/')
#
# t1 = masker.fit_transform(n1)
# print(t1.shape)
#
# path3 = '/media/12T/practice/' + sub[-1] + '.mat'
# sio.savemat(path3, {'ROI': t1})
# print(path2+'done!!')








dir = '/mnt/hcp/homes/jwlee/HCP_train_data/tfMRI_MOTOR_LR/4d/AAL_116/*'

for sub in glob(dir):
    try:
        plt.xlabel('time')
        plt.ylabel('ROIs')
        plt.imshow(sub)
        plt.title("Original_X")
        plt.colorbar()
        break

    except:
        pass