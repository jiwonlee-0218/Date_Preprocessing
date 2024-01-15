import nibabel as nib
from nilearn.input_data import NiftiLabelsMasker
import scipy.io as sio
import numpy as np

path = '/media/12T/practice/atlas/destrieux_2009/destrieux2009_rois_lateralized.nii.gz'

atlas = nib.load(path)

masker = NiftiLabelsMasker(labels_img=atlas, standardize=True)

path2 = '/media/12T/practice/100206/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz'
n1 = nib.load(path2)

sub = path2.split('/')

t2 = masker.fit_transform(n1)
print(t2.shape)

t3 = nib.load('/media/12T/des2009_roi_later.ptseries.nii')
t3 = np.array(t3.get_fdata())
mean = np.mean(t3, 0)
std = np.std(t3, 0)
t3_2 = (t3-mean)/std

path3 = '/media/12T/practice/' + sub[-1] + '.mat'
sio.savemat(path3, {'ROI': t2})
print(path2+'done!!')
