import nibabel as nib
from nilearn.input_data import NiftiLabelsMasker
import scipy.io as sio

#path = '/media/12T/practice/atlas/aal_roi_atlas.nii.gz'

#atlas = nib.load(path)

#masker = NiftiLabelsMasker(labels_img=atlas, standardize=True)

#path2 = '/media/12T/practice/100206/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz'
#n1 = nib.load(path2)

#sub = path2.split('/')

#t2 = masker.fit_transform(n1)
#print(t2.shape)

#path3 = '/media/12T/practice/' + sub[-1] + '.mat'
#sio.savemat(path3, {'ROI': t2})
#print(path2+'done!!')

import numpy as np

path = '/media/12T/ADHD200/Outputs/output/cpac_cpac-default-pipeline/sub-0026001_ses-1/func/sub-0026001_ses-1_task-rest_run-1_desc-preproc-1_bold.nii.gz'
path2 = '/media/12T/ADHD200/Outputs/output/cpac_cpac-default-pipeline/sub-0026001_ses-1/func/sub-0026001_ses-1_task-rest_run-1_desc-preproc-2_bold.nii.gz'


p = nib.load(path)
print(p.shape)
p2 = nib.load(path2)
print(p2.shape)
print(p2.affine)
pdata = p.get_fdata()
print('mean p: ', np.mean(pdata[32,32,27,:]))
p2data = p2.get_fdata()
print('mean p2: ', np.mean(p2data[32,32,27,:]))
c = pdata - p2data
print(c.min())
print(c.max())
print(p.shape)

path_raw = '/media/12T/ADHD200/RawDataBIDS/Brown/sub-0026001/ses-1/func/sub-0026001_ses-1_task-rest_run-1_bold.nii.gz'
p_raw = nib.load(path_raw)
print(p_raw.shape)
print(p_raw.affine)
