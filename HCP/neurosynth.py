import nibabel as nib
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
import scipy.io as sio
from glob import glob
import os
import os.path
import numpy as np



atlas = nib.load('/media/12T/practice/atlas/ho_roi_atlas.nii.gz')

n1 = nib.load('/home/djk/Documents/MDD/sensorimotor network_association-test_z_FDR_0.01.nii.gz')
masker = NiftiLabelsMasker(labels_img=atlas, standardize=True)


atlas_list=['ho']
t1 = masker.fit_transform(n1)

print(t1.shape)

