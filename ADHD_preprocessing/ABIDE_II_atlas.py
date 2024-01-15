import os.path

import numpy as np
import scipy.io as sio
from glob import glob
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
import nibabel as nib
import os


''' .mat파일 만들기 '''

cite = ['Brown', 'KKI', 'NeuroIMAGE', 'NYU', 'OHSU', 'Peking', 'Pittsburgh', 'WashU']
atlas_name = ['AAL1_116', 'CC_200', 'Yeo7_51', 'Yeo17_114']

global_path = '/DataRead/ADHD200/Preprocessed/dparsf/4D/filt_global'
no_gloabl_path = ' /DataRead/ADHD200/Preprocessed/dparsf/4D/filt_noglobal'



for i in cite:
    cite_path = path + i +'/ROIs/' + atlas_dir
    os.makedirs(os.path.join(cite_path, 'global'), exist_ok=True)
    os.makedirs(os.path.join(cite_path, 'noglobal'), exist_ok=True)









''' global '''
for i in cite:
    cite_path = path + i +'/ROIs/' + atlas_dir
    os.makedirs(os.path.join(cite_path, 'global'), exist_ok=True)
    os.makedirs(os.path.join(cite_path, 'noglobal'), exist_ok=True)


















''' no global '''



















# for subject in sorted(glob(f'/DataRead/KUMC_MH/2304_HAN_suk/rs/{start_str}*.nii')):
#     print(subject)
#     sub = nib.load(subject)
#     subject_id = subject.split('/')[-1].split('.')[0]
#
#     atlas_path = '/DataRead/KUMC_MH/2304_HAN_suk/atlas/HarvardOxford/HarvardOxford-cort-maxprob-thr50-2mm.nii.gz'
#     HarvardOxford = nib.load(atlas_path)
#
#     masker = NiftiLabelsMasker(labels_img=HarvardOxford, standardize=True)
#
#     t1 = masker.fit_transform(sub)
#     print(t1.shape)
#
#     save_path = '/DataRead/KUMC_MH/2304_HAN_suk/timeseries_data/HarvardOxford_thr50/' + subject_id+'.mat'
#     sio.savemat(save_path, {'ROI': t1})