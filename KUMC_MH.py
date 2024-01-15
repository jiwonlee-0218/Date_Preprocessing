import os.path

import numpy as np
import scipy.io as sio
from glob import glob
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
import nibabel as nib


''' .mat파일 만들기 '''
# start_str = 'swau'
#
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










''' timeseries data (180, 200)랑 label같이 dictionary형태로 '''
# start_str = 'swau'
# save_path = '/DataRead/KUMC_MH/2304_HAN_suk/timeseries_data/dict'
#
#
# for subject in sorted(glob(f'/DataRead/KUMC_MH/2304_HAN_suk/timeseries_data/Schaefer2018_17Networks/{start_str}*.mat')):
#
#
#     a = sio.loadmat(subject)
#     a = a['ROI']
#
#     if a.shape[0] == 180:
#         print(subject)
#         a = np.expand_dims(a, axis=0)
#
#
#         if subject.split('/')[-1] == 'swau1001.mat':
#             mdd_dataset = a
#         else:
#             mdd_dataset = np.concatenate((mdd_dataset, a))
#
#
#
#     else:
#         print("sorry..: ",subject.split('/')[-1], a.shape)
#         continue
#
# label_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1,
# 2, 2, 2, 2, 2, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 2, 2, 1, 2, 1, 1, 2, 2, 2, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 2, 2, 2, 2, 2, 1,
# 2, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 2, 1, 2, 1, 2, 2, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
# 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
#
# mdd_dataset_label = np.array(label_list)
#
# np.savez_compressed('/DataRead/KUMC_MH/2304_HAN_suk/timeseries_data/dict/kumc_mh_mdd_Schaefer17Net', mdd_dataset=mdd_dataset, mdd_dataset_label=mdd_dataset_label)












''' timeseries data로 만든 fc (200, 200)랑 label같이 dictionary형태로 '''
# def connectivity(input, type='correlation', vectorization=True, fisher_t=False):
#     '''kind{“covariance”, “correlation”, “partial correlation”, “tangent”, “precision”}, optional'''
#     measure = ConnectivityMeasure(kind=type, vectorize=vectorization, discard_diagonal=True)
#     mt = measure.fit_transform(input)
#
#     if fisher_t == True:
#         for i in range(len(mt)):
#             mt[i][:] = np.arctanh(mt[i][:])
#     return mt
#
#
#
# start_str = 'swau'
#
# for subject in sorted(glob(f'/DataRead/KUMC_MH/2304_HAN_suk/timeseries_data/Schaefer2018_17Networks/{start_str}*.mat')):
#
#
#
#     a = sio.loadmat(subject)
#     a = a['ROI']
#
#     if a.shape[0] == 180:
#         print(subject)
#         a = np.expand_dims(a, axis=0) #(1, 180, 200)==(1, time, roi)
#         fc = connectivity(a, type='correlation', vectorization=False, fisher_t=False) #(1, 200, 200)
#
#
#         if subject.split('/')[-1] == 'swau1001.mat':
#             mdd_fc_dataset = fc
#         else:
#             mdd_fc_dataset = np.concatenate((mdd_fc_dataset, fc))
#
#
#
#     else:
#         print("sorry..: ",subject.split('/')[-1], a.shape)
#         continue
#
# label_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1,
# 2, 2, 2, 2, 2, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 2, 2, 1, 2, 1, 1, 2, 2, 2, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 2, 2, 2, 2, 2, 1,
# 2, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 2, 1, 2, 1, 2, 2, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
# 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
#
# mdd_fc_dataset_label = np.array(label_list)
#
# np.savez_compressed('/DataRead/KUMC_MH/2304_HAN_suk/FC_data/dict/kumc_mh_mdd_Schaefer17Net', mdd_fc_dataset=mdd_fc_dataset, mdd_fc_dataset_label=mdd_fc_dataset_label)