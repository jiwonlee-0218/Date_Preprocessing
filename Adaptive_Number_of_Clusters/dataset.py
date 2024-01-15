import os
import pickle
import torch
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from random import shuffle, randrange, choices
from nilearn import image, datasets
from nilearn.input_data import NiftiLabelsMasker
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn.connectome import ConnectivityMeasure


sourcedir= '/media/12T/practice/data'
roi= 'harvard_oxford'
path = '/mnt/hcp/HCP/*'

class DatasetHCPTask(torch.utils.data.Dataset):
    def __init__(self, sourcedir, roi, dynamic_length=None, smoothing_fwhm=None):
        super().__init__()

        self.filename = 'hcp-task'
        self.filename += f'_roi-{roi}'
        if smoothing_fwhm is not None: self.filename += f'_fwhm-{smoothing_fwhm}'

        if roi=='schaefer': self.roi = datasets.fetch_atlas_schaefer_2018(data_dir=os.path.join(sourcedir, 'roi'))
        elif roi=='aal': self.roi = datasets.fetch_atlas_aal(data_dir=os.path.join(sourcedir, 'roi'))
        elif roi=='destrieux': self.roi = datasets.fetch_atlas_destrieux_2009(data_dir=os.path.join(sourcedir, 'roi'))
        elif roi=='harvard_oxford': self.roi = datasets.fetch_atlas_harvard_oxford(atlas_name='cort-maxprob-thr25-2mm', data_dir=os.path.join(sourcedir, 'roi'))

        task_timepoints = {'EMOTION': 176, 'GAMBLING': 253, 'LANGUAGE': 316, 'MOTOR': 284, 'RELATIONAL': 232, 'SOCIAL': 274, 'WM': 405}
        self.sourcedir = sourcedir
        self.dynamic_length = dynamic_length
        self.task_list = list(task_timepoints.keys())
        self.task_list.sort()
        print(self.task_list)

        if os.path.isfile(os.path.join(sourcedir, f'{self.filename}.pth')):
            self.timeseries_list, self.label_list = torch.load(os.path.join(sourcedir, f'{self.filename}.pth'))
        else:
            roi_masker = NiftiLabelsMasker(image.load_img(self.roi['maps']))
            self.timeseries_list = []
            self.label_list = []
            self.id = {'EMOTION': [], 'GAMBLING': [], 'LANGUAGE' : [], 'MOTOR' : [], 'RELATIONAL' : [], 'SOCIAL' : [], 'WM' : []}
            for task in tqdm(self.task_list):
                try:
                    for sub in tqdm(sorted(glob(path))):
                        try:
                            name = sub.split('/')[-1]
                            file_name = 'tfMRI_' + task + '_LR'

                            timeseries = roi_masker.fit_transform(  image.load_img(os.path.join('/mnt/hcp/HCP', name, 'MNINonLinear/Results/', file_name, file_name+'.nii.gz'))  )
                            if not len(timeseries) == task_timepoints[task]:
                                print(f"short timeseries: {len(timeseries)}")
                                continue
                            self.timeseries_list.append(timeseries)
                            self.label_list.append(task)
                            self.id[task].append(name)


                            print(name, file_name,'success')
                        except:
                            pass
                except:
                    pass
            torch.save((self.timeseries_list, self.label_list, self.id), os.path.join(sourcedir, f'{self.filename}.pth')) #'./data/hcp-task_roi-aal.pth'


DatasetHCPTask(sourcedir, roi)











########################################################################################################################데이터 확인
# aal_home = '/DataCommon/jwlee/aal_roi_atlas.nii.gz'
# aal_datacommon = '/DataCommon/jwlee/data/roi/aal_SPM12/aal/atlas/AAL.nii'
# data_100206 = '/DataCommon/jwlee/100206_tfMRI_EMOTION_LR.nii.gz'
# data_100307 = '/DataCommon/jwlee/100307_tfMRI_EMOTION_LR.nii.gz'
# data_100408 = '/DataCommon/jwlee/100408_tfMRI_EMOTION_LR.nii.gz'
#
# a1 = nib.load(aal_home) #65x77x63
# a2 = nib.load(aal_datacommon) #91x109x91
#
# a1_data = a1.get_fdata()
# a2_data = a2.get_fdata()
#
# roi_masker_a1 = NiftiLabelsMasker(image.load_img(a1))
# roi_masker_a2 = NiftiLabelsMasker(image.load_img(a2))
#
# timeseries_a1 = roi_masker_a1.fit_transform(  image.load_img(   data_100408   )  )
# timeseries_a2 = roi_masker_a2.fit_transform(  image.load_img(   data_100408   )  )
#
# sub_a1 = np.reshape(timeseries_a1, ((1,) + timeseries_a1.shape))
# sub_a2 = np.reshape(timeseries_a2, ((1,) + timeseries_a2.shape))
#
# def connectivity(input, type='correlation', vectorization=True, fisher_t=False):
#     '''kind{“covariance”, “correlation”, “partial correlation”, “tangent”, “precision”}, optional'''
#     measure = ConnectivityMeasure(kind=type, vectorize=vectorization, discard_diagonal=True)
#     mt = measure.fit_transform(input)
#     if fisher_t == True:
#         for i in range(len(mt)):
#             mt[i][:] = np.arctanh(mt[i][:])
#     return mt
#
# fc_a1 = connectivity(sub_a1, type='correlation', vectorization=False, fisher_t=False)
# fc_a2 = connectivity(sub_a2, type='correlation', vectorization=False, fisher_t=False)
#
#
# plt.imshow(fc_a1[0])
# plt.colorbar()
# plt.show()
#
# plt.imshow(fc_a2[0])
# plt.colorbar()
# plt.show()