import os
import pickle
import torch
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from random import shuffle, randrange, choices
from nilearn import image, maskers, datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from nilearn.connectome import ConnectivityMeasure




class DatasetHCPTask(torch.utils.data.Dataset):
    def __init__(self, sourcedir, roi, crop_length=None, k_fold=None, smoothing_fwhm=None):
        super().__init__()

        self.filename = 'train_hcp-task'
        self.filename += f'_roi-{roi}'
        if smoothing_fwhm is not None: self.filename += f'_fwhm-{smoothing_fwhm}'

        if roi == 'schaefer':
            self.roi = datasets.fetch_atlas_schaefer_2018(data_dir=os.path.join(sourcedir, 'roi'))
        elif roi == 'aal':
            self.roi = datasets.fetch_atlas_aal(data_dir=os.path.join(sourcedir, 'roi'))
        elif roi == 'destrieux':
            self.roi = datasets.fetch_atlas_destrieux_2009(data_dir=os.path.join(sourcedir, 'roi'))
        elif roi == 'harvard_oxford':
            self.roi = datasets.fetch_atlas_harvard_oxford(atlas_name='cort-maxprob-thr25-2mm',
                                                           data_dir=os.path.join(sourcedir, 'roi'))

        task_timepoints = {'EMOTION': 176, 'GAMBLING': 253, 'LANGUAGE': 316, 'MOTOR': 284, 'RELATIONAL': 232, 'SOCIAL': 274, 'WM': 405}
        self.sourcedir = sourcedir
        self.crop_length = crop_length
        self.task_list = list(task_timepoints.keys())
        self.task_list.sort()
        print(self.task_list)

        if os.path.isfile(os.path.join(sourcedir, f'{self.filename}.pth')):
            self.timeseries_list, self.label_list = torch.load(os.path.join(sourcedir, f'{self.filename}.pth'))
            # self.timeseries_list, self.label_list = torch.load(os.path.join(sourcedir, f'{self.filename}.pth')) #for aal(practice)

        else:
            roi_masker = maskers.NiftiLabelsMasker(image.load_img(self.roi['maps']))
            self.timeseries_list = []
            self.label_list = []
            for task in self.task_list:
                img_list = [f for f in os.listdir(os.path.join(sourcedir, 'img', 'TASK', task)) if f.endswith('nii.gz')]
                img_list.sort()
                for subject in tqdm(img_list, ncols=60, desc=f'prep:{task.lower()[:3]}'):
                    timeseries = roi_masker.fit_transform(image.load_img(
                        os.path.join(self.sourcedir, 'img', 'TASK', task,
                                     subject)))  # image.load_img('./data/img/TASK/RELATIONAL/100206.nii.gz') -> timeseries.shape (232, 116)
                    if not len(timeseries) == task_timepoints[task]:
                        print(f"short timeseries: {len(timeseries)}")
                        continue
                    self.timeseries_list.append(
                        timeseries)  # subject가 2명이면 7개의 task니까 len(self.timeseries) == 14,  self.timeseries_list[0].shape == (176, 116)
                    self.label_list.append(
                        task)  # subject가 2명이면 7개의 task니까 len(self.label_list) == 14 ['EMOTION', 'EMOTION', 'GAMBLING', 'GAMBLING', 'LANGUAGE', 'LANGUAGE', 'MOTOR', 'MOTOR', 'RELATIONAL', ...]
            torch.save((self.timeseries_list, self.label_list), os.path.join(sourcedir, f'{self.filename}.pth'))  # './data/hcp-task_roi-aal.pth'







        if k_fold > 1:
            self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0)
            self.k = None
        else:
            self.k_fold = None

        self.num_nodes = self.timeseries_list[0].shape[1]
        self.num_classes = len(set(self.label_list))
        self.train = None




    def __len__(self):
        return len(self.fold_idx) if self.k is not None else len(self.timeseries_list)


    def connectivity(self, input, type='correlation', vectorization=True, fisher_t=False):
        '''kind{“covariance”, “correlation”, “partial correlation”, “tangent”, “precision”}, optional'''
        measure = ConnectivityMeasure(kind=type, vectorize=vectorization, discard_diagonal=True)
        mt = measure.fit_transform(input)

        if fisher_t == True:
            for i in range(len(mt)):
                mt[i][:] = np.arctanh(mt[i][:])
        return mt



    def set_fold(self, fold, train=True):
        if not self.k_fold:
            return
        self.k = fold
        train_idx, test_idx = list( self.k_fold.split(self.timeseries_list, self.label_list) )[fold] #fold가 고정됨


        if train:
            shuffle(train_idx)
            self.fold_idx = train_idx
            self.train = True
        else:
            self.fold_idx = test_idx
            self.train = False




    def __getitem__(self, idx):
        timeseries = self.timeseries_list[self.fold_idx[idx]]
        timeseries = (timeseries - np.mean(timeseries, axis=0, keepdims=True)) / (np.std(timeseries, axis=0, keepdims=True) + 1e-9)
        if not self.crop_length is None:
            timeseries = timeseries[:self.crop_length]
            sub = np.reshape(timeseries, ((1,) + timeseries.shape))
            fc = self.connectivity(sub, type='correlation', vectorization=False, fisher_t=False)
            fc = np.squeeze(fc, 0)

        task = self.label_list[self.fold_idx[idx]]

        for task_idx, _task in enumerate(self.task_list):
            if task == _task:
                label = task_idx

        return {'timeseries': torch.tensor(fc, dtype=torch.float32), 'label': torch.tensor(label)}







class DatasetHCPTask_test(torch.utils.data.Dataset):
    def __init__(self, sourcedir, roi, crop_length=None, k_fold=None, smoothing_fwhm=None):
        super().__init__()

        self.filename = 'test_hcp-task'
        self.filename += f'_roi-{roi}'
        if smoothing_fwhm is not None: self.filename += f'_fwhm-{smoothing_fwhm}'

        if roi == 'schaefer':
            self.roi = datasets.fetch_atlas_schaefer_2018(data_dir=os.path.join(sourcedir, 'roi'))
        elif roi == 'aal':
            self.roi = datasets.fetch_atlas_aal(data_dir=os.path.join(sourcedir, 'roi'))
        elif roi == 'destrieux':
            self.roi = datasets.fetch_atlas_destrieux_2009(data_dir=os.path.join(sourcedir, 'roi'))
        elif roi == 'harvard_oxford':
            self.roi = datasets.fetch_atlas_harvard_oxford(atlas_name='cort-maxprob-thr25-2mm',
                                                           data_dir=os.path.join(sourcedir, 'roi'))

        task_timepoints = {'EMOTION': 176, 'GAMBLING': 253, 'LANGUAGE': 316, 'MOTOR': 284, 'RELATIONAL': 232, 'SOCIAL': 274, 'WM': 405}
        self.sourcedir = sourcedir
        self.crop_length = crop_length
        self.task_list = list(task_timepoints.keys())
        self.task_list.sort()
        print(self.task_list)

        if os.path.isfile(os.path.join(sourcedir, f'{self.filename}.pth')):
            self.timeseries_list, self.label_list = torch.load(os.path.join(sourcedir, f'{self.filename}.pth'))
            # self.timeseries_list, self.label_list = torch.load(os.path.join(sourcedir, f'{self.filename}.pth')) #for aal(practice)

        else:
            roi_masker = maskers.NiftiLabelsMasker(image.load_img(self.roi['maps']))
            self.timeseries_list = []
            self.label_list = []
            for task in self.task_list:
                img_list = [f for f in os.listdir(os.path.join(sourcedir, 'img', 'TASK', task)) if f.endswith('nii.gz')]
                img_list.sort()
                for subject in tqdm(img_list, ncols=60, desc=f'prep:{task.lower()[:3]}'):
                    timeseries = roi_masker.fit_transform(image.load_img(
                        os.path.join(self.sourcedir, 'img', 'TASK', task,
                                     subject)))  # image.load_img('./data/img/TASK/RELATIONAL/100206.nii.gz') -> timeseries.shape (232, 116)
                    if not len(timeseries) == task_timepoints[task]:
                        print(f"short timeseries: {len(timeseries)}")
                        continue
                    self.timeseries_list.append(
                        timeseries)  # subject가 2명이면 7개의 task니까 len(self.timeseries) == 14,  self.timeseries_list[0].shape == (176, 116)
                    self.label_list.append(
                        task)  # subject가 2명이면 7개의 task니까 len(self.label_list) == 14 ['EMOTION', 'EMOTION', 'GAMBLING', 'GAMBLING', 'LANGUAGE', 'LANGUAGE', 'MOTOR', 'MOTOR', 'RELATIONAL', ...]
            torch.save((self.timeseries_list, self.label_list), os.path.join(sourcedir, f'{self.filename}.pth'))  # './data/hcp-task_roi-aal.pth'







        if k_fold > 1:
            self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0)
            self.k = None
        else:
            self.k_fold = None

        self.num_nodes = self.timeseries_list[0].shape[1]
        self.num_classes = len(set(self.label_list))
        self.train = None




    def __len__(self):
        return len(self.fold_idx) if self.k is not None else len(self.timeseries_list)




    def connectivity(self, input, type='correlation', vectorization=True, fisher_t=False):
        '''kind{“covariance”, “correlation”, “partial correlation”, “tangent”, “precision”}, optional'''
        measure = ConnectivityMeasure(kind=type, vectorize=vectorization, discard_diagonal=True)
        mt = measure.fit_transform(input)

        if fisher_t == True:
            for i in range(len(mt)):
                mt[i][:] = np.arctanh(mt[i][:])
        return mt





    def __getitem__(self, idx):
        timeseries = self.timeseries_list[idx]
        timeseries = (timeseries - np.mean(timeseries, axis=0, keepdims=True)) / (np.std(timeseries, axis=0, keepdims=True) + 1e-9)
        if not self.crop_length is None:
            timeseries = timeseries[:self.crop_length]
            sub = np.reshape(timeseries, ((1,) + timeseries.shape))
            fc = self.connectivity(sub, type='correlation', vectorization=False, fisher_t=False)
            fc = np.squeeze(fc, 0)
        task = self.label_list[idx]

        for task_idx, _task in enumerate(self.task_list):
            if task == _task:
                label = task_idx

        return {'timeseries': torch.tensor(fc, dtype=torch.float32), 'label': torch.tensor(label)}







