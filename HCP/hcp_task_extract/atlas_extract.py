# .mat file 만들기

import nibabel as nib
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
import scipy.io as sio
from glob import glob
import os
import os.path
import numpy as np

from nilearn.datasets import fetch_atlas_destrieux_2009
# exc = []
# path = '/media/12T/practice/atlas/aal_roi_atlas.nii.gz' # Automated Anatomical Labeling (AAL)_116
# path = './atlas/AAL3v1_for_SPM12/AAL3/AAL3v1_1mm.nii.gz' # AAL3v1_166 (if MNI > 1mm, this is not valid)
# path = './atlas/yeo_atlas_2mm_17.nii.gz' # Yeo17_114
# path = './atlas/yeo_atlas_2mm.nii.gz' # Yeo7_51
# path = './atlas/cc200_roi_atlas.nii.gz' # Craddock200_200
# path = './atlas/cc400_roi_atlas.nii.gz' # Craddock400_392
# path = './atlas/ho_roi_atlas.nii.gz' # Harvard-Oxford (HO)_111
# path = './atlas/fsl/data/atlases/HarvardOxford/HarvardOxford-cortl-maxprob-thr25-2mm.nii.gz' # HO_cort_96
# path = './atlas/fsl/data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr25-2mm.nii.gz' # HO_sub_21
# path = './atlas/tt_roi_atlas.nii.gz' # Talaraich and Tournoux (TT)_110
# path = './atlas/ez_roi_atlas.nii.gz' # Eickhoff-Zilles (EZ)_116
# path = './atlas/dos160_roi_atlas.nii.gz' # Dosenbach160_161 (the last ROI is not used)
# path = './atlas/destrieux_2009/destrieux2009_rois_lateralized.nii.gz' # Destrieux_148
# path = './atlas/schaefer_2018/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii.gz' # Schaefer_100 {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000}
# path = './atlas/schaefer_2018/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz' # Schaefer_400 {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000}

# atlas = nib.load(path)

# schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=400, data_dir='./atlas')
# HO_cort = datasets.fetch_atlas_harvard_oxford(atlas_name='cort-maxprob-thr25-2mm', data_dir='./atlas', symmetric_split=True)
# atlas = HO_cort.maps

# masker_2mm = NiftiLabelsMasker(labels_img=atlas_2,standardize=True, low_pass=0.054, high_pass=0.039, t_r=3) # bandpass filtering
# masker = NiftiLabelsMasker(labels_img=atlas,standardize=True) # normalize
# masker_2mm = NiftiLabelsMasker(labels_img=atlas_2,standardize=False) # without normalize

###### for ABIDE1 4d nii ######
# for fn in glob('/DataCommon3/daheo/TC/*'):
#     try:
#         n1 = nib.load(fn)
#         file = fn.split('/')
#         prepro_sub = file[-1].split('.')
#         sub = prepro_sub[-2].split('_func_')
#
#         t1 = masker.fit_transform(n1)
#         print(t1.shape)
#
#         path = '/DataCommon3/daheo/ROI/HOsub_/TC/' + sub[-2] + '.mat'
#         sio.savemat(path, {'ROI': t1})
#         print(fn + ' done!')
#     except:
#         exc.append(fn)
#         pass
#
# print(exc)

# for fn in glob('/mnt/f71146c3-8fea-47c7-88f9-0f423ff37ba9/ADNI2GO/Bhermiteb0a0/FunImgARCWSFB/*/*'):
#     try:
#         n1 = nib.load(fn)
#         sub = fn.split('/')
#
#         t1 = masker.fit_transform(n1)
#         print(t1.shape)
#
#         path = '/mnt/f71146c3-8fea-47c7-88f9-0f423ff37ba9/ADNI2GO/Bhermiteb0a0/Yeo7_51/' + sub[-2] + '.mat'
#         sio.savemat(path, {'ROI': t1})
#         print(fn + ' done!')
#     except:
#         exc.append(fn)
#         pass
#
# print(exc)

''' atlas + motor task '''
# data_list=['tfMRI_MOTOR_LR','tfMRI_MOTOR_RL']
# atlas_list=['AAL_116','Yeo7','Yeo17']
#
# for fn in data_list:
#     try:
#         os.makedirs('/mnt/hcp/homes/jwlee/HCP_train_data/'+fn+'/4d', exist_ok=True)
#
#
#         for al in atlas_list:
#             try:
#                 os.makedirs('/mnt/hcp/homes/jwlee/HCP_train_data/'+fn+'/4d/'+al, exist_ok=True)
#
#
#
#                 for sub in glob('/mnt/hcp/HCP/*/MNINonLinear/Results/'+fn):
#                     try:
#
#                         sub1 = sub.split('/')
#                         file = '/mnt/hcp/homes/jwlee/HCP_train_data/' + fn + '/4d/' + al + '/' + sub1[-4] + '.mat'
#
#                         if os.path.isfile(file):
#                             print(fn + '/4d/' + al + '/' + sub1[-4] + '.mat', "파일이 존재합니다.")
#                             continue
#
#                         if al == 'AAL_116':
#                             atlas = nib.load('/media/12T/practice/atlas/aal_roi_atlas.nii.gz')
#                         elif al == 'Yeo7':
#                             atlas = nib.load('/media/12T/practice/atlas/yeo_atlas_2mm.nii.gz')
#                         else:
#                             atlas = nib.load('/media/12T/practice/atlas/yeo_atlas_2mm_17.nii.gz')
#
#                         masker = NiftiLabelsMasker(labels_img=atlas, standardize=True)
#
#                         if fn == 'tfMRI_MOTOR_LR':
#                             n1 = nib.load(sub + '/tfMRI_MOTOR_LR.nii.gz')
#                         else:
#                             n1 = nib.load(sub + '/tfMRI_MOTOR_RL.nii.gz')
#
#                         print(al)
#
#
#
#                         t1 = masker.fit_transform(n1)
#                         print(t1.shape)
#
#                         path2 = '/mnt/hcp/homes/jwlee/HCP_train_data/' + fn + '/4d/' + al + '/' + sub1[-4] + '.mat'
#                         sio.savemat(path2, {'ROI' : t1})
#                         print(path2 + 'done!')
#                     except:
#                         pass
#             except:
#                 pass
#     except:
#         pass




'''for raw signal'''
# for b in range(1,6):
#     for fn in glob('/media/j/8463e954-8823-4176-ad79-f1ab9690d06e/ADNI_split_5band/CN_band%d/FunImgARCFWS/*/*'%b):
#         n1 = nib.load(fn);
#         sub = fn.split('/')
#
#         t1 = masker_2mm.fit_transform(n1)
#         print(t1.shape)
#
#         path = '/media/j/8463e954-8823-4176-ad79-f1ab9690d06e/ADNI_split_5band/CN_band%d/yeo17_cn_no/yeo17_'%b + sub[
#             -2] + '.mat'
#         sio.savemat(path, {'ROI': t1})
#         print(fn + ' done!')


'''for alff'''
# for fn in glob('/media/j/8463e954-8823-4176-ad79-f1ab9690d06e/ADNI_full_band/CN/Results/ALFF_FunImgARCFWS/*'):
#     n1 = nib.load(fn);
#     # sub = fn.split('/')
#     _, tail = os.path.split(fn)
#
#     t1 = masker_2mm.fit_transform(n1)
#     print(t1.shape)
#
#     path = '/media/j/8463e954-8823-4176-ad79-f1ab9690d06e/ADNI_full_band/CN/ex/yeo17_'+os.path.splitext(tail)[0]+'.mat'
#     sio.savemat(path,{'ROI':t1})
#     print(fn+' done!')


'''for ho atlas to motor task'''

# data_list=['tfMRI_MOTOR_LR','tfMRI_MOTOR_RL']
# atlas_list=['ho']
#
# for fn in data_list:
#     try:
#         os.makedirs('/mnt/hcp/homes/jwlee/HCP_train_data/'+fn+'/4d', exist_ok=True)
#
#
#         for al in atlas_list:
#             try:
#                 os.makedirs('/mnt/hcp/homes/jwlee/HCP_train_data/'+fn+'/4d/'+al, exist_ok=True)
#
#
#
#                 for sub in glob('/mnt/hcp/HCP/*/MNINonLinear/Results/'+fn):
#                     try:
#
#                         sub1 = sub.split('/')
#                         file = '/mnt/hcp/homes/jwlee/HCP_train_data/' + fn + '/4d/' + al + '/' + sub1[-4] + '.mat'
#
#                         if os.path.isfile(file):
#                             print(fn + '/4d/' + al + '/' + sub1[-4] + '.mat', "파일이 존재합니다.")
#                             continue
#
#                         if al == 'ho':
#                             atlas = nib.load('/media/12T/practice/atlas/ho_roi_atlas.nii.gz')
#
#
#                         masker = NiftiLabelsMasker(labels_img=atlas, standardize=True)
#
#                         if fn == 'tfMRI_MOTOR_LR':
#                             n1 = nib.load(sub + '/tfMRI_MOTOR_LR.nii.gz')
#                         else:
#                             n1 = nib.load(sub + '/tfMRI_MOTOR_RL.nii.gz')
#
#                         print(al)
#
#
#
#                         t1 = masker.fit_transform(n1)
#                         print(t1.shape)
#
#                         path2 = '/mnt/hcp/homes/jwlee/HCP_train_data/' + fn + '/4d/' + al + '/' + sub1[-4] + '.mat'
#                         sio.savemat(path2, {'ROI' : t1})
#                         print(path2 + 'done!')
#                     except:
#                         pass
#             except:
#                 pass
#     except:
#         pass


'''for yeo7_network atlas to motor task'''

data_list=['tfMRI_MOTOR_LR','tfMRI_MOTOR_RL']
atlas_list=['yeo7_network']

for fn in data_list:
    try:
        os.makedirs('/mnt/hcp/homes/jwlee/HCP_train_data/'+fn+'/4d', exist_ok=True)


        for al in atlas_list:
            try:
                os.makedirs('/mnt/hcp/homes/jwlee/HCP_train_data/'+fn+'/4d/'+al, exist_ok=True)



                for sub in glob('/mnt/hcp/HCP/*/MNINonLinear/Results/'+fn):
                    try:

                        sub1 = sub.split('/')
                        file = '/mnt/hcp/homes/jwlee/HCP_train_data/' + fn + '/4d/' + al + '/' + sub1[-4] + '.mat'

                        if os.path.isfile(file):
                            print(fn + '/4d/' + al + '/' + sub1[-4] + '.mat', "파일이 존재합니다.")
                            continue

                        if al == 'yeo7_network':
                            atlas = nib.load('/media/12T/practice/atlas/yeo_2011/Yeo_JNeurophysiol11_MNI152/Yeo2011_7Networks_MNI152_FreeSurferConformed1mm.nii.gz')


                        masker = NiftiLabelsMasker(labels_img=atlas, standardize=True)

                        if fn == 'tfMRI_MOTOR_LR':
                            n1 = nib.load(sub + '/tfMRI_MOTOR_LR.nii.gz')
                        else:
                            n1 = nib.load(sub + '/tfMRI_MOTOR_RL.nii.gz')

                        print(al)



                        t1 = masker.fit_transform(n1)
                        print(t1.shape)

                        path2 = '/mnt/hcp/homes/jwlee/HCP_train_data/' + fn + '/4d/' + al + '/' + sub1[-4] + '.mat'
                        sio.savemat(path2, {'ROI' : t1})
                        print(path2 + 'done!')
                    except:
                        pass
            except:
                pass
    except:
        pass