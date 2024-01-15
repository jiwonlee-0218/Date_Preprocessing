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
# path = '/media/12T/practice/atlas/ho_roi_atlas.nii.gz' # Harvard-Oxford (HO)_111
# path = './atlas/AAL3v1_for_SPM12/AAL3/AAL3v1_1mm.nii.gz' # AAL3v1_166 (if MNI > 1mm, this is not valid)
# path = '/media/12T/practice/atlas/yeo_atlas_2mm_17.nii.gz' # Yeo17_114
# path = '/media/12T/practice/atlas/yeo_atlas_2mm.nii.gz' # Yeo7_51
# path = '/media/12T/practice/atlas/cc200_roi_atlas.nii.gz' # Craddock200_200
# path = '/media/12T/practice/atlas/cc400_roi_atlas.nii.gz' # Craddock400_392
# path = './atlas/fsl/data/atlases/HarvardOxford/HarvardOxford-cortl-maxprob-thr25-2mm.nii.gz' # HO_cort_96
# path = './atlas/fsl/data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr25-2mm.nii.gz' # HO_sub_21
# path = './atlas/tt_roi_atlas.nii.gz' # Talaraich and Tournoux (TT)_110
# path = './atlas/ez_roi_atlas.nii.gz' # Eickhoff-Zilles (EZ)_116
# path = './atlas/dos160_roi_atlas.nii.gz' # Dosenbach160_161 (the last ROI is not used)
# path = '/media/12T/practice/atlas/destrieux_2009/destrieux2009_rois_lateralized.nii.gz' # Destrieux_148
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
# # atlas_list=['AAL_116','Yeo7','Yeo17','HO_111','Destrieux_148']
# atlas_list=['Shen_1mm','Shen_2mm']
#
# for fn in data_list:
#     try:
#
#
#         for al in atlas_list:
#             try:
#
#
#
#                 for sub in glob('/mnt/hcp/HCP/*'):
#                     try:
#
#                         sub1 = sub.split('/')
#
#                         if fn == 'tfMRI_MOTOR_LR':
#                             file = '/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] +'/MOTOR/LR/' + al + '_' + sub1[-1] +'.mat'
#
#                             if os.path.isfile(file):
#                                 print('HCP_extract/' + sub1[-1] +'/MOTOR/LR/' + al + '_' + sub1[-1] +'.mat', "파일이 존재합니다.")
#                                 continue
#
#                         if fn == 'tfMRI_MOTOR_RL':
#                             file = '/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] +'/MOTOR/RL/' + al + '_' + sub1[-1] +'.mat'
#
#                             if os.path.isfile(file):
#                                 print('HCP_extract/' + sub1[-1] +'/MOTOR/RL/' + al + '_' + sub1[-1] +'.mat', "파일이 존재합니다.")
#                                 continue
#
#                         if al == 'AAL_116':
#                             atlas = nib.load('/media/12T/practice/atlas/aal_roi_atlas.nii.gz')
#                         elif al == 'Yeo7':
#                             atlas = nib.load('/media/12T/practice/atlas/yeo_atlas_2mm.nii.gz')
#                         elif al == 'Yeo17':
#                             atlas = nib.load('/media/12T/practice/atlas/yeo_atlas_2mm_17.nii.gz')
#                         elif al == 'HO_111':
#                             atlas = nib.load('/media/12T/practice/atlas/ho_roi_atlas.nii.gz')
#                         elif al == 'Shen_1mm':
#                             atlas = nib.load('/media/12T/practice/atlas/shen_1mm_268_parcellation.nii.gz')
#                         elif al == 'Shen_2mm':
#                             atlas = nib.load('/media/12T/practice/atlas/shen_2mm_268_parcellation.nii.gz')
#                         else:
#                             atlas = nib.load('/media/12T/practice/atlas/destrieux_2009/destrieux2009_rois_lateralized.nii.gz')
#
#                         masker = NiftiLabelsMasker(labels_img=atlas, standardize=True)
#
#                         if fn == 'tfMRI_MOTOR_LR':
#                             n1 = nib.load('/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] +'/MOTOR/LR/tfMRI_MOTOR_LR.nii.gz')
#                             t1 = masker.fit_transform(n1)
#                             print(t1.shape)
#                             path2 = '/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] + '/MOTOR/LR/' + al + '_' + sub1[-1] +'.mat'
#                             sio.savemat(path2, {'ROI': t1})
#                             print(path2 + 'done!')
#                         else:
#                             n1 = nib.load('/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] +'/MOTOR/RL/tfMRI_MOTOR_RL.nii.gz')
#                             t1 = masker.fit_transform(n1)
#                             print(t1.shape)
#                             path2 = '/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] + '/MOTOR/RL/' + al + '_' + sub1[-1] + '.mat'
#                             sio.savemat(path2, {'ROI': t1})
#                             print(path2 + 'done!')
#
#                     except:
#                         pass
#             except:
#                 pass
#     except:
#         pass





''' atlas + language task '''
# data_list=['tfMRI_LANGUAGE_LR','tfMRI_LANGUAGE_RL']
# # atlas_list=['AAL_116','Yeo7','Yeo17','HO_111','Destrieux_148']
# atlas_list=['Shen_1mm','Shen_2mm']
#
#
# for fn in data_list:
#     try:
#
#
#         for al in atlas_list:
#             try:
#
#
#
#                 for sub in glob('/mnt/hcp/HCP/*'):
#                     try:
#
#                         sub1 = sub.split('/')
#
#                         if fn == 'tfMRI_LANGUAGE_LR':
#                             file = '/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] +'/LANGUAGE/LR/' + al + '_' + sub1[-1] +'.mat'
#
#                             if os.path.isfile(file):
#                                 print('HCP_extract/' + sub1[-1] +'/LANGUAGE/LR/' + al + '_' + sub1[-1] +'.mat', "파일이 존재합니다.")
#                                 continue
#
#                         if fn == 'tfMRI_LANGUAGE_RL':
#                             file = '/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] +'/LANGUAGE/RL/' + al + '_' + sub1[-1] +'.mat'
#
#                             if os.path.isfile(file):
#                                 print('HCP_extract/' + sub1[-1] +'/LANGUAGE/RL/' + al + '_' + sub1[-1] +'.mat', "파일이 존재합니다.")
#                                 continue
#
#                         if al == 'AAL_116':
#                             atlas = nib.load('/media/12T/practice/atlas/aal_roi_atlas.nii.gz')
#                         elif al == 'Yeo7':
#                             atlas = nib.load('/media/12T/practice/atlas/yeo_atlas_2mm.nii.gz')
#                         elif al == 'Yeo17':
#                             atlas = nib.load('/media/12T/practice/atlas/yeo_atlas_2mm_17.nii.gz')
#                         elif al == 'HO_111':
#                             atlas = nib.load('/media/12T/practice/atlas/ho_roi_atlas.nii.gz')
#                         elif al == 'Shen_1mm':
#                             atlas = nib.load('/media/12T/practice/atlas/shen_1mm_268_parcellation.nii.gz')
#                         elif al == 'Shen_2mm':
#                             atlas = nib.load('/media/12T/practice/atlas/shen_2mm_268_parcellation.nii.gz')
#                         else:
#                             atlas = nib.load('/media/12T/practice/atlas/destrieux_2009/destrieux2009_rois_lateralized.nii.gz')
#
#                         masker = NiftiLabelsMasker(labels_img=atlas, standardize=True)
#
#                         if fn == 'tfMRI_LANGUAGE_LR':
#                             n1 = nib.load('/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] +'/LANGUAGE/LR/tfMRI_LANGUAGE_LR.nii.gz')
#                             t1 = masker.fit_transform(n1)
#                             print(t1.shape)
#                             path2 = '/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] + '/LANGUAGE/LR/' + al + '_' + sub1[-1] +'.mat'
#                             sio.savemat(path2, {'ROI': t1})
#                             print(path2 + 'done!')
#                         else:
#                             n1 = nib.load('/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] +'/LANGUAGE/RL/tfMRI_LANGUAGE_RL.nii.gz')
#                             t1 = masker.fit_transform(n1)
#                             print(t1.shape)
#                             path2 = '/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] + '/LANGUAGE/RL/' + al + '_' + sub1[-1] + '.mat'
#                             sio.savemat(path2, {'ROI': t1})
#                             print(path2 + 'done!')
#
#                     except:
#                         pass
#             except:
#                 pass
#     except:
#         pass








''' atlas + gambling '''

# data_list=['tfMRI_GAMBLING_LR','tfMRI_GAMBLING_RL']
# # atlas_list=['AAL_116','Yeo7','Yeo17','HO_111','Destrieux_148']
# atlas_list=['Shen_1mm','Shen_2mm']
#
# for fn in data_list:
#     try:
#
#
#         for al in atlas_list:
#             try:
#
#
#
#                 for sub in glob('/mnt/hcp/HCP/*'):
#                     try:
#
#                         sub1 = sub.split('/')
#
#                         if fn == 'tfMRI_GAMBLING_LR':
#                             file = '/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] +'/GAMBLING/LR/' + al + '_' + sub1[-1] +'.mat'
#
#                             if os.path.isfile(file):
#                                 print('HCP_extract/' + sub1[-1] +'/GAMBLING/LR/' + al + '_' + sub1[-1] +'.mat', "파일이 존재합니다.")
#                                 continue
#
#                         if fn == 'tfMRI_GAMBLING_RL':
#                             file = '/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] +'/GAMBLING/RL/' + al + '_' + sub1[-1] +'.mat'
#
#                             if os.path.isfile(file):
#                                 print('HCP_extract/' + sub1[-1] +'/GAMBLING/RL/' + al + '_' + sub1[-1] +'.mat', "파일이 존재합니다.")
#                                 continue
#
#                         if al == 'AAL_116':
#                             atlas = nib.load('/media/12T/practice/atlas/aal_roi_atlas.nii.gz')
#                         elif al == 'Yeo7':
#                             atlas = nib.load('/media/12T/practice/atlas/yeo_atlas_2mm.nii.gz')
#                         elif al == 'Yeo17':
#                             atlas = nib.load('/media/12T/practice/atlas/yeo_atlas_2mm_17.nii.gz')
#                         elif al == 'HO_111':
#                             atlas = nib.load('/media/12T/practice/atlas/ho_roi_atlas.nii.gz')
#                         elif al == 'Shen_1mm':
#                             atlas = nib.load('/media/12T/practice/atlas/shen_1mm_268_parcellation.nii.gz')
#                         elif al == 'Shen_2mm':
#                             atlas = nib.load('/media/12T/practice/atlas/shen_2mm_268_parcellation.nii.gz')
#                         else:
#                             atlas = nib.load('/media/12T/practice/atlas/destrieux_2009/destrieux2009_rois_lateralized.nii.gz')
#
#                         masker = NiftiLabelsMasker(labels_img=atlas, standardize=True)
#
#                         if fn == 'tfMRI_GAMBLING_LR':
#                             n1 = nib.load('/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] +'/GAMBLING/LR/tfMRI_GAMBLING_LR.nii.gz')
#                             t1 = masker.fit_transform(n1)
#                             print(t1.shape)
#                             path2 = '/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] + '/GAMBLING/LR/' + al + '_' + sub1[-1] +'.mat'
#                             sio.savemat(path2, {'ROI': t1})
#                             print(path2 + 'done!')
#                         else:
#                             n1 = nib.load('/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] +'/GAMBLING/RL/tfMRI_GAMBLING_RL.nii.gz')
#                             t1 = masker.fit_transform(n1)
#                             print(t1.shape)
#                             path2 = '/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] + '/GAMBLING/RL/' + al + '_' + sub1[-1] + '.mat'
#                             sio.savemat(path2, {'ROI': t1})
#                             print(path2 + 'done!')
#
#                     except:
#                         pass
#             except:
#                 pass
#     except:
#         pass













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

# data_list=['tfMRI_MOTOR_LR','tfMRI_MOTOR_RL']
# atlas_list=['Shen_1mm','Shen_2mm']
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
#                         if al == 'yeo7_network':
#                             atlas = nib.load('/media/12T/practice/atlas/yeo_2011/Yeo_JNeurophysiol11_MNI152/Yeo2011_7Networks_MNI152_FreeSurferConformed1mm.nii.gz')
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






''' atlas + relational '''

data_list=['tfMRI_RELATIONAL_LR','tfMRI_RELATIONAL_RL']
# atlas_list=['AAL_116','Yeo7','Yeo17','HO_111','Destrieux_148']
atlas_list=['Shen_1mm','Shen_2mm']

for fn in data_list:
    try:


        for al in atlas_list:
            try:



                for sub in glob('/mnt/hcp/HCP/*'):
                    try:

                        sub1 = sub.split('/')

                        if fn == 'tfMRI_RELATIONAL_LR':
                            file = '/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] +'/RELATIONAL/LR/' + al + '_' + sub1[-1] +'.mat'

                            if os.path.isfile(file):
                                print('HCP_extract/' + sub1[-1] +'/RELATIONAL/LR/' + al + '_' + sub1[-1] +'.mat', "파일이 존재합니다.")
                                continue

                        if fn == 'tfMRI_RELATIONAL_RL':
                            file = '/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] +'/RELATIONAL/RL/' + al + '_' + sub1[-1] +'.mat'

                            if os.path.isfile(file):
                                print('HCP_extract/' + sub1[-1] +'/RELATIONAL/RL/' + al + '_' + sub1[-1] +'.mat', "파일이 존재합니다.")
                                continue

                        if al == 'AAL_116':
                            atlas = nib.load('/media/12T/practice/atlas/aal_roi_atlas.nii.gz')
                        elif al == 'Yeo7':
                            atlas = nib.load('/media/12T/practice/atlas/yeo_atlas_2mm.nii.gz')
                        elif al == 'Yeo17':
                            atlas = nib.load('/media/12T/practice/atlas/yeo_atlas_2mm_17.nii.gz')
                        elif al == 'HO_111':
                            atlas = nib.load('/media/12T/practice/atlas/ho_roi_atlas.nii.gz')
                        elif al == 'Shen_1mm':
                            atlas = nib.load('/media/12T/practice/atlas/shen_1mm_268_parcellation.nii.gz')
                        elif al == 'Shen_2mm':
                            atlas = nib.load('/media/12T/practice/atlas/shen_2mm_268_parcellation.nii.gz')
                        else:
                            atlas = nib.load('/media/12T/practice/atlas/destrieux_2009/destrieux2009_rois_lateralized.nii.gz')

                        masker = NiftiLabelsMasker(labels_img=atlas, standardize=True)

                        if fn == 'tfMRI_RELATIONAL_LR':
                            n1 = nib.load('/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] +'/RELATIONAL/LR/tfMRI_RELATIONAL_LR.nii.gz')
                            t1 = masker.fit_transform(n1)
                            print(t1.shape)
                            path2 = '/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] + '/RELATIONAL/LR/' + al + '_' + sub1[-1] +'.mat'
                            sio.savemat(path2, {'ROI': t1})
                            print(path2 + 'done!')
                        else:
                            n1 = nib.load('/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] +'/RELATIONAL/RL/tfMRI_RELATIONAL_RL.nii.gz')
                            t1 = masker.fit_transform(n1)
                            print(t1.shape)
                            path2 = '/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] + '/RELATIONAL/RL/' + al + '_' + sub1[-1] + '.mat'
                            sio.savemat(path2, {'ROI': t1})
                            print(path2 + 'done!')

                    except:
                        pass
            except:
                pass
    except:
        pass










''' atlas + emotion '''

# data_list=['tfMRI_EMOTION_LR','tfMRI_EMOTION_RL']
# # atlas_list=['AAL_116','Yeo7','Yeo17','HO_111','Destrieux_148']
# atlas_list=['Shen_1mm','Shen_2mm']
#
# for fn in data_list:
#     try:
#
#
#         for al in atlas_list:
#             try:
#
#
#
#                 for sub in glob('/mnt/hcp/HCP/*'):
#                     try:
#
#                         sub1 = sub.split('/')
#
#                         if fn == 'tfMRI_EMOTION_LR':
#                             file = '/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] +'/EMOTION/LR/' + al + '_' + sub1[-1] +'.mat'
#
#                             if os.path.isfile(file):
#                                 print('HCP_extract/' + sub1[-1] +'/EMOTION/LR/' + al + '_' + sub1[-1] +'.mat', "파일이 존재합니다.")
#                                 continue
#
#                         if fn == 'tfMRI_EMOTION_RL':
#                             file = '/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] +'/EMOTION/RL/' + al + '_' + sub1[-1] +'.mat'
#
#                             if os.path.isfile(file):
#                                 print('HCP_extract/' + sub1[-1] +'/EMOTION/RL/' + al + '_' + sub1[-1] +'.mat', "파일이 존재합니다.")
#                                 continue
#
#                         if al == 'AAL_116':
#                             atlas = nib.load('/media/12T/practice/atlas/aal_roi_atlas.nii.gz')
#                         elif al == 'Yeo7':
#                             atlas = nib.load('/media/12T/practice/atlas/yeo_atlas_2mm.nii.gz')
#                         elif al == 'Yeo17':
#                             atlas = nib.load('/media/12T/practice/atlas/yeo_atlas_2mm_17.nii.gz')
#                         elif al == 'HO_111':
#                             atlas = nib.load('/media/12T/practice/atlas/ho_roi_atlas.nii.gz')
#                         elif al == 'Shen_1mm':
#                             atlas = nib.load('/media/12T/practice/atlas/shen_1mm_268_parcellation.nii.gz')
#                         elif al == 'Shen_2mm':
#                             atlas = nib.load('/media/12T/practice/atlas/shen_2mm_268_parcellation.nii.gz')
#                         else:
#                             atlas = nib.load('/media/12T/practice/atlas/destrieux_2009/destrieux2009_rois_lateralized.nii.gz')
#
#                         masker = NiftiLabelsMasker(labels_img=atlas, standardize=True)
#
#                         if fn == 'tfMRI_EMOTION_LR':
#                             n1 = nib.load('/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] +'/EMOTION/LR/tfMRI_EMOTION_LR.nii.gz')
#                             t1 = masker.fit_transform(n1)
#                             print(t1.shape)
#                             path2 = '/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] + '/EMOTION/LR/' + al + '_' + sub1[-1] +'.mat'
#                             sio.savemat(path2, {'ROI': t1})
#                             print(path2 + 'done!')
#                         else:
#                             n1 = nib.load('/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] +'/EMOTION/RL/tfMRI_EMOTION_RL.nii.gz')
#                             t1 = masker.fit_transform(n1)
#                             print(t1.shape)
#                             path2 = '/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] + '/EMOTION/RL/' + al + '_' + sub1[-1] + '.mat'
#                             sio.savemat(path2, {'ROI': t1})
#                             print(path2 + 'done!')
#
#                     except:
#                         pass
#             except:
#                 pass
#     except:
#         pass




''' atlas + social '''

# data_list=['tfMRI_SOCIAL_LR','tfMRI_SOCIAL_RL']
# # atlas_list=['AAL_116','Yeo7','Yeo17','HO_111','Destrieux_148']
# atlas_list=['Shen_1mm','Shen_2mm']
#
#
# for fn in data_list:
#     try:
#
#
#         for al in atlas_list:
#             try:
#
#
#
#                 for sub in glob('/mnt/hcp/HCP/*'):
#                     try:
#
#                         sub1 = sub.split('/')
#
#                         if fn == 'tfMRI_SOCIAL_LR':
#                             file = '/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] +'/SOCIAL/LR/' + al + '_' + sub1[-1] +'.mat'
#
#                             if os.path.isfile(file):
#                                 print('HCP_extract/' + sub1[-1] +'/SOCIAL/LR/' + al + '_' + sub1[-1] +'.mat', "파일이 존재합니다.")
#                                 continue
#
#                         if fn == 'tfMRI_SOCIAL_RL':
#                             file = '/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] +'/SOCIAL/RL/' + al + '_' + sub1[-1] +'.mat'
#
#                             if os.path.isfile(file):
#                                 print('HCP_extract/' + sub1[-1] +'/SOCIAL/RL/' + al + '_' + sub1[-1] +'.mat', "파일이 존재합니다.")
#                                 continue
#
#                         if al == 'AAL_116':
#                             atlas = nib.load('/media/12T/practice/atlas/aal_roi_atlas.nii.gz')
#                         elif al == 'Yeo7':
#                             atlas = nib.load('/media/12T/practice/atlas/yeo_atlas_2mm.nii.gz')
#                         elif al == 'Yeo17':
#                             atlas = nib.load('/media/12T/practice/atlas/yeo_atlas_2mm_17.nii.gz')
#                         elif al == 'HO_111':
#                             atlas = nib.load('/media/12T/practice/atlas/ho_roi_atlas.nii.gz')
#                         elif al == 'Shen_1mm':
#                             atlas = nib.load('/media/12T/practice/atlas/shen_1mm_268_parcellation.nii.gz')
#                         elif al == 'Shen_2mm':
#                             atlas = nib.load('/media/12T/practice/atlas/shen_2mm_268_parcellation.nii.gz')
#                         else:
#                             atlas = nib.load('/media/12T/practice/atlas/destrieux_2009/destrieux2009_rois_lateralized.nii.gz')
#
#                         masker = NiftiLabelsMasker(labels_img=atlas, standardize=True)
#
#                         if fn == 'tfMRI_SOCIAL_LR':
#                             n1 = nib.load('/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] +'/SOCIAL/LR/tfMRI_SOCIAL_LR.nii.gz')
#                             t1 = masker.fit_transform(n1)
#                             print(t1.shape)
#                             path2 = '/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] + '/SOCIAL/LR/' + al + '_' + sub1[-1] +'.mat'
#                             sio.savemat(path2, {'ROI': t1})
#                             print(path2 + 'done!')
#                         else:
#                             n1 = nib.load('/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] +'/SOCIAL/RL/tfMRI_SOCIAL_RL.nii.gz')
#                             t1 = masker.fit_transform(n1)
#                             print(t1.shape)
#                             path2 = '/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] + '/SOCIAL/RL/' + al + '_' + sub1[-1] + '.mat'
#                             sio.savemat(path2, {'ROI': t1})
#                             print(path2 + 'done!')
#
#                     except:
#                         pass
#             except:
#                 pass
#     except:
#         pass





''' atlas + wm '''

# data_list=['tfMRI_WM_LR','tfMRI_WM_RL']
# atlas_list=['AAL_116','Yeo7','Yeo17','HO_111','Destrieux_148','Shen_1mm','Shen_2mm']
# # atlas_list=['Shen_1mm','Shen_2mm']
#
# for fn in data_list:
#     try:
#
#
#         for al in atlas_list:
#             try:
#
#
#
#                 for sub in glob('/mnt/hcp/HCP/*'):
#                     try:
#
#                         sub1 = sub.split('/')
#
#                         if fn == 'tfMRI_WM_LR':
#                             file = '/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] +'/WM/LR/' + al + '_' + sub1[-1] +'.mat'
#
#                             if os.path.isfile(file):
#                                 print('HCP_extract/' + sub1[-1] +'/WM/LR/' + al + '_' + sub1[-1] +'.mat', "파일이 존재합니다.")
#                                 continue
#
#                         if fn == 'tfMRI_WM_RL':
#                             file = '/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] +'/WM/RL/' + al + '_' + sub1[-1] +'.mat'
#
#                             if os.path.isfile(file):
#                                 print('HCP_extract/' + sub1[-1] +'/WM/RL/' + al + '_' + sub1[-1] +'.mat', "파일이 존재합니다.")
#                                 continue
#
#                         if al == 'AAL_116':
#                             atlas = nib.load('/media/12T/practice/atlas/aal_roi_atlas.nii.gz')
#                         elif al == 'Yeo7':
#                             atlas = nib.load('/media/12T/practice/atlas/yeo_atlas_2mm.nii.gz')
#                         elif al == 'Yeo17':
#                             atlas = nib.load('/media/12T/practice/atlas/yeo_atlas_2mm_17.nii.gz')
#                         elif al == 'HO_111':
#                             atlas = nib.load('/media/12T/practice/atlas/ho_roi_atlas.nii.gz')
#                         elif al == 'Shen_1mm':
#                             atlas = nib.load('/media/12T/practice/atlas/shen_1mm_268_parcellation.nii.gz')
#                         elif al == 'Shen_2mm':
#                             atlas = nib.load('/media/12T/practice/atlas/shen_2mm_268_parcellation.nii.gz')
#                         else:
#                             atlas = nib.load('/media/12T/practice/atlas/destrieux_2009/destrieux2009_rois_lateralized.nii.gz')
#
#                         masker = NiftiLabelsMasker(labels_img=atlas, standardize=True)
#
#                         if fn == 'tfMRI_WM_LR':
#                             n1 = nib.load('/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] +'/WM/LR/tfMRI_WM_LR.nii.gz')
#                             t1 = masker.fit_transform(n1)
#                             print(t1.shape)
#                             path2 = '/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] + '/WM/LR/' + al + '_' + sub1[-1] +'.mat'
#                             sio.savemat(path2, {'ROI': t1})
#                             print(path2 + 'done!')
#                         else:
#                             n1 = nib.load('/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] +'/WM/RL/tfMRI_WM_RL.nii.gz')
#                             t1 = masker.fit_transform(n1)
#                             print(t1.shape)
#                             path2 = '/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] + '/WM/RL/' + al + '_' + sub1[-1] + '.mat'
#                             sio.savemat(path2, {'ROI': t1})
#                             print(path2 + 'done!')
#
#                     except:
#                         pass
#             except:
#                 pass
#     except:
#         pass




''' remove'''
# data_list=['tfMRI_EMOTION_LR','tfMRI_EMOTION_RL']
# atlas_list=['Shen_1mm','Shen_2mm']
#
# for fn in data_list:
#     try:
#
#
#         for al in atlas_list:
#             try:
#
#
#
#                 for sub in glob('/mnt/hcp/HCP/*'):
#                     try:
#
#                         sub1 = sub.split('/')
#
#                         if fn == 'tfMRI_EMOTION_LR':
#                             file = '/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] +'/EMOTION/LR/' + al + '_' + sub1[-1] +'.mat'
#
#                             if os.path.isfile(file):
#                                 os.remove(file)
#                                 print('HCP_extract/' + sub1[-1] +'/EMOTION/LR/' + al + '_' + sub1[-1] +'.mat', "파일을 삭제하였습니다.")
#                                 continue
#
#                         if fn == 'tfMRI_EMOTION_RL':
#                             file = '/mnt/hcp/homes/jwlee/HCP_extract/' + sub1[-1] +'/EMOTION/RL/' + al + '_' + sub1[-1] +'.mat'
#
#                             if os.path.isfile(file):
#                                 os.remove(file)
#                                 print('HCP_extract/' + sub1[-1] +'/EMOTION/RL/' + al + '_' + sub1[-1] +'.mat', "파일을 삭제하였습니다.")
#                                 continue
#
#
#                     except:
#                         pass
#             except:
#                 pass
#     except:
#         pass