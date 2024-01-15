# HCP task fMRI time labeling

import os
import numpy as np
from glob import glob
import nibabel as nib
import scipy.io as sio

frames = 284
Tr = 0.72
whole_sc = frames * Tr


data_path = '/mnt/hcp/homes/jwlee/HCP_train_data/tfMRI_MOTOR_LR/4d/AAL_116/'
dir_path = '/mnt/hcp/HCP/'
list = sorted(os.listdir(data_path))
print(list)

def split_task_only_LR():
    for i in range(len(list)):
        sub = list[i].split('.')[0]

        n1 = sio.loadmat(data_path + list[i])
        n1 = n1['ROI']

        sub_split_data = dict(rh=[], lh=[], rf=[], lf=[], t=[], fix=[], cue=[])
        for fn in glob(dir_path + sub +'/MNINonLinear/Results/tfMRI_MOTOR_LR/EVs/*'):

            #print(fn)
            file = fn.split('/')
            if file[-1] == 'cue.txt':
                t_cue = np.loadtxt(dir_path + list[i].split('.')[0]+'/MNINonLinear/Results/tfMRI_MOTOR_LR/EVs/'+file[-1], delimiter='\t')  # numpy.ndarray(10,3)

            if file[-1] == 't.txt':
                t_t = np.loadtxt(dir_path + list[i].split('.')[0]+'/MNINonLinear/Results/tfMRI_MOTOR_LR/EVs/'+file[-1], delimiter='\t')  # numpy.ndarray(2,3)

            if file[-1] == 'rh.txt':
                t_rh = np.loadtxt(dir_path + list[i].split('.')[0]+'/MNINonLinear/Results/tfMRI_MOTOR_LR/EVs/'+file[-1], delimiter='\t')  # numpy.ndarray(2,3)

            if file[-1] == 'lh.txt':
                t_lh = np.loadtxt(dir_path + list[i].split('.')[0]+'/MNINonLinear/Results/tfMRI_MOTOR_LR/EVs/'+file[-1], delimiter='\t')  # numpy.ndarray(2,3)

            if file[-1] == 'rf.txt':
                t_rf = np.loadtxt(dir_path + list[i].split('.')[0]+'/MNINonLinear/Results/tfMRI_MOTOR_LR/EVs/'+file[-1], delimiter='\t')  # numpy.ndarray(2,3)

            if file[-1] == 'lf.txt':
                t_lf = np.loadtxt(dir_path + list[i].split('.')[0]+'/MNINonLinear/Results/tfMRI_MOTOR_LR/EVs/'+file[-1], delimiter='\t')  # numpy.ndarray(2,3)

            if file[-1] == 'Sync.txt':
                t_fix = np.array([[116,137],[200,221],[263,284]]) # numpy.ndarray(3,2)





        label_list = [0 for i in range(284)]

#        print(np.round(t_cue[0, 0] / Tr).astype(int))
        label_list[:np.round(t_cue[0, 0] / Tr).astype(int)] = [7 for i in range(np.round(t_cue[0, 0] / Tr).astype(int))]

        for cue_id in range(10):
            label_list[np.round(t_cue[cue_id, 0] / Tr).astype(int):np.round((t_cue[cue_id, 0] + t_cue[cue_id, 1]) / Tr).astype(int)] = [0 for i in range(np.round((t_cue[cue_id, 0] + t_cue[cue_id, 1]) / Tr).astype(int) - np.round(t_cue[cue_id, 0] / Tr).astype(int))]

        for t_id in range(2):
            label_list[np.round(t_t[t_id,0]/Tr).astype(int):np.round((t_t[t_id,0]+t_t[t_id,1])/Tr).astype(int)] = [3 for i in range(np.round((t_t[t_id,0]+t_t[t_id,1])/Tr).astype(int) - np.round(t_t[t_id,0]/Tr).astype(int) )]

        for rh_id in range(2):
            label_list[np.round(t_rh[rh_id,0]/Tr).astype(int):np.round((t_rh[rh_id,0]+t_rh[rh_id,1])/Tr).astype(int)] = [1 for i in range(np.round((t_rh[rh_id,0]+t_rh[rh_id,1])/Tr).astype(int) - np.round(t_rh[rh_id,0]/Tr).astype(int) )]

        for lh_id in range(2):
            label_list[np.round(t_lh[lh_id,0] / Tr).astype(int):np.round((t_lh[lh_id, 0] + t_lh[lh_id, 1]) / Tr).astype(int)] = [5 for i in range(np.round((t_lh[lh_id, 0] + t_lh[lh_id, 1]) / Tr).astype(int) - np.round(t_lh[lh_id,0] / Tr).astype(int))]

        for rf_id in range(2):
            label_list[np.round(t_rf[rf_id,0] / Tr).astype(int):np.round((t_rf[rf_id, 0] + t_rf[rf_id, 1]) / Tr).astype(int)] = [4 for i in range(np.round((t_rf[rf_id, 0] + t_rf[rf_id, 1]) / Tr).astype(int) - np.round(t_rf[rf_id,0] / Tr).astype(int))]

        for lf_id in range(2):
            label_list[np.round(t_lf[lf_id,0] / Tr).astype(int):np.round((t_lf[lf_id, 0] + t_lf[lf_id, 1]) / Tr).astype(int)] = [2 for i in range(np.round((t_lf[lf_id, 0] + t_lf[lf_id, 1]) / Tr).astype(int) - np.round(t_lf[lf_id,0] / Tr).astype(int))]

        for fix_id in range(3):
            label_list[np.round(t_fix[fix_id, 0]).astype(int):np.round(t_fix[fix_id, 1]).astype(int)] = [6 for i in range(np.round(t_fix[fix_id, 1]).astype(int) - np.round(t_fix[fix_id, 0]).astype(int))]


#        print(label_list)

        label_list = np.array(label_list)
        label_list = np.expand_dims(label_list, axis=0)

        if i == 0:
            label_list_LR = label_list

        else:
            label_list_LR = np.concatenate((label_list_LR, label_list))


    print(label_list_LR.shape)

    return label_list_LR


data_path2 = '/mnt/hcp/homes/jwlee/HCP_train_data/tfMRI_MOTOR_RL/4d/AAL_116/'
dir_path2 = '/mnt/hcp/HCP/'
list2 = sorted(os.listdir(data_path2))
print(list2)

def split_task_only_RL():
    for i in range(len(list2)):
        sub = list2[i].split('.')[0]

        n1 = sio.loadmat(data_path2 + list2[i])
        n1 = n1['ROI']

        sub_split_data = dict(rh=[], lh=[], rf=[], lf=[], t=[], fix=[], cue=[])
        for fn in glob(dir_path2 + sub +'/MNINonLinear/Results/tfMRI_MOTOR_RL/EVs/*'):

            #print(fn)
            file = fn.split('/')
            if file[-1] == 'cue.txt':
                t_cue = np.loadtxt(dir_path2 + list2[i].split('.')[0]+'/MNINonLinear/Results/tfMRI_MOTOR_RL/EVs/'+file[-1], delimiter='\t')  # numpy.ndarray(10,3)

            if file[-1] == 't.txt':
                t_t = np.loadtxt(dir_path2 + list2[i].split('.')[0]+'/MNINonLinear/Results/tfMRI_MOTOR_RL/EVs/'+file[-1], delimiter='\t')  # numpy.ndarray(2,3)

            if file[-1] == 'rh.txt':
                t_rh = np.loadtxt(dir_path2 + list2[i].split('.')[0]+'/MNINonLinear/Results/tfMRI_MOTOR_RL/EVs/'+file[-1], delimiter='\t')  # numpy.ndarray(2,3)

            if file[-1] == 'lh.txt':
                t_lh = np.loadtxt(dir_path2 + list2[i].split('.')[0]+'/MNINonLinear/Results/tfMRI_MOTOR_RL/EVs/'+file[-1], delimiter='\t')  # numpy.ndarray(2,3)

            if file[-1] == 'rf.txt':
                t_rf = np.loadtxt(dir_path2 + list2[i].split('.')[0]+'/MNINonLinear/Results/tfMRI_MOTOR_RL/EVs/'+file[-1], delimiter='\t')  # numpy.ndarray(2,3)

            if file[-1] == 'lf.txt':
                t_lf = np.loadtxt(dir_path2 + list2[i].split('.')[0]+'/MNINonLinear/Results/tfMRI_MOTOR_RL/EVs/'+file[-1], delimiter='\t')  # numpy.ndarray(2,3)

            if file[-1] == 'Sync.txt':
                t_fix = np.array([[116,137],[200,221],[263,284]]) # numpy.ndarray(3,2)





        label_list = [0 for i in range(284)]

#        print(np.round(t_cue[0, 0] / Tr).astype(int))
        label_list[:np.round(t_cue[0, 0] / Tr).astype(int)] = [7 for i in range(np.round(t_cue[0, 0] / Tr).astype(int))]

        for cue_id in range(10):
            label_list[np.round(t_cue[cue_id, 0] / Tr).astype(int):np.round((t_cue[cue_id, 0] + t_cue[cue_id, 1]) / Tr).astype(int)] = [0 for i in range(np.round((t_cue[cue_id, 0] + t_cue[cue_id, 1]) / Tr).astype(int) - np.round(t_cue[cue_id, 0] / Tr).astype(int))]

        for t_id in range(2):
            label_list[np.round(t_t[t_id,0]/Tr).astype(int):np.round((t_t[t_id,0]+t_t[t_id,1])/Tr).astype(int)] = [3 for i in range(np.round((t_t[t_id,0]+t_t[t_id,1])/Tr).astype(int) - np.round(t_t[t_id,0]/Tr).astype(int) )]

        for rh_id in range(2):
            label_list[np.round(t_rh[rh_id,0]/Tr).astype(int):np.round((t_rh[rh_id,0]+t_rh[rh_id,1])/Tr).astype(int)] = [1 for i in range(np.round((t_rh[rh_id,0]+t_rh[rh_id,1])/Tr).astype(int) - np.round(t_rh[rh_id,0]/Tr).astype(int) )]

        for lh_id in range(2):
            label_list[np.round(t_lh[lh_id,0] / Tr).astype(int):np.round((t_lh[lh_id, 0] + t_lh[lh_id, 1]) / Tr).astype(int)] = [5 for i in range(np.round((t_lh[lh_id, 0] + t_lh[lh_id, 1]) / Tr).astype(int) - np.round(t_lh[lh_id,0] / Tr).astype(int))]

        for rf_id in range(2):
            label_list[np.round(t_rf[rf_id,0] / Tr).astype(int):np.round((t_rf[rf_id, 0] + t_rf[rf_id, 1]) / Tr).astype(int)] = [4 for i in range(np.round((t_rf[rf_id, 0] + t_rf[rf_id, 1]) / Tr).astype(int) - np.round(t_rf[rf_id,0] / Tr).astype(int))]

        for lf_id in range(2):
            label_list[np.round(t_lf[lf_id,0] / Tr).astype(int):np.round((t_lf[lf_id, 0] + t_lf[lf_id, 1]) / Tr).astype(int)] = [2 for i in range(np.round((t_lf[lf_id, 0] + t_lf[lf_id, 1]) / Tr).astype(int) - np.round(t_lf[lf_id,0] / Tr).astype(int))]

        for fix_id in range(3):
            label_list[np.round(t_fix[fix_id, 0]).astype(int):np.round(t_fix[fix_id, 1]).astype(int)] = [6 for i in range(np.round(t_fix[fix_id, 1]).astype(int) - np.round(t_fix[fix_id, 0]).astype(int))]


#        print(label_list)

        label_list = np.array(label_list)
        label_list = np.expand_dims(label_list, axis=0)

        if i == 0:
            label_list_RL = label_list

        else:
            label_list_RL = np.concatenate((label_list_RL, label_list))


    print(label_list_RL.shape)
    return label_list_RL






np.savez_compressed('/media/12T/practice/LR_DICT/hcp_motor_label', label_list_LR=split_task_only_LR(), label_list_RL=split_task_only_RL())








