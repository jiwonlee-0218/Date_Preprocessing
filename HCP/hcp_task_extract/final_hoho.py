# directory last (3)

import nibabel as nib
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
import scipy.io as sio
from glob import glob
import os
import os.path
import numpy as np

task_list = ['LH', 'LF', 'RH', 'RF', 'T', 'Cue']

p = np.load('/media/12T/practice/LR_DICT/ho_hcp_motor.npz')
data = p['ho_tfMRI_MOTOR_RL']
data_label = p['ho_label_MOTOR_RL']
# print(data.shape[0]) # 1080

dir_path = '/mnt/hcp/HCP/'
LR_path = '/mnt/hcp/homes/jwlee/HCP_train_data/tfMRI_MOTOR_RL/4d/ho/*'

Tr = 0.72
count = 0










for i in sorted(glob(LR_path)):
    ''' 디렉토리 생성 '''
    name = i.split('/')[-1]
    name = name.split('.')[0]
    os.makedirs('/media/12T/practice/LR_DICT/ho_subject/MOTOR_RL/' + name, exist_ok=True)

    for fn in glob(dir_path + name + '/MNINonLinear/Results/tfMRI_MOTOR_RL/EVs/*'):

            # print(fn)
        file = fn.split('/')
        if file[-1] == 'cue.txt':
            t_cue = np.loadtxt(dir_path + name + '/MNINonLinear/Results/tfMRI_MOTOR_RL/EVs/' + file[-1],
                                   delimiter='\t')  # numpy.ndarray(10,3)

        if file[-1] == 't.txt':
            t_t = np.loadtxt(dir_path + name + '/MNINonLinear/Results/tfMRI_MOTOR_RL/EVs/' + file[-1],
                                 delimiter='\t')  # numpy.ndarray(2,3)

        if file[-1] == 'rh.txt':
            t_rh = np.loadtxt(dir_path + name + '/MNINonLinear/Results/tfMRI_MOTOR_RL/EVs/' + file[-1],
                                  delimiter='\t')  # numpy.ndarray(2,3)

        if file[-1] == 'lh.txt':
            t_lh = np.loadtxt(dir_path + name + '/MNINonLinear/Results/tfMRI_MOTOR_RL/EVs/' + file[-1],
                                  delimiter='\t')  # numpy.ndarray(2,3)

        if file[-1] == 'rf.txt':
            t_rf = np.loadtxt(dir_path + name + '/MNINonLinear/Results/tfMRI_MOTOR_RL/EVs/' + file[-1],
                                  delimiter='\t')  # numpy.ndarray(2,3)

        if file[-1] == 'lf.txt':
            t_lf = np.loadtxt(dir_path + name + '/MNINonLinear/Results/tfMRI_MOTOR_RL/EVs/' + file[-1],
                                  delimiter='\t')  # numpy.ndarray(2,3)

        if file[-1] == 'Sync.txt':
            t_fix = np.array([[116, 137], [200, 221], [263, 284]])  # numpy.ndarray(3,2)






    for tl in task_list: # ['LH', 'LF', 'RH', 'RF', 'T', 'Cue']
        # os.makedirs('/media/12T/practice/LR_DICT/ho_subject/MOTOR_LR/' + name + '/' + tl, exist_ok=True)
        print(name)
        a = sio.loadmat(i)
        a = a['ROI']


        if tl == 'LH':
            for lh_id in range(2):
                if lh_id == 0:
                    LH1 = a[np.round(t_lh[lh_id, 0] / Tr).astype(int):np.round((t_lh[lh_id, 0] + t_lh[lh_id, 1]) / Tr).astype(int)]
                    LH1_label = data_label[count, np.round(t_lh[lh_id, 0] / Tr).astype(int):np.round((t_lh[lh_id, 0] + t_lh[lh_id, 1]) / Tr).astype(int)]
                if lh_id == 1:
                    LH2 = a[np.round(t_lh[lh_id, 0] / Tr).astype(int):np.round((t_lh[lh_id, 0] + t_lh[lh_id, 1]) / Tr).astype(int)]
                    LH2_label = data_label[count, np.round(t_lh[lh_id, 0] / Tr).astype(int):np.round((t_lh[lh_id, 0] + t_lh[lh_id, 1]) / Tr).astype(int)]

            np.savez_compressed('/media/12T/practice/LR_DICT/ho_subject/MOTOR_RL/' + name + '/LH', LH1_label=LH1_label, LH1=LH1, LH2=LH2, LH2_label=LH2_label)

        if tl == 'LF':
            for lf_id in range(2):
                if lf_id == 0:
                    LF1 = a[np.round(t_lf[lf_id,0] / Tr).astype(int):np.round((t_lf[lf_id, 0] + t_lf[lf_id, 1]) / Tr).astype(int)]
                    LF1_label = data_label[count, np.round(t_lf[lf_id,0] / Tr).astype(int):np.round((t_lf[lf_id, 0] + t_lf[lf_id, 1]) / Tr).astype(int)]
                if lf_id == 1:
                    LF2 = a[np.round(t_lf[lf_id,0] / Tr).astype(int):np.round((t_lf[lf_id, 0] + t_lf[lf_id, 1]) / Tr).astype(int)]
                    LF2_label = data_label[count, np.round(t_lf[lf_id,0] / Tr).astype(int):np.round((t_lf[lf_id, 0] + t_lf[lf_id, 1]) / Tr).astype(int)]

            np.savez_compressed('/media/12T/practice/LR_DICT/ho_subject/MOTOR_RL/' + name + '/LF', LF1_label= LF1_label, LF1=LF1,LF2=LF2, LF2_label=LF2_label)



        if tl == 'RH':
            for rh_id in range(2):
                if rh_id == 0:
                    RH1 = a[np.round(t_rh[rh_id,0]/Tr).astype(int):np.round((t_rh[rh_id,0]+t_rh[rh_id,1])/Tr).astype(int)]
                    RH1_label = data_label[count, np.round(t_rh[rh_id,0]/Tr).astype(int):np.round((t_rh[rh_id,0]+t_rh[rh_id,1])/Tr).astype(int)]
                if rh_id == 1:
                    RH2 = a[np.round(t_rh[rh_id,0]/Tr).astype(int):np.round((t_rh[rh_id,0]+t_rh[rh_id,1])/Tr).astype(int)]
                    RH2_label = data_label[count, np.round(t_rh[rh_id,0]/Tr).astype(int):np.round((t_rh[rh_id,0]+t_rh[rh_id,1])/Tr).astype(int)]

            np.savez_compressed('/media/12T/practice/LR_DICT/ho_subject/MOTOR_RL/' + name + '/RH', RH1_label=RH1_label, RH1=RH1,RH2=RH2, RH2_label=RH2_label)


        if tl == 'RF':
            for rf_id in range(2):
                if rf_id == 0:
                    RF1 = a[np.round(t_rf[rf_id,0] / Tr).astype(int):np.round((t_rf[rf_id, 0] + t_rf[rf_id, 1]) / Tr).astype(int)]
                    RF1_label = data_label[count, np.round(t_rf[rf_id,0] / Tr).astype(int):np.round((t_rf[rf_id, 0] + t_rf[rf_id, 1]) / Tr).astype(int)]
                if rf_id == 1:
                    RF2 = a[np.round(t_rf[rf_id,0] / Tr).astype(int):np.round((t_rf[rf_id, 0] + t_rf[rf_id, 1]) / Tr).astype(int)]
                    RF2_label = data_label[count, np.round(t_rf[rf_id,0] / Tr).astype(int):np.round((t_rf[rf_id, 0] + t_rf[rf_id, 1]) / Tr).astype(int)]
            np.savez_compressed('/media/12T/practice/LR_DICT/ho_subject/MOTOR_RL/' + name + '/RF', RF1_label=RF1_label,RF1=RF1, RF2=RF2, RF2_label=RF2_label)

        if tl == 'T':
            for t_id in range(2):
                if t_id == 0:
                    T1 = a[np.round(t_t[t_id,0]/Tr).astype(int):np.round((t_t[t_id,0]+t_t[t_id,1])/Tr).astype(int)]
                    T1_label = data_label[count, np.round(t_t[t_id,0]/Tr).astype(int):np.round((t_t[t_id,0]+t_t[t_id,1])/Tr).astype(int)]
                if t_id == 1:
                    T2 = a[np.round(t_t[t_id,0]/Tr).astype(int):np.round((t_t[t_id,0]+t_t[t_id,1])/Tr).astype(int)]
                    T2_label = data_label[count, np.round(t_t[t_id,0]/Tr).astype(int):np.round((t_t[t_id,0]+t_t[t_id,1])/Tr).astype(int)]
            np.savez_compressed('/media/12T/practice/LR_DICT/ho_subject/MOTOR_RL/' + name + '/T', T1_label=T1_label, T1=T1, T2=T2,T2_label=T2_label)

        if tl == 'Cue':
            for cue_id in range(10):
                if cue_id == 0:
                    CUE1 = a[np.round(t_cue[cue_id, 0] / Tr).astype(int):np.round((t_cue[cue_id, 0] + t_cue[cue_id, 1]) / Tr).astype(int)]
                    CUE1_label = data_label[count, np.round(t_cue[cue_id, 0] / Tr).astype(int):np.round((t_cue[cue_id, 0] + t_cue[cue_id, 1]) / Tr).astype(int)]
                if cue_id == 1:
                    CUE2 = a[np.round(t_cue[cue_id, 0] / Tr).astype(int):np.round((t_cue[cue_id, 0] + t_cue[cue_id, 1]) / Tr).astype(int)]
                    CUE2_label = data_label[count, np.round(t_cue[cue_id, 0] / Tr).astype(int):np.round((t_cue[cue_id, 0] + t_cue[cue_id, 1]) / Tr).astype(int)]
                if cue_id == 2:
                    CUE3 = a[np.round(t_cue[cue_id, 0] / Tr).astype(int):np.round((t_cue[cue_id, 0] + t_cue[cue_id, 1]) / Tr).astype(int)]
                    CUE3_label = data_label[count, np.round(t_cue[cue_id, 0] / Tr).astype(int):np.round((t_cue[cue_id, 0] + t_cue[cue_id, 1]) / Tr).astype(int)]
                if cue_id == 3:
                    CUE4 = a[np.round(t_cue[cue_id, 0] / Tr).astype(int):np.round((t_cue[cue_id, 0] + t_cue[cue_id, 1]) / Tr).astype(int)]
                    CUE4_label = data_label[count, np.round(t_cue[cue_id, 0] / Tr).astype(int):np.round((t_cue[cue_id, 0] + t_cue[cue_id, 1]) / Tr).astype(int)]
                if cue_id == 4:
                    CUE5 = a[np.round(t_cue[cue_id, 0] / Tr).astype(int):np.round((t_cue[cue_id, 0] + t_cue[cue_id, 1]) / Tr).astype(int)]
                    CUE5_label = data_label[count, np.round(t_cue[cue_id, 0] / Tr).astype(int):np.round((t_cue[cue_id, 0] + t_cue[cue_id, 1]) / Tr).astype(int)]
                if cue_id == 5:
                    CUE6 = a[np.round(t_cue[cue_id, 0] / Tr).astype(int):np.round((t_cue[cue_id, 0] + t_cue[cue_id, 1]) / Tr).astype(int)]
                    CUE6_label = data_label[count, np.round(t_cue[cue_id, 0] / Tr).astype(int):np.round((t_cue[cue_id, 0] + t_cue[cue_id, 1]) / Tr).astype(int)]
                if cue_id == 6:
                    CUE7 = a[np.round(t_cue[cue_id, 0] / Tr).astype(int):np.round((t_cue[cue_id, 0] + t_cue[cue_id, 1]) / Tr).astype(int)]
                    CUE7_label = data_label[count, np.round(t_cue[cue_id, 0] / Tr).astype(int):np.round((t_cue[cue_id, 0] + t_cue[cue_id, 1]) / Tr).astype(int)]
                if cue_id == 7:
                    CUE8 = a[np.round(t_cue[cue_id, 0] / Tr).astype(int):np.round((t_cue[cue_id, 0] + t_cue[cue_id, 1]) / Tr).astype(int)]
                    CUE8_label = data_label[count, np.round(t_cue[cue_id, 0] / Tr).astype(int):np.round((t_cue[cue_id, 0] + t_cue[cue_id, 1]) / Tr).astype(int)]
                if cue_id == 8:
                    CUE9 = a[np.round(t_cue[cue_id, 0] / Tr).astype(int):np.round((t_cue[cue_id, 0] + t_cue[cue_id, 1]) / Tr).astype(int)]
                    CUE9_label = data_label[count, np.round(t_cue[cue_id, 0] / Tr).astype(int):np.round((t_cue[cue_id, 0] + t_cue[cue_id, 1]) / Tr).astype(int)]
                if cue_id == 9:
                    CUE10 = a[np.round(t_cue[cue_id, 0] / Tr).astype(int):np.round((t_cue[cue_id, 0] + t_cue[cue_id, 1]) / Tr).astype(int)]
                    CUE10_label = data_label[count, np.round(t_cue[cue_id, 0] / Tr).astype(int):np.round((t_cue[cue_id, 0] + t_cue[cue_id, 1]) / Tr).astype(int)]
            np.savez_compressed('/media/12T/practice/LR_DICT/ho_subject/MOTOR_RL/' + name + '/Cue', Cue1=CUE1, Cue2=CUE2, Cue3=CUE3,Cue4=CUE4, Cue5=CUE5, Cue6=CUE6, Cue7=CUE7, Cue8=CUE8, Cue9=CUE9, Cue10=CUE10,
                                Cue1_label=CUE1_label, Cue2_label=CUE2_label,Cue3_label=CUE3_label,Cue4_label=CUE4_label,Cue5_label=CUE5_label,Cue6_label=CUE6_label,Cue7_label=CUE7_label,Cue8_label=CUE8_label,Cue9_label=CUE9_label,Cue10_label=CUE10_label,)





    count = count + 1











# np.savez_compressed('/media/12T/practice/LR_DICT/ho_subject/MOTOR_LR/'+name+'/LH', LH1_label=, LH1=, LH2=, LH2_label= )
# np.savez_compressed('/media/12T/practice/LR_DICT/ho_subject/MOTOR_LR/'+name+'/LF', LF1_label=, LF1=, LF2=, LF2_label= )
# np.savez_compressed('/media/12T/practice/LR_DICT/ho_subject/MOTOR_LR/'+name+'/RH', RH1_label=, RH1=, RH2=, RH2_label= )
# np.savez_compressed('/media/12T/practice/LR_DICT/ho_subject/MOTOR_LR/'+name+'/RF', RF1_label=, RF1=, RF2=, RF2_label= )
# np.savez_compressed('/media/12T/practice/LR_DICT/ho_subject/MOTOR_LR/'+name+'/T', T1_label=, T1=, T2=, T2_label= )
# np.savez_compressed('/media/12T/practice/LR_DICT/ho_subject/MOTOR_LR/'+name+'/Cue', Cue1=,Cue2=,Cue3=,Cue4=,Cue5=,Cue6=,Cue7=,Cue8=,Cue9=,Cue10=, )
