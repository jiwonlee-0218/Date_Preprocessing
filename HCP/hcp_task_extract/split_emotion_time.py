# HCP task fMRI time labeling

import os
import numpy as np
from glob import glob
import nibabel as nib
import scipy.io as sio



frames = 176
Tr = 0.72
whole_sc = frames * Tr


data_path = '/mnt/hcp/homes/jwlee/HCP_train_data/tfMRI_EMOTION_LR/AAL/'  ### mat file  1042 subject
dir_path = '/mnt/hcp/HCP/'
list = sorted(os.listdir(data_path))
print(list) # ['AAL_116_100206.mat', 'AAL_116_100307.mat', ... , ] list: 1042



def split_task_only_LR():
    for i in range(len(list)):
        sub = list[i].split('.')[0]  # 100206
        if sub == '136126':
            continue


        n1 = sio.loadmat(data_path + list[i])
        n1 = n1['ROI']
        print(n1.shape)  # (176, 116)

        sub_split_data = dict(fear=[], neut=[])
        for fn in glob(dir_path + sub +'/MNINonLinear/Results/tfMRI_EMOTION_LR/EVs/*.txt'):


            file = fn.split('/')
            if file[-1] == 'cue.txt':
                t_cue = np.loadtxt(dir_path + sub +'/MNINonLinear/Results/tfMRI_EMOTION_LR/EVs/'+file[-1], delimiter='\t')
            if file[-1] == 'fear.txt':
                t_fear = np.loadtxt(dir_path + sub +'/MNINonLinear/Results/tfMRI_EMOTION_LR/EVs/'+file[-1], delimiter='\t')  # numpy.ndarray(3,3)

            if file[-1] == 'neut.txt':
                t_neut = np.loadtxt(dir_path + sub +'/MNINonLinear/Results/tfMRI_EMOTION_LR/EVs/'+file[-1], delimiter='\t')  # numpy.ndarray(3,3)




        label_list = [0 for i in range(176)]





        for cue_id in range(6):
            label_list[ np.round(t_cue[cue_id , 0] / Tr).astype(int) - 11 : np.round((t_cue[cue_id, 0] + t_cue[cue_id, 1]) / Tr).astype(int) - 11 ] = [0 for i in range(np.round((t_cue[cue_id, 0] + t_cue[cue_id, 1]) / Tr).astype(int) - np.round(t_cue[cue_id, 0] / Tr).astype(int))]

        for fear_id in range(3):
            label_list[ np.round(t_fear[fear_id , 0] / Tr).astype(int) - 11 : np.round((t_fear[fear_id, 0] + t_fear[fear_id, 1]) / Tr).astype(int) - 11 ] = [2 for i in range(np.round((t_fear[fear_id , 0] + t_fear[fear_id , 1]) / Tr).astype(int) - np.round(t_fear[fear_id , 0] / Tr).astype(int))]

        for neut_id in range(3):
            label_list[ np.round(t_neut[neut_id , 0] / Tr).astype(int) - 11 : np.round((t_neut[neut_id, 0] + t_neut[neut_id, 1]) / Tr).astype(int) - 11  ] = [1 for i in range(np.round((t_neut[neut_id , 0] + t_neut[neut_id , 1]) / Tr).astype(int) - np.round(t_neut[neut_id , 0] / Tr).astype(int) )]




#        print(label_list)

        label_list = np.array(label_list)
        label_list = np.expand_dims(label_list, axis=0)  # (1, 176)

        if i == 0:
            label_list_LR = label_list

        else:
            label_list_LR = np.concatenate((label_list_LR, label_list))


    print(label_list_LR.shape)

    return label_list_LR

















data_path2 = '/mnt/hcp/homes/jwlee/HCP_train_data/tfMRI_EMOTION_RL/AAL/'
dir_path2 = '/mnt/hcp/HCP/'
list2 = sorted(os.listdir(data_path2))
print(list2)  # 1049




def split_task_only_RL():

    for i in range(len(list2)):
        sub = list2[i].split('.')[0]

        n1 = sio.loadmat(data_path2 + list2[i])
        n1 = n1['ROI']

        for fn in glob(dir_path2 + sub +'/MNINonLinear/Results/tfMRI_EMOTION_RL/EVs/*.txt'):


            file = fn.split('/')
            if file[-1] == 'cue.txt':
                t_cue = np.loadtxt(dir_path2 + sub + '/MNINonLinear/Results/tfMRI_EMOTION_RL/EVs/' + file[-1], delimiter='\t')

            if file[-1] == 'fear.txt':
                t_fear = np.loadtxt(dir_path + sub + '/MNINonLinear/Results/tfMRI_EMOTION_RL/EVs/' + file[-1], delimiter='\t')  # numpy.ndarray(3,3)

            if file[-1] == 'neut.txt':
                t_neut = np.loadtxt(dir_path + sub + '/MNINonLinear/Results/tfMRI_EMOTION_RL/EVs/' + file[-1], delimiter='\t')  # numpy.ndarray(3,3)





        label_list_RL = [0 for i in range(176)]

#

        for cue_id in range(6):
            label_list_RL[   np.round(t_cue[cue_id, 0] / Tr).astype(int) - 11  :  np.round((t_cue[cue_id, 0] + t_cue[cue_id, 1]) / Tr).astype(int) - 11  ] = [    0 for i in range(np.round((t_cue[cue_id, 0] + t_cue[cue_id, 1]) / Tr).astype(int) - np.round(t_cue[cue_id, 0] / Tr).astype(int))]

        for fear_id in range(3):
            label_list_RL[   np.round(t_fear[fear_id, 0] / Tr).astype(int) - 11   :  np.round((t_fear[fear_id, 0] + t_fear[fear_id, 1]) / Tr).astype(int) - 11   ] = [   2 for i in range(np.round((t_fear[fear_id, 0] + t_fear[fear_id, 1]) / Tr).astype(int) - np.round(t_fear[fear_id, 0] / Tr).astype(int))]

        for neut_id in range(3):
            label_list_RL[   np.round(t_neut[neut_id, 0] / Tr).astype(int) - 11   :  np.round((t_neut[neut_id, 0] + t_neut[neut_id, 1]) / Tr).astype(int) - 11   ] = [   1 for i in range(np.round((t_neut[neut_id, 0] + t_neut[neut_id, 1]) / Tr).astype(int) - np.round(t_neut[neut_id, 0] / Tr).astype(int))]


#        print(label_list)

        label_list_RL = np.array(label_list_RL)
        label_list_RL = np.expand_dims(label_list_RL, axis=0)

        if i == 0:
            total_label_list_RL = label_list_RL


        else:
            total_label_list_RL = np.concatenate((total_label_list_RL, label_list_RL))
            print(i,sub, total_label_list_RL.shape)

    print(total_label_list_RL.shape)
    return total_label_list_RL



Label_list_LR = split_task_only_LR()
Label_list_RL = split_task_only_RL()




np.savez_compressed('/media/12T/practice/LR_DICT/emotion/hcp_emotion_label', label_list_LR=Label_list_LR, label_list_RL=Label_list_RL)






# import shutil
#
# data_path = '/mnt/hcp/homes/jwlee/HCP_train_data/tfMRI_EMOTION_RL/AAL/*'
# save_path = '/mnt/hcp/homes/jwlee/HCP_train_data/tfMRI_EMOTION_RL/AAL/'
#
# for i in sorted(glob(data_path)):
#     name = i.split('/')[-1] # AAL_116_724446.mat
#     old_name = os.path.join(save_path, name)
#     print(old_name)
#
#     new_name = name[8:14] # 724446
#     new_name = new_name + '.mat'
#     new_name = os.path.join(save_path, new_name)
#     print(new_name)
#
#     shutil.move(old_name, new_name)


# import shutil
#
# file_path = '/mnt/hcp/HCP/100206/MNINonLinear/Results/tfMRI_EMOTION_LR/EVs/cue.txt'
#
#
#
# data_path = '/mnt/hcp/homes/jwlee/HCP_train_data/tfMRI_EMOTION_RL/AAL/'
# dir_path = '/mnt/hcp/HCP/'
# list = sorted(os.listdir(data_path))
#
# for i in range(len(list)):
#
#     sub = list[i].split('.')[0]
#
#
#     destination_path = dir_path + sub +'/MNINonLinear/Results/tfMRI_EMOTION_RL/EVs/'
#     shutil.copyfile(file_path, os.path.join(destination_path, 'cue.txt'))
#     print(sub, ' finish!!')
