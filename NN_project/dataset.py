import os
import torch
import numpy as np


def dataloader(args):

    # you need to access google drive and download dataset
    train_timeseries, train_label = torch.load('/home/jwlee/HMM/NN_project/data/train_hcp-task_roi-aal.pth')
    subtask_label = np.load('/home/jwlee/HMM/NN_project/data/subtask_labels_detail.npy', allow_pickle=True).item()
    test_timeseries, test_label = torch.load('/home/jwlee/HMM/NN_project/data/test_hcp-task_roi-aal.pth')



    task_timepoints = {'EMOTION': 176, 'GAMBLING': 253, 'LANGUAGE': 316, 'MOTOR': 284, 'RELATIONAL': 232, 'SOCIAL': 274, 'WM': 405}
    task_list = list(task_timepoints.keys())


    train_list = []
    for i in range(len(train_timeseries)):
        timeseries = train_timeseries[i]
        timeseries = (timeseries - np.mean(timeseries, axis=0, keepdims=True)) / (np.std(timeseries, axis=0, keepdims=True) + 1e-9)
        timeseries = timeseries[:176]
        train_list.append(timeseries)

    train_label_list = []
    for k in range(len(train_label)):
        for task_idx, _task in enumerate(task_list):
            if train_label[k] == _task:
                train_label_list.append(task_idx)





    test_list = []
    for j in range(len(test_timeseries)):
        tst_timeseries = test_timeseries[j]
        tst_timeseries = (tst_timeseries - np.mean(tst_timeseries, axis=0, keepdims=True)) / (np.std(tst_timeseries, axis=0, keepdims=True) + 1e-9)
        tst_timeseries = tst_timeseries[:176]
        test_list.append(tst_timeseries)


    test_label_list = []
    for p in range(len(test_label)):
        for task_idx, _task in enumerate(task_list):
            if test_label[p] == _task:
                test_label_list.append(task_idx)


    subtask_dict = {}
    for task_idx, _task in enumerate(task_list):
        subtask_dict[_task] = subtask_label[_task][:176]



    return np.array(train_list), np.array(train_label_list), np.array(test_list), np.array(test_label_list), subtask_dict, task_list



