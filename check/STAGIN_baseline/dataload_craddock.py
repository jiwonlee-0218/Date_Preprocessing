import numpy as np
import torch
import os
import glob
import random
import pandas as pd
from random import shuffle, randrange
import torch
from sklearn.model_selection import train_test_split
# from torch.utils.data import DataLoader, TensorDataset
# from nilearn.connectome import ConnectivityMeasure
# from torch.utils.data import DataLoader, TensorDataset
# import scipy.io as sio
# from sklearn.model_selection import StratifiedKFold, train_test_split
# import csv
# import math
# from natsort import natsorted


def dataloader_hcp(args, fold, config=None, phase=None):

    signal, conn_l = torch.load('/DataCommon2/jwlee/2023_07_29_ST_Adaptive_Number_Of_Clusters/data/train_hcp-task_roi-aal.pth')

    signal_list = []
    for i in range(len(signal)):
        timeseries = signal[i]
        timeseries = (timeseries - np.mean(timeseries, axis=0, keepdims=True)) / (np.std(timeseries, axis=0, keepdims=True) + 1e-9)
        sampling_init = randrange((len(timeseries) - args.dynamic_length) + 1)
        timeseries = timeseries[sampling_init:sampling_init + args.dynamic_length]
        signal_list.append(timeseries)
    all_signal = np.stack(signal_list)


    task_timepoints = {'EMOTION': 176, 'GAMBLING': 253, 'LANGUAGE': 316, 'MOTOR': 284, 'RELATIONAL': 232, 'SOCIAL': 274, 'WM': 405}
    task_list = list(task_timepoints.keys())
    task_list.sort()

    label_list = []
    for i in range(len(conn_l)):
        for task_idx, _task in enumerate(task_list):
            if conn_l[i] == _task:
                label_list.append(task_idx)
    label_list = np.stack(label_list)

    all_signal = torch.tensor(all_signal, dtype=torch.float32)  ### add
    all_label = torch.from_numpy(label_list).long()

    x_train, x_valid, y_train, y_valid = train_test_split(all_signal, all_label, test_size=0.2, shuffle=True, stratify=all_label, random_state=34)
    train_loader = [x_train, y_train]
    val_loader = [x_valid, y_valid]

    return train_loader, val_loader



def dataloader(args, fold, train=True, config=None, phase=None):

    data_path = '/DataRead/REST-meta-MDD/REST-meta-MDD-Phase1-Sharing/'

    if train:

        # Directory paths
        if args.prec_glob:
            main_path = data_path + 'ROISignals_FunImgARglobalCWF/'
        else:
            main_path = data_path + 'ROISignals_FunImgARCWF/'

        if args.input_size == 116:
            cond_path = main_path + '830MDDvs771NC_aal.npz'
            idxd_path = data_path + 'CV5/830MDDvs771NC_2/rp%d_f%d.npz' % (args.rp, fold+1)
        elif args.input_size == 112:
            cond_path = main_path + '830MDDvs771NC_ho.npz'
            idxd_path = data_path + 'CV5/830MDDvs771NC_2/rp%d_f%d.npz' % (args.rp, fold+1)
        elif args.input_size == 160:
            cond_path = main_path + '830MDDvs771NC_Dosenbach_Combat.npz'
            idxd_path = data_path + 'CV5/830MDDvs771NC_2/rp%d_f%d.npz' % (args.rp, fold+1)
        elif args.input_size == 200:
            cond_path = main_path + '830MDDvs771NC_cc200_Combat.npz'
            idxd_path = data_path + 'CV5/830MDDvs771NC_2/rp%d_f%d.npz' % (args.rp, fold+1)

        signal_list = []


        conn_dict = np.load(cond_path, allow_pickle=True)
        indx_dict = np.load(idxd_path)

        signal = conn_dict['signal'] ### add
        conn_d = conn_dict['fc']
        conn_l = conn_dict['label']
        conn_l[conn_l == -1] = 0


        for i in range(len(signal)):
            timeseries = signal[i]
            sampling_init = randrange((len(timeseries) - args.dynamic_length) + 1)
            timeseries = timeseries[sampling_init:sampling_init + args.dynamic_length]
            signal_list.append(timeseries)
        all_signal = np.stack(signal_list)


        trnidx = indx_dict['trn_idx']
        validx = indx_dict['val_idx']
        tstidx = indx_dict['tst_idx']


        all_signal = torch.tensor(all_signal, dtype=torch.float32) ### add
        conn_d = torch.tensor(conn_d, dtype=torch.float32)
        conn_l = torch.from_numpy(conn_l).float()

        train_loader = [all_signal[trnidx], conn_d[trnidx], conn_l[trnidx]]
        val_loader = [all_signal[validx], conn_d[validx], conn_l[validx]]
        test_loader = [all_signal[tstidx], conn_d[tstidx], conn_l[tstidx]]

    else:
        # Directory paths
        if args.prec_glob:
            main_path = data_path + 'ROISignals_FunImgARglobalCWF/'
        else:
            main_path = data_path + 'ROISignals_FunImgARCWF/'

        if args.input_size == 116:
            cond_path = main_path + '830MDDvs771NC_aal.npz'
            idxd_path = data_path + 'CV5/830MDDvs771NC_2/rp%d_f%d.npz' % (args.rp, fold + 1)
        elif args.input_size == 112:
            cond_path = main_path + '830MDDvs771NC_ho.npz'
            idxd_path = data_path + 'CV5/830MDDvs771NC_2/rp%d_f%d.npz' % (args.rp, fold + 1)
        elif args.input_size == 160:
            cond_path = main_path + '830MDDvs771NC_Dosenbach_Combat.npz'
            idxd_path = data_path + 'CV5/830MDDvs771NC_2/rp%d_f%d.npz' % (args.rp, fold + 1)
        elif args.input_size == 200:
            cond_path = main_path + '830MDDvs771NC_cc200_Combat.npz'
            idxd_path = data_path + 'CV5/830MDDvs771NC_2/rp%d_f%d.npz' % (args.rp, fold + 1)

        signal_list = []

        conn_dict = np.load(cond_path, allow_pickle=True)
        indx_dict = np.load(idxd_path)

        signal = conn_dict['signal']  ### add
        conn_d = conn_dict['fc']
        conn_l = conn_dict['label']
        conn_l[conn_l == -1] = 0


        trnidx = indx_dict['trn_idx']
        validx = indx_dict['val_idx']
        tstidx = indx_dict['tst_idx']


        conn_d = torch.tensor(conn_d, dtype=torch.float32)
        conn_l = torch.from_numpy(conn_l).float()

        # signal.shape (1601,)    signal[0].shape -> (200, 200)    signal[200].shape -> (202, 200)


        train_loader = [signal[trnidx], conn_d[trnidx], conn_l[trnidx]] #tr_loader[0].shape (960,)    vl_loader[0].shape (320,)    ts_loader[0].shape (321,)
        val_loader = [signal[validx], conn_d[validx], conn_l[validx]]
        test_loader = [signal[tstidx], conn_d[tstidx], conn_l[tstidx]]



    return train_loader, val_loader, test_loader



def dataloader_indx(args, fold, config=None, phase=None):

    data_path = '/DataRead/REST-meta-MDD/REST-meta-MDD-Phase1-Sharing/'
    demo_path = '/DataRead/REST-meta-MDD/Stat_Sub_Info/Stat_Sub_Info_848MDDvs794NC_dropsite4_830MDDvs771NC.csv'
    data_id = pd.read_csv(demo_path)['SubID'].to_numpy()
    # demo_path = '/DataRead/REST-meta-MDD/REST-meta-MDD-PhenotypicData_WithHAMDSubItem_V4.xlsx'
    # HAMD_id = pd.read_excel(demo_path, sheet_name='MDD')['ID']
    # HAMD = pd.read_excel(demo_path, sheet_name='MDD')['HAMD']

    # Directory paths
    if args.prec_glob:
        main_path = data_path + 'ROISignals_FunImgARglobalCWF/'
    else:
        main_path = data_path + 'ROISignals_FunImgARCWF/'

    if args.input_size == 116:
        cond_path = main_path + '830MDDvs771NC_AAL_Combat.npz'
        idxd_path = data_path + 'CV5/830MDDvs771NC_2/rp%d_f%d.npz' % (args.rp, fold+1)
    elif args.input_size == 112:
        cond_path = main_path + '830MDDvs771NC_HO_Combat.npz'
        idxd_path = data_path + 'CV5/830MDDvs771NC_2/rp%d_f%d.npz' % (args.rp, fold+1)
    elif args.input_size == 160:
        cond_path = main_path + '830MDDvs771NC_Dosenbach_Combat.npz'
        idxd_path = data_path + 'CV5/830MDDvs771NC_2/rp%d_f%d.npz' % (args.rp, fold+1)
    elif args.input_size == 200:
        cond_path = main_path + '830MDDvs771NC_Craddock_Combat.npz'
        idxd_path = data_path + 'CV5/830MDDvs771NC_2/rp%d_f%d.npz' % (args.rp, fold+1)

    conn_dict = np.load(cond_path)
    indx_dict = np.load(idxd_path)

    conn_d = conn_dict['fc']
    conn_l = conn_dict['label']
    conn_l[conn_l == -1] = 0
    trnidx = indx_dict['trn_idx']
    validx = indx_dict['val_idx']
    tstidx = indx_dict['tst_idx']

    conn_d = torch.tensor(conn_d, dtype=torch.float32)
    conn_l = torch.from_numpy(conn_l).long()

    train_loader = [conn_d[trnidx], conn_l[trnidx], data_id[trnidx]]
    val_loader = [conn_d[validx], conn_l[validx], data_id[validx]]
    test_loader = [conn_d[tstidx], conn_l[tstidx], data_id[tstidx]]

    return train_loader, val_loader, test_loader



# def dataloader_mlr(args, fold, config=None, phase=None):
#
#     data_path = '/DataRead/REST-meta-MDD/REST-meta-MDD-Phase1-Sharing/'
#     # Directory paths
#     if args.prec_glob:
#         main_path = data_path + 'ROISignals_FunImgARglobalCWF/'
#     else:
#         main_path = data_path + 'ROISignals_FunImgARCWF/'
#
#     if args.input_size == 116:
#         cond_path = main_path + '830MDDvs771NC_AAL_Combat_MLR.npz'
#         idxd_path = data_path + 'CV5/830MDDvs771NC_2/rp%d_f%d.npz' % (args.rp, fold+1)
#     elif args.input_size == 112:
#         cond_path = main_path + '830MDDvs771NC_HO_Combat_MLR.npz'
#         idxd_path = data_path + 'CV5/830MDDvs771NC_2/rp%d_f%d.npz' % (args.rp, fold+1)
#     elif args.input_size == 160:
#         cond_path = main_path + '830MDDvs771NC_Dosenbach_Combat_MLR.npz'
#         idxd_path = data_path + 'CV5/830MDDvs771NC_2/rp%d_f%d.npz' % (args.rp, fold+1)
#     elif args.input_size == 200:
#         cond_path = main_path + '830MDDvs771NC_??_Combat_MLR.npz'
#         idxd_path = data_path + 'CV5/830MDDvs771NC_2/rp%d_f%d.npz' % (args.rp, fold+1)
#
#     conn_dict = np.load(cond_path)
#     indx_dict = np.load(idxd_path)
#
#     conn_d = conn_dict['fc']
#     conn_l = conn_dict['label']
#     conn_l[conn_l == -1] = 0
#     trnidx = indx_dict['trn_idx']
#     validx = indx_dict['val_idx']
#     tstidx = indx_dict['tst_idx']
#
#     train_loader = convert_Dloader(args.bs, conn_d[trnidx], conn_l[trnidx], num_workers=0, shuffle=True)
#     val_loader = convert_Dloader(validx.shape[0], conn_d[validx], conn_l[validx], num_workers=0, shuffle=False)
#     test_loader = convert_Dloader(tstidx.shape[0], conn_d[tstidx], conn_l[tstidx], num_workers=0, shuffle=False)
#
#     return train_loader, val_loader, test_loader
#
#
# def dataloader_ttest(args, fold, config=None, phase=None):
#     from scipy.stats import ttest_ind
#     data_path = '/DataRead/REST-meta-MDD/REST-meta-MDD-Phase1-Sharing/'
#     # Directory paths
#     if args.prec_glob:
#         main_path = data_path + 'ROISignals_FunImgARglobalCWF/'
#     else:
#         main_path = data_path + 'ROISignals_FunImgARCWF/'
#
#     if args.input_size == 116:
#         cond_path = main_path + '830MDDvs771NC_AAL_Combat.npz'
#         idxd_path = data_path + 'CV5/830MDDvs771NC_2/rp%d_f%d.npz' % (args.rp, fold+1)
#     elif args.input_size == 112:
#         cond_path = main_path + '830MDDvs771NC_HO_Combat.npz'
#         idxd_path = data_path + 'CV5/830MDDvs771NC_2/rp%d_f%d.npz' % (args.rp, fold+1)
#     elif args.input_size == 160:
#         cond_path = main_path + '830MDDvs771NC_Dosenbach_Combat.npz'
#         idxd_path = data_path + 'CV5/830MDDvs771NC_2/rp%d_f%d.npz' % (args.rp, fold+1)
#     elif args.input_size == 200:
#         cond_path = main_path + '830MDDvs771NC_Craddock_Combat.npz'
#         idxd_path = data_path + 'CV5/830MDDvs771NC_2/rp%d_f%d.npz' % (args.rp, fold+1)
#
#     conn_dict = np.load(cond_path)
#     indx_dict = np.load(idxd_path)
#
#     conn_d = conn_dict['fc']
#     conn_l = conn_dict['label']
#     conn_l[conn_l == -1] = 0
#     trnidx = indx_dict['trn_idx']
#     validx = indx_dict['val_idx']
#     tstidx = indx_dict['tst_idx']
#
#     trn_asd_idx = np.argwhere(conn_l[trnidx] == 1).squeeze()
#     trn_td_idx = np.argwhere(conn_l[trnidx] == 0).squeeze()
#     _, p = ttest_ind(conn_d[trnidx][trn_asd_idx], conn_d[trnidx][trn_td_idx])
#     select = np.argwhere(p > 0.05)
#     conn_d[:, select[0], select[1]] = 0
#
#     train_loader = convert_Dloader(args.bs, conn_d[trnidx], conn_l[trnidx], num_workers=0, shuffle=True)
#     val_loader = convert_Dloader(validx.shape[0], conn_d[validx], conn_l[validx], num_workers=0, shuffle=False)
#     test_loader = convert_Dloader(tstidx.shape[0], conn_d[tstidx], conn_l[tstidx], num_workers=0, shuffle=False)
#
#     return train_loader, val_loader, test_loader
#
# def dataloader_lasso(args, fold, config=None, phase=None):
#     from sklearn.linear_model import Lasso
#
#     data_path = '/DataRead/REST-meta-MDD/REST-meta-MDD-Phase1-Sharing/'
#     # Directory paths
#     if args.prec_glob:
#         main_path = data_path + 'ROISignals_FunImgARglobalCWF/'
#     else:
#         main_path = data_path + 'ROISignals_FunImgARCWF/'
#
#     if args.input_size == 116:
#         cond_path = main_path + '830MDDvs771NC_AAL_Combat.npz'
#         idxd_path = data_path + 'CV5/830MDDvs771NC_2/rp%d_f%d.npz' % (args.rp, fold+1)
#     elif args.input_size == 112:
#         cond_path = main_path + '830MDDvs771NC_HO_Combat.npz'
#         idxd_path = data_path + 'CV5/830MDDvs771NC_2/rp%d_f%d.npz' % (args.rp, fold+1)
#     elif args.input_size == 160:
#         cond_path = main_path + '830MDDvs771NC_Dosenbach_Combat.npz'
#         idxd_path = data_path + 'CV5/830MDDvs771NC_2/rp%d_f%d.npz' % (args.rp, fold+1)
#     elif args.input_size == 200:
#         cond_path = main_path + '830MDDvs771NC_Craddock_Combat.npz'
#         idxd_path = data_path + 'CV5/830MDDvs771NC_2/rp%d_f%d.npz' % (args.rp, fold+1)
#
#     conn_dict = np.load(cond_path)
#     indx_dict = np.load(idxd_path)
#
#     conn_d = conn_dict['fc']
#     conn_l = conn_dict['label']
#     conn_l[conn_l == -1] = 0
#     trnidx = indx_dict['trn_idx']
#     validx = indx_dict['val_idx']
#     tstidx = indx_dict['tst_idx']
#
#     idx = np.triu_indices(args.input_size, 1)
#
#     grid_score = []
#     grid_model = []
#     # for alpha in np.arange(0.002, 0.0025, 0.0001):
#     for alpha in [0.001, 0.005]:
#         lasso = Lasso(alpha=alpha)
#         lasso.fit(conn_d[trnidx][:, idx[0], idx[1]], conn_l[trnidx])
#         score = lasso.score(conn_d[validx][:, idx[0], idx[1]], conn_l[validx])
#         grid_score.append(score)
#         grid_model.append(lasso)
#
#     best_loc = np.argmax(np.array(grid_score))
#     best_lasso = grid_model[best_loc]
#     best_feat = np.argwhere(np.abs(best_lasso.coef_) != 0).squeeze(-1)
#
#     mask = np.zeros((args.input_size, args.input_size))
#     for f in best_feat.tolist():
#         mask[idx[0][f], idx[1][f]] = 1
#     mask = mask + mask.transpose()
#
#     train_loader = convert_Dloader(args.bs, conn_d[trnidx]*mask, conn_l[trnidx], num_workers=0, shuffle=True)
#     val_loader = convert_Dloader(validx.shape[0], conn_d[validx]*mask, conn_l[validx], num_workers=0, shuffle=False)
#     test_loader = convert_Dloader(tstidx.shape[0], conn_d[tstidx]*mask, conn_l[tstidx], num_workers=0, shuffle=False)
#
#     return train_loader, val_loader, test_loader

