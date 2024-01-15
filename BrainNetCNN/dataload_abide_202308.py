import numpy as np
import torch
import os
import glob
import random
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from nilearn.connectome import ConnectivityMeasure
import scipy.io as sio
from sklearn.model_selection import StratifiedKFold, train_test_split


def dataloader(args, fold, config=None, phase=None):

    main_path = '/DataRead/ABIDE/ABIDE_I_Raw_Download/ABIDE_corr/{}_{}_{}.npz'.format(args.prec_type, args.atlas, args.prec_glob)
    indx_path = '/DataRead/ABIDE/ABIDE_I_Raw_Download/ABIDE_cv/cv5/rp%d_f%d.npz' % (args.rp, fold)

    data_dict = np.load(main_path)
    conn_d = data_dict['conn']
    conn_l = data_dict['labels']
    subj = data_dict['subID']

    if os.path.exists(indx_path):
        pass
    else:
        alli = np.arange(conn_d.shape[0])
        for r in range(10):
            skf = StratifiedKFold(n_splits=5, shuffle=True)
            f = 1
            for train_index, test_index in skf.split(alli, conn_l):
                X_trainval, X_test = alli[train_index], alli[test_index]
                y_trainval, y_test = conn_l[train_index], conn_l[test_index]
                X_train, X_valid = train_test_split(X_trainval, test_size=(1/4), shuffle=True, stratify=y_trainval)
                assert np.unique(np.concatenate([X_train, X_valid, X_test])).shape[0] == conn_d.shape[0]
                np.savez('/DataRead/ABIDE/ABIDE_I_Raw_Download/ABIDE_cv/cv5/rp%d_f%d.npz' % (r + 1, f), trn_idx=X_train, val_idx=X_valid, tst_idx=X_test,
                         trn_sub=subj[X_train], val_sub=subj[X_valid], tst_sub=subj[X_test])
                f += 1

    indx_dict = np.load(indx_path)
    trnidx = indx_dict['trn_idx']
    validx = indx_dict['val_idx']
    tstidx = indx_dict['tst_idx']

    # autism 1 normal 2 -> autism 1 normal 0
    conn_l = abs(conn_l - 2)

    conn_d = torch.tensor(conn_d, dtype=torch.float32)
    conn_l = torch.from_numpy(conn_l).long()

    train_loader = [conn_d[trnidx], conn_l[trnidx]]
    val_loader = [conn_d[validx], conn_l[validx]]
    test_loader = [conn_d[tstidx], conn_l[tstidx]]

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

