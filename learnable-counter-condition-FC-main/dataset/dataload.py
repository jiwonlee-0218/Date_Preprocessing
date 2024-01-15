import numpy as np
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from random import shuffle


def dataloader(args, fold, config=None, phase=None):

    cond_path = '/DataPath/'
    idxd_path = '/IndexPath/rp%d_f%d.npz' % (args.rp, args.fold)

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

    train_loader = [conn_d[trnidx], conn_l[trnidx]]
    val_loader = [conn_d[validx], conn_l[validx]]
    test_loader = [conn_d[tstidx], conn_l[tstidx]]

    return train_loader, val_loader, test_loader





# class MDD_Dataset_train(torch.utils.data.Dataset):
#     def __init__(self, args):
#         super().__init__()
#
#
#         a = np.load('/DataRead/KUMC_MH/2304_HAN_suk/FC_data/dict/kumc_mh_mdd_cc200.npz')
#         self.fc = a['mdd_fc_dataset']
#         self.label = a['mdd_fc_dataset_label']
#
#         k_fold = args.fold
#
#
#
#         if k_fold > 1:
#             self.k_fold = StratifiedKFold(k_fold, shuffle=True, random_state=0)
#             self.k = None
#         else:
#             self.k_fold = None
#
#         self.num_nodes = args.input_size
#         self.num_classes = args.num_label
#         self.train = None
#
#
#
#
#     def __len__(self):
#         return len(self.fold_idx) if self.k is not None else self.fc.shape[0]
#
#
#
#
#     def set_fold(self, fold, train=True):
#         if not self.k_fold:
#             return
#         self.k = fold
#         train_idx, test_idx = list( self.k_fold.split(self.timeseries_list, self.label_list) )[fold]
#
#
#         if train:
#             shuffle(train_idx)
#             self.fold_idx = train_idx
#             self.train = True
#         else:
#             self.fold_idx = test_idx
#             self.train = False
#
#
#
#
#     def __getitem__(self, idx):
#         timeseries = self.timeseries_list[self.fold_idx[idx]]
#         timeseries = (timeseries - np.mean(timeseries, axis=0, keepdims=True)) / (np.std(timeseries, axis=0, keepdims=True) + 1e-9)
#         if not self.standardized_length is None:
#             timeseries = timeseries[:self.standardized_length]
#         task = self.label_list[self.fold_idx[idx]]
#
#         for task_idx, _task in enumerate(self.task_list):
#             if task == _task:
#                 label = task_idx
#
#         return {'timeseries': torch.tensor(timeseries, dtype=torch.float32), 'label': torch.tensor(label)}