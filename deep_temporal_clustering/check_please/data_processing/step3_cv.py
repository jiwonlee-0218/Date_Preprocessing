import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import os

data_path = '/home/jwlee/HMM/deep_temporal_clustering/check_please/Data/'
# Load social variable dataset
vari_path = data_path + 'var0525_0929.csv'
pd_socio = pd.read_csv(vari_path, usecols=['subjectkey', 'i_educ', 'i_income', 'n_pov', 'n_edu_h', 'race.ethnicity'])
scio_subj = pd_socio['subjectkey'].to_numpy()


# if not os.path.exists(rept_path2):
#     os.makedirs(rept_path2)
#     os.makedirs(rept_path3)

def mk_cv(num_race, case, seed, socio_var1=None, socio_var2=None):
    """
    num_race:  1 (white) 2 (black) 3 (hispanic) 4 (asian) 5 (etc)
    socio_var: 'i_educ', 'i_income', 'n_pov', 'n_edu_h'
    """
    print(num_race, case, seed, socio_var1, socio_var2)
    skf = KFold(n_splits=5, shuffle=True, random_state=seed)

    if case == 'single':
        # allidx = np.load(data_path + 'race_threshold25/race%d_%s_thr25.npz' % (num_race, socio_var1))
        # save_path_god = data_path + 'race_threshold25_cv/rp%d_r%d_god_%s.npz' % (seed, num_race, socio_var1)
        # save_path_bad = data_path + 'race_threshold25_cv/rp%d_r%d_bad_%s.npz' % (seed, num_race, socio_var1)
        allidx = np.load(data_path + 'race_threshold10/race%d_%s_thr10.npz' % (num_race, socio_var1))
        save_path_god = data_path + 'race_threshold10_cv/rp%d_r%d_god_%s.npz' % (seed, num_race, socio_var1)
        save_path_bad = data_path + 'race_threshold10_cv/rp%d_r%d_bad_%s.npz' % (seed, num_race, socio_var1)
        # allidx = np.load(data_path + 'race_threshold_option/race%d_%s_top21_bot12.npz' % (num_race, socio_var1))
        # save_path_god = data_path + 'race_threshold_option_cv/rp%d_r%d_god_%s_top21_bot12.npz' % (seed, num_race, socio_var1)
        # save_path_bad = data_path + 'race_threshold_option_cv/rp%d_r%d_bad_%s_top21_bot12.npz' % (seed, num_race, socio_var1)
    elif case == 'cross':
        # allidx = np.load(data_path + 'race_threshold25/race%d_%s_%s_thr25.npz' % (num_race, socio_var1, socio_var2))
        # save_path_god = data_path + 'race_threshold25_cv/rp%d_r%d_god_%s_%s.npz' % (seed, num_race, socio_var1, socio_var2)
        # save_path_bad = data_path + 'race_threshold25_cv/rp%d_r%d_bad_%s_%s.npz' % (seed, num_race, socio_var1, socio_var2)
        allidx = np.load(data_path + 'race_threshold10/race%d_%s_%s_thr10.npz' % (num_race, socio_var1, socio_var2))  # load from race_threshold10
        save_path_god = data_path + 'race_threshold10_cv/rp%d_r%d_god_%s_%s.npz' % (seed, num_race, socio_var1, socio_var2)  # save to cv root
        save_path_bad = data_path + 'race_threshold10_cv/rp%d_r%d_bad_%s_%s.npz' % (seed, num_race, socio_var1, socio_var2)  # save to cv root

    elif case == 'quad':
        allidx = np.load(data_path + 'race_threshold25/race%d_%s_%s_%s_%s_thr25.npz' % (num_race, 'i_educ', 'i_income', 'n_edu_h', 'n_pov'))
        save_path_god = data_path + 'race_threshold25_cv/rp%d_r%d_god_i_educ_i_income_n_edu_h_n_pov.npz' % (seed, num_race)
        save_path_bad = data_path + 'race_threshold25_cv/rp%d_r%d_bad_i_educ_i_income_n_edu_h_n_pov.npz' % (seed, num_race)

    god_idx = allidx['good_idx']
    bad_idx = allidx['bad_idx']

    trn, val, tst = [], [], []
    for train_index, test_index in skf.split(god_idx):
        X_trainval, X_test = god_idx[train_index], god_idx[test_index]
        X_train, X_valid = train_test_split(X_trainval, test_size=0.25, shuffle=True)
        trn.append(X_train), val.append(X_valid), tst.append(X_test)
        print('g', len(X_train), len(X_valid), len(X_test))
    np.savez(save_path_god, trn_idx=trn, val_idx=val, tst_idx=tst)

    trn, val, tst = [], [], []
    for train_index, test_index in skf.split(bad_idx):
        X_trainval, X_test = bad_idx[train_index], bad_idx[test_index]
        X_train, X_valid = train_test_split(X_trainval, test_size=0.25, shuffle=True)
        trn.append(X_train), val.append(X_valid), tst.append(X_test)
        print('b', len(X_train), len(X_valid), len(X_test))
    np.savez(save_path_bad, trn_idx=trn, val_idx=val, tst_idx=tst)
    return ("======================")


# for socio in ['i_educ', 'i_income', 'n_pov', 'n_edu_h']:
#     for r in [1, 2, 3]:
#         # 1210, 1218, 1224, 1225, 1231
#         for s in [3290, 3738, 5930, 1905, 2012, 2017, 3278, 1593]:
#             mk_cv(r, 'single', s, socio_var1=socio, socio_var2=None)

# for socio in ['i_educ']:
#     for r in [1]:
#         for s in [1210, 1218, 1224, 1225, 1231, 3290, 3738, 5930, 1905, 2012, 2017, 3278, 1593]:
#             mk_cv(r, 'single', s, socio_var1=socio, socio_var2=None)

for socio in [['i_educ', 'i_income'], ['i_educ', 'n_edu_h'], ['i_educ', 'n_pov'], ['i_income', 'n_edu_h'], ['i_income', 'n_pov'], ['n_edu_h', 'n_pov']]:
    for r in [1, 2, 3]:
        for s in [1210, 1218, 1224, 1225, 1231, 3290, 3738, 5930, 1905, 2012, 2017, 3278, 1593]:
            mk_cv(r, 'cross', s, socio_var1=socio[0], socio_var2=socio[1])

# for r in [1, 2, 3]:
#     # 1210, 1218, 1224, 1225, 1231
#     for s in [3290, 3738, 5930, 1905, 2012, 2017, 3278, 1593]:
#         mk_cv(r, 'quad', s)
