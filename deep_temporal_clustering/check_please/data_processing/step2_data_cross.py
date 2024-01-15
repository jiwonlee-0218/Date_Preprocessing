import numpy as np
import pandas as pd

data_path = '/home/jwlee/HMM/deep_temporal_clustering/check_please/Data/'
# Load social variable dataset
vari_path = data_path + 'var0525_0929.csv'
pd_socio = pd.read_csv(vari_path, usecols=['subjectkey', 'i_educ', 'i_income', 'n_pov', 'n_edu_h', 'race.ethnicity'])
scio_subj = pd_socio['subjectkey'].to_numpy()

def mk_cross_data(num_race, socio_var1, socio_var2):

    # socio1 = np.load(data_path + 'race_threshold25/race%d_%s_thr25.npz' % (num_race, socio_var1), allow_pickle=True)
    socio1 = np.load(data_path + 'race_threshold10/race%d_%s_thr10.npz' % (num_race, socio_var1), allow_pickle=True)
    socio1_god_idx, socio1_bad_idx = socio1['good_idx'], socio1['bad_idx']
    # socio2 = np.load(data_path + 'race_threshold25/race%d_%s_thr25.npz' % (num_race, socio_var2), allow_pickle=True)
    socio2 = np.load(data_path + 'race_threshold10/race%d_%s_thr10.npz' % (num_race, socio_var2), allow_pickle=True)
    socio2_god_idx, socio2_bad_idx = socio2['good_idx'], socio2['bad_idx']
    socio12_god_idx = np.intersect1d(socio1_god_idx, socio2_god_idx)
    socio12_bad_idx = np.intersect1d(socio1_bad_idx, socio2_bad_idx)

    # np.savez('race%d_%s_%s_thr25.npz' % (num_race, socio_var1, socio_var2), good_idx=socio12_god_idx, bad_idx=socio12_bad_idx,
    #          good_subkey=scio_subj[socio12_god_idx], bad_subkey=scio_subj[socio12_bad_idx],
    #          good_value_socio1=pd_socio[socio_var1][socio12_god_idx].to_numpy(), bad_value_socio1=pd_socio[socio_var1][socio12_bad_idx].to_numpy(),
    #          good_value_socio2=pd_socio[socio_var2][socio12_god_idx].to_numpy(), bad_value_socio2=pd_socio[socio_var2][socio12_bad_idx].to_numpy())
    np.savez('race%d_%s_%s_thr10.npz' % (num_race, socio_var1, socio_var2), good_idx=socio12_god_idx, bad_idx=socio12_bad_idx,
             good_subkey=scio_subj[socio12_god_idx], bad_subkey=scio_subj[socio12_bad_idx],
             good_value_socio1=pd_socio[socio_var1][socio12_god_idx].to_numpy(), bad_value_socio1=pd_socio[socio_var1][socio12_bad_idx].to_numpy(),
             good_value_socio2=pd_socio[socio_var2][socio12_god_idx].to_numpy(), bad_value_socio2=pd_socio[socio_var2][socio12_bad_idx].to_numpy())
    return print(socio12_god_idx.shape[0], socio12_bad_idx.shape[0])



print("====== i educ & i_income ======")
mk_cross_data(1, 'i_educ', 'i_income')
mk_cross_data(2, 'i_educ', 'i_income')
mk_cross_data(3, 'i_educ', 'i_income')

print("====== i educ & n_pov ======")
mk_cross_data(1, 'i_educ', 'n_pov')
mk_cross_data(2, 'i_educ', 'n_pov')
mk_cross_data(3, 'i_educ', 'n_pov')

print("====== i educ & n edu h  ======")
mk_cross_data(1, 'i_educ', 'n_edu_h')
mk_cross_data(2, 'i_educ', 'n_edu_h')
mk_cross_data(3, 'i_educ', 'n_edu_h')

print("====== i_income & n_pov ======")
mk_cross_data(1, 'i_income', 'n_pov')
mk_cross_data(2, 'i_income', 'n_pov')
mk_cross_data(3, 'i_income', 'n_pov')

print("====== i_income & n edu h  ======")
mk_cross_data(1, 'i_income', 'n_edu_h')
mk_cross_data(2, 'i_income', 'n_edu_h')
mk_cross_data(3, 'i_income', 'n_edu_h')

print("====== n_edu_h & n pov ======")
mk_cross_data(1, 'n_edu_h', 'n_pov')
mk_cross_data(2, 'n_edu_h', 'n_pov')
mk_cross_data(3, 'n_edu_h', 'n_pov')
