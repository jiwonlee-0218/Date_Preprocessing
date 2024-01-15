import numpy as np
import pandas as pd

data_path = '/home/jwlee/HMM/deep_temporal_clustering/check_please/Data/'
# Load social variable dataset
vari_path = data_path + 'var0525_0929.csv'
pd_socio = pd.read_csv(vari_path, usecols=['subjectkey', 'i_educ', 'i_income', 'n_pov', 'n_edu_h', 'race.ethnicity'])
scio_subj = pd_socio['subjectkey'].to_numpy()

num_race = 1

# socio1 = np.load(data_path + 'race_threshold25/race%d_%s_thr25.npz' % (num_race, 'i_educ'), allow_pickle=True)
socio1 = np.load(data_path + 'race_threshold10/race%d_%s_thr10.npz' % (num_race, 'i_educ'), allow_pickle=True)
socio1_god_idx, socio1_bad_idx = socio1['good_idx'], socio1['bad_idx']
# socio2 = np.load(data_path + 'race_threshold25/race%d_%s_thr25.npz' % (num_race, 'i_income'), allow_pickle=True)
socio2 = np.load(data_path + 'race_threshold10/race%d_%s_thr10.npz' % (num_race, 'i_income'), allow_pickle=True)
socio2_god_idx, socio2_bad_idx = socio2['good_idx'], socio2['bad_idx']
# socio3 = np.load(data_path + 'race_threshold25/race%d_%s_thr25.npz' % (num_race, 'n_edu_h'), allow_pickle=True)
socio3 = np.load(data_path + 'race_threshold10/race%d_%s_thr10.npz' % (num_race, 'n_edu_h'), allow_pickle=True)
socio3_god_idx, socio3_bad_idx = socio3['good_idx'], socio3['bad_idx']
# socio4 = np.load(data_path + 'race_threshold25/race%d_%s_thr25.npz' % (num_race, 'n_pov'), allow_pickle=True)
socio4 = np.load(data_path + 'race_threshold10/race%d_%s_thr10.npz' % (num_race, 'n_pov'), allow_pickle=True)
socio4_god_idx, socio4_bad_idx = socio4['good_idx'], socio4['bad_idx']

socio1234_god_idx = list(set.intersection(set(socio1_god_idx), set(socio2_god_idx), set(socio3_god_idx), set(socio4_god_idx)))
socio1234_bad_idx = list(set.intersection(set(socio1_bad_idx), set(socio2_bad_idx), set(socio3_bad_idx), set(socio4_bad_idx)))

# np.savez('race%d_%s_%s_%s_%s_thr25.npz' % (num_race, 'i_educ', 'i_income', 'n_edu_h', 'n_pov'), good_idx=socio1234_god_idx, bad_idx=socio1234_bad_idx,
#          good_subkey=scio_subj[socio1234_god_idx], bad_subkey=scio_subj[socio1234_bad_idx],
#          good_value_socio1=pd_socio['i_educ'][socio1234_god_idx].to_numpy(), bad_value_socio1=pd_socio['i_educ'][socio1234_bad_idx].to_numpy(),
#          good_value_socio2=pd_socio['i_income'][socio1234_god_idx].to_numpy(), bad_value_socio2=pd_socio['i_income'][socio1234_bad_idx].to_numpy(),
#          good_value_socio3=pd_socio['n_edu_h'][socio1234_god_idx].to_numpy(), bad_value_socio3=pd_socio['n_edu_h'][socio1234_bad_idx].to_numpy(),
#          good_value_socio4=pd_socio['n_pov'][socio1234_god_idx].to_numpy(), bad_value_socio4=pd_socio['n_pov'][socio1234_bad_idx].to_numpy())
# np.savez('race%d_%s_%s_%s_%s_thr10.npz' % (num_race, 'i_educ', 'i_income', 'n_edu_h', 'n_pov'), good_idx=socio1234_god_idx, bad_idx=socio1234_bad_idx,
#          good_subkey=scio_subj[socio1234_god_idx], bad_subkey=scio_subj[socio1234_bad_idx],
#          good_value_socio1=pd_socio['i_educ'][socio1234_god_idx].to_numpy(), bad_value_socio1=pd_socio['i_educ'][socio1234_bad_idx].to_numpy(),
#          good_value_socio2=pd_socio['i_income'][socio1234_god_idx].to_numpy(), bad_value_socio2=pd_socio['i_income'][socio1234_bad_idx].to_numpy(),
#          good_value_socio3=pd_socio['n_edu_h'][socio1234_god_idx].to_numpy(), bad_value_socio3=pd_socio['n_edu_h'][socio1234_bad_idx].to_numpy(),
#          good_value_socio4=pd_socio['n_pov'][socio1234_god_idx].to_numpy(), bad_value_socio4=pd_socio['n_pov'][socio1234_bad_idx].to_numpy())


print(len(socio1234_god_idx), len(socio1234_bad_idx))

