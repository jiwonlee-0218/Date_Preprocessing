import numpy as np
import pandas as pd
import random

main_path = '/home/jwlee/HMM/deep_temporal_clustering/check_please/Data/'
vari_path = main_path + 'var0525_0929.csv'

# Load connectivity dataset
conn_path = main_path + 'con_aparc_count_nanx.csv'
conn = pd.read_csv(conn_path).to_numpy()
conn_subj = conn[:, 1]
conn_data = conn[:, 2:]

# Load social variable dataset
pd_socio = pd.read_csv(vari_path, usecols=['subjectkey', 'i_educ', 'i_income', 'n_pov', 'n_edu_h', 'race.ethnicity'])
scio_subj = pd_socio['subjectkey'].to_numpy()

assert (conn_subj == scio_subj).all()


def mk_data(num_race, socio_var, top25, bot25):

    """
    num_race:  1 (white) 2 (black) 3 (hispanic) 4 (asian) 5 (etc)
    socio_var: 'i_educ', 'i_income', 'n_pov', 'n_edu_h'
    top25: top 25% criterion (good environment)
    bot24: bottom 25% criterion (bad environment)
    """

    if socio_var == 'n_pov': # high => 이웃들중에 가난한 비율이 높다는 것
        bad = (pd_socio[(pd_socio['race.ethnicity'] == num_race) & (pd_socio[socio_var] >= top25)]).index
        good = (pd_socio[(pd_socio['race.ethnicity'] == num_race) & (pd_socio[socio_var] <= bot25)]).index
    else:
        good = (pd_socio[(pd_socio['race.ethnicity'] == num_race) & (pd_socio[socio_var] >= top25)]).index
        bad = (pd_socio[(pd_socio['race.ethnicity'] == num_race) & (pd_socio[socio_var] <= bot25)]).index


    for i in range(10):
        index = random.choice(good)
        if pd_socio['race.ethnicity'][index] == num_race:
            print('good')


    # np.savez('race%d_%s_thr25.npz' % (num_race, socio_var), good_idx=good.to_numpy(), bad_idx=bad.to_numpy(),
    #          good_subkey=scio_subj[good], bad_subkey=scio_subj[bad],
    #          good_value=pd_socio[socio_var][good].to_numpy(), bad_value=pd_socio[socio_var][bad].to_numpy())

    # np.savez('race%d_%s_thr10.npz' % (num_race, socio_var), good_idx=good.to_numpy(), bad_idx=bad.to_numpy(),
    #          good_subkey=scio_subj[good], bad_subkey=scio_subj[bad],
    #          good_value=pd_socio[socio_var][good].to_numpy(), bad_value=pd_socio[socio_var][bad].to_numpy())
    # np.savez('race%d_%s_top%d_bot%d.npz' % (num_race, socio_var, top25, bot25), good_idx=good.to_numpy(), bad_idx=bad.to_numpy(),
    #          good_subkey=scio_subj[good], bad_subkey=scio_subj[bad],
    #          good_value=pd_socio[socio_var][good].to_numpy(), bad_value=pd_socio[socio_var][bad].to_numpy())

    return print(good.shape[0], bad.shape[0])


print("====== i educ ======")
mk_data(1, 'i_educ', 19, 16)
mk_data(2, 'i_educ', 17, 13)
mk_data(3, 'i_educ', 18, 13)

# mk_data(1, 'i_educ', 20, 14)
# mk_data(2, 'i_educ', 19, 12)
# mk_data(3, 'i_educ', 19, 10)

# mk_data(1, 'i_educ', 21, 12)


print("====== i income ======")
mk_data(1, 'i_income', 9, 8)
mk_data(2, 'i_income', 8, 3)
mk_data(3, 'i_income', 8, 5)

# mk_data(1, 'i_income', 10, 6)
# mk_data(2, 'i_income', 9, 1)
# mk_data(3, 'i_income', 9, 2)

print("====== n pov ======")
mk_data(1, 'n_pov', 9.18, 2.26)
mk_data(2, 'n_pov', 31.84, 9.35)
mk_data(3, 'n_pov', 23.09, 5.99)

# mk_data(1, 'n_pov', 15.77, 0.9599)
# mk_data(2, 'n_pov', 43.669998, 4.25)
# mk_data(3, 'n_pov', 34.869999, 2.83)

print("====== n edu h ======")
mk_data(1, 'n_edu_h', 97.1, 90.69)
mk_data(2, 'n_edu_h', 91.55, 77.38)
mk_data(3, 'n_edu_h', 92.33, 71.09)

# mk_data(1, 'n_edu_h', 98.360001, 84.989998)
# mk_data(2, 'n_edu_h', 94.690002, 69.25)
# mk_data(3, 'n_edu_h', 96.150002, 53.68)

