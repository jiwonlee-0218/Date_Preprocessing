import numpy as np



data_path = '/DataRead/REST-meta-MDD/REST-meta-MDD-Phase1-Sharing/ROISignals_FunImgARCWF/830MDDvs771NC_cc200_Combat.npz'

conn_dict = np.load(data_path, allow_pickle=True)

conn_f = conn_dict['signal']