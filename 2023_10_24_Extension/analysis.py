import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import pandas as pd
import numpy as np


# task_list = ['emotion', 'gambling', 'language', 'motor', 'relational', 'social', 'wm']
# sourcedir = '/DataCommon2/jwlee/2023_08_11_ST_Adaptive_Number_Of_Clusters/data'

# subtask_label_list = np.load('/home/jwlee/HMM/BrainNetFormer/data/subtask_labels_detail.npy', allow_pickle=True).item()
#
#
# for i in task_list:
#     print(i)
#     a = subtask_label_list[i][:176]
#     print(a)
#
#     print('-----------------------------------------')

# task_block_timing = pd.read_csv(os.path.join(argv.sourcedir, 'behavioral', f'hcp_taskrest_{task}.csv'))
# for i, timing in enumerate(task_block_timing):
#     task_block_timing[timing] *= (i + 1)


# ----------------------------------------------------------------------------------------------------------------------
# for task in task_list:
#     task_block_timing = pd.read_csv(os.path.join(sourcedir, 'subtask_labels', f'{task}_subtask_label.csv'))
#     print(task_block_timing.shape)
#     for i, timing in enumerate(task_block_timing):
#         task_block_timing[timing] *= (i + 1)
#     print(task_block_timing)






# ----------------------------------------------------------------------------------------------------------------------
targetdir_list = [
    # '/DataCommon2/jwlee/2023_08_11_ST_Adaptive_Number_Of_Clusters/updated_attention_FC_10/result/experiment_renewal2_1',
    # '/DataCommon2/jwlee/2023_08_11_ST_Adaptive_Number_Of_Clusters/updated_attention_FC_10/result/experiment_renewal2_1_eigen15',
    # '/DataCommon2/jwlee/2023_08_11_ST_Adaptive_Number_Of_Clusters/updated_attention_FC_10/result/experiment_renewal2_1_eigen15_new5',
    # '/DataCommon2/jwlee/2023_08_11_ST_Adaptive_Number_Of_Clusters/updated_attention_FC_10/result/experiment_renewal2_1_eigen15_new6',
    # '/DataCommon2/jwlee/2023_08_11_ST_Adaptive_Number_Of_Clusters/updated_attention_FC_10/result/experiment_renewal2_1_eigen20_new1',
    # '/DataCommon2/jwlee/2023_08_11_ST_Adaptive_Number_Of_Clusters/updated_attention_FC_10/result/experiment_renewal2_1_eigen20_new2',
    '/DataCommon2/jwlee/2023_08_11_ST_Adaptive_Number_Of_Clusters/updated_attention_FC_10/result/experiment_renewal2_1_eigen20_new3'
]

# task_list = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM']
task_list = ['GAMBLING', 'RELATIONAL']
temporal_result = {}
k_fold = 5


for targetdir in targetdir_list:
    for task in task_list:
        temporal_result[task] = np.concatenate([np.load(os.path.join(targetdir, 'attention', str(k), 'temporal-result', f'{task}.npy')) for k in range(k_fold)])
        tr = (temporal_result[task].mean(0) - np.min(temporal_result[task].mean(0))) / (np.max(temporal_result[task].mean(0)) - np.min(temporal_result[task].mean(0)))
        plt.imshow(tr, cmap='hot')
        plt.colorbar()
        plt.show()
        plt.close()



