import numpy as np
import csv
import pandas as pd
import os
import shutil
import glob


"""
Check the realignment parameters for quality control
Criterion: FD mean > 0.2 -> abandon
by Eunsong Kang
"""

criterion = 'mean'
path1 = '/media/j/fMRI_etc_data/ADHD/Preprocessed_ADHD200_GSR/'
qc_fail_list = []
with open('/media/j/fMRI_etc_data/ADHD/QC_headmotion_FD_fail_dparsf_JW_list.txt', 'a') as the_file:
    for file in sorted(glob.glob(path1 + '*/RealignParameter/*/FD_Power_*.txt')):
        with open(file) as f:
            cite = file.split('/')[-4].split('_')[-2]
            lines = f.readlines()
            all_values = [float(i) for i in lines]

            if criterion == 'mean':
                qc_v = np.array(all_values).mean()
            elif criterion == 'max':
                qc_v = np.max(np.array(all_values))
        if qc_v > 0.2:
            print(cite, os.path.basename(file), '{:.4f}'.format(qc_v))
            # the_file.write('{}\t{}\t{:.4f}\n'.format(cite,os.path.basename(file), qc_v))
        else:
            print(cite, os.path.basename(file))

the_file.close()



import numpy as np
import csv
import pandas as pd
import os
import shutil
import glob


# """
# Check the realignment parameters for quality control
# Criterion: FD mean > 0.2
# by Eunsong Kang
# """
#
# criterion = 'mean'
# path1 = '/DataRead/ADHD200/Preprocessed/dparsf_download_from_bisi/ADHD200_DPARSF/'
# qc_fail_list = []
# with open('/DataRead/ADHD200/Preprocessed/QC_headmotion_FD_fail_dparsf_bisi_list.txt', 'a') as the_file:
#     for file in sorted(glob.glob(path1 + '*/RealignParameter/*/FD_Power_*.txt')):
#         with open(file) as f:
#             lines = f.readlines()
#             all_values = [float(i) for i in lines]
#
#             if criterion == 'mean':
#                 qc_v = np.array(all_values).mean()
#             elif criterion == 'max':
#                 qc_v = np.max(np.array(all_values))
#         if qc_v > 0.2:
#             print(os.path.basename(file), '{:.4f}'.format(qc_v))
#             the_file.write('{}\t{:.4f}\n'.format(os.path.basename(file), qc_v))
# the_file.close()