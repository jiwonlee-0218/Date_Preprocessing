import nibabel as nib
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
import scipy.io as sio
from glob import glob
import os
import os.path
import numpy as np
import shutil

list=['LR','RL']

for li in list:
    for fn in glob('/mnt/hcp/homes/jwlee/HCP_extract/*'):
        try:
            print(fn)
            sub1 = fn.split('/')
            print(sub1[-1])

            # file_name = 'AAL_116_'+sub1[-1]+'.mat'
            # print(file_name)
            path = '/mnt/hcp/homes/jwlee/HCP_extract/'+sub1[-1]+'/WM/'+li+'/*'
            print(path)

            file_list = glob(path)
            file_list_mat = [file for file in file_list if file.endswith('AAL_116_'+sub1[-1]+'.mat')]
            print(file_list_mat)



            real_path = file_list_mat[0]
            copy_path = '/media/12T/practice/AAL_116_WM/'+li+'/'+sub1[-1]+'.mat'

            shutil.copy(real_path, copy_path)

        except:
            continue

