import numpy as np
import scipy.io as sio
from glob import glob
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
import nibabel as nib
import os

cite = ['BNI', 'EMC', 'ETH', 'GU', 'IP', 'IU', 'KKI', 'KUL', 'NYU_1', 'NYU_2', 'OHSU', 'ONRC', 'SDSU', 'SU',
        'TCD', 'UCD', 'UCLA', 'UCLA_Long', 'UCLA_Long_follow', 'UPSM_Long', 'UPSM_Long_follow', 'USM', 'U_MIA']
main_path = '/mnt/47dd34f8-f6bf-4a5d-9a2c-fac4499356bc/ABIDE2/211109_preprocessed_using_dparsf_by_JI/'
atlas_dir = 'CC200'
atlas_path = nib.load('/home/j/Desktop/Jaein_code/atlas/atlas_61_73_61/rCC200.nii')

# for i in cite:
#         cite_path = data_path + i + '/ROIs/' + atlas_dir
#         os.makedirs(os.path.join(cite_path, 'global'), exist_ok=True)
#         os.makedirs(os.path.join(cite_path, 'noglobal'), exist_ok=True)

# /mnt/47dd34f8-f6bf-4a5d-9a2c-fac4499356bc/ABIDE2/211109_preprocessed_using_dparsf_by_JI/BNI/FunImgARCWSF    -- noglobal data
# /mnt/47dd34f8-f6bf-4a5d-9a2c-fac4499356bc/ABIDE2/211109_preprocessed_using_dparsf_by_JI/BNI/FunImgARglobalCWSF    -- global data

dir_type = ['global', 'noglobal']
data_type = ['FunImgARglobalCWSF', 'FunImgARCWSF']
for i in cite:
        for num, j in enumerate(dir_type):
                cite_path = main_path + i + '/ROIs/' + atlas_dir
                save_path = os.path.join(cite_path, j)
                print(num)


                data_path = os.path.join(main_path, i, data_type[num]) + '/*/*'



                for fn in sorted(glob(data_path)):

                        sub = nib.load(fn)
                        # print(fn.split('/')[-4], sub.shape)


                        sub_id = fn.split('/')[-2]

                        masker = NiftiLabelsMasker(labels_img=atlas_path, standardize=True)

                        t1 = masker.fit_transform(sub)
                        print(t1.shape)

                        save_path = os.path.join(cite_path, j) + '/' + sub_id+'.mat'
                        sio.savemat(save_path, {'ROI': t1})
                        print(save_path, '   done!!!')
                print()








