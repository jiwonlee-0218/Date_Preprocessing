import nibabel as nib
from glob import glob


def time_point_check(path):
    list = sorted(glob(path))


    for i in list:

        n1 = nib.load(i)
        print(n1.shape)


#time_point_check(path='/mnt/hcp/homes/jwlee/HCP_train_data/rfMRI_REST1_LR/Atlas.dtseries/AAL_116/*')
time_point_check(path='/mnt/hcp/homes/jwlee/HCP_train_data/rfMRI_REST1_LR/Atlas_hp2000_clean.dtseries/aparc/*')

