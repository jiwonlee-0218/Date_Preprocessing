from nilearn.datasets import fetch_atlas_aal
from nilearn.datasets import fetch_atlas_destrieux_2009
from nilearn.datasets import fetch_atlas_yeo_2011
from nilearn.datasets import fetch_atlas_harvard_oxford
import nibabel as nib


# aal_atlas = fetch_atlas_aal(data_dir='/media/12T/practice/atlas_test')
#
# destrieux_2009_atlas = fetch_atlas_destrieux_2009(data_dir='/media/12T/practice/atlas_test')
#
# yeo_atlas = fetch_atlas_yeo_2011(data_dir='/media/12T/practice/atlas_test')
#
# atlas_name_list = ['cort-maxprob-thr0-1mm', 'cort-maxprob-thr0-2mm', 'cort-maxprob-thr25-1mm',
# 'cort-maxprob-thr25-2mm', 'cort-maxprob-thr50-1mm', 'cort-maxprob-thr50-2mm', 'cort-prob-1mm', 'cort-prob-2mm',
# 'cortl-maxprob-thr0-1mm', 'cortl-maxprob-thr0-2mm', 'cortl-maxprob-thr25-1mm', 'cortl-maxprob-thr25-2mm', 'cortl-maxprob-thr50-1mm',
# 'cortl-maxprob-thr50-2mm', 'cortl-prob-1mm', 'cortl-prob-2mm', 'sub-maxprob-thr0-1mm', 'sub-maxprob-thr0-2mm', 'sub-maxprob-thr25-1mm', 'sub-maxprob-thr25-2mm', 'sub-maxprob-thr50-1mm', 'sub-maxprob-thr50-2mm', 'sub-prob-1mm', 'sub-prob-2mm']
# for atlas_name_list in atlas_name_list:
#     ho_atlas = fetch_atlas_harvard_oxford(atlas_name=atlas_name_list, data_dir='/media/12T/practice/atlas_test')


# atlas folder
cc200 = nib.load('/media/12T/practice/atlas/cc200_roi_atlas.nii.gz').get_fdata() #(63, 75, 61)
cc400 = nib.load('/media/12T/practice/atlas/cc400_roi_atlas.nii.gz').get_fdata() #(63, 75, 61)
dos = nib.load('/media/12T/practice/atlas/dos160_roi_atlas.nii.gz').get_fdata() #(61, 73, 61)
yeo = nib.load('/media/12T/practice/atlas/yeo_atlas_2mm.nii.gz').get_fdata() #(91, 109, 91)-2mm
yeo17 = nib.load('/media/12T/practice/atlas/yeo_atlas_2mm_17.nii.gz').get_fdata() #(91, 109, 91)-2mm


# atlas test folder download from https://nilearn.github.io/stable/modules/datasets.html
aal = nib.load('/media/12T/practice/atlas_test/aal_SPM12/aal/atlas/AAL.nii').get_fdata() #(91, 109, 91)-2mm
destrieux_2009 = nib.load('/media/12T/practice/atlas_test/destrieux_2009/destrieux2009_rois.nii.gz').get_fdata() #(76, 93, 76)
HO = nib.load('/media/12T/practice/atlas_test/fsl_HO/data/atlases/HarvardOxford/HarvardOxford-cortl-maxprob-thr0-2mm.nii.gz').get_fdata() #(91, 109, 91)-2mm
# HO = nib.load('/media/12T/practice/atlas_test/fsl_HO/data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr0-2mm.nii.gz').get_fdata() #(91, 109, 91)-2mm
yeo = nib.load('/media/12T/practice/atlas_test/yeo_2011/Yeo_JNeurophysiol11_MNI152/FSL_MNI152_FreeSurferConformed_1mm.nii.gz').get_fdata()

