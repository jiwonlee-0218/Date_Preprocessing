import numpy as np
import nibabel as nib
from glob import glob
import os
import shutil
from datetime import datetime

''' copy file '''
# file_path = '/mnt/47dd34f8-f6bf-4a5d-9a2c-fac4499356bc/preprocessing_fMRI/ADHD200/*'
# func_path = '/mnt/47dd34f8-f6bf-4a5d-9a2c-fac4499356bc/preprocessing_fMRI/Preprocessed_ADHD200_Real_v2/FunImg'
# T1_path = '/mnt/47dd34f8-f6bf-4a5d-9a2c-fac4499356bc/preprocessing_fMRI/Preprocessed_ADHD200_Real_v2/T1Img'
#
# os.makedirs(func_path, exist_ok=True)
# os.makedirs(T1_path, exist_ok=True)
#
# for idx, sub in enumerate(sorted(glob(file_path))):
#     try:
#         anat_file = sub+'/anat_1/NIfTI/rest.nii.gz'
#         func_file = sub+'/rest_1/NIfTI/rest.nii.gz'
#
#
#         os.makedirs(os.path.join(T1_path, f'sub_{str(idx).zfill(3)}'), exist_ok=True)
#         os.makedirs(os.path.join(func_path, f'sub_{str(idx).zfill(3)}'), exist_ok=True)
#
#
#
#         shutil.copyfile(anat_file, os.path.join(T1_path, f'sub_{str(idx).zfill(3)}')+f'/sub{str(idx).zfill(3)}_'+'anat.nii.gz')
#         print(idx, anat_file)
#         shutil.copyfile(func_file, os.path.join(func_path, f'sub_{str(idx).zfill(3)}')+f'/sub{str(idx).zfill(3)}_'+'func.nii.gz')
#         print(idx, func_file)
#         print()
#
#
#     except:
#         pass





''' remove file '''
# exc_word = 'Covremoved'
# subject = 'sub_530'
# FunImgAR = '/mnt/47dd34f8-f6bf-4a5d-9a2c-fac4499356bc/preprocessing_fMRI/Preprocessed_ADHD200_OHSU/FunImgAR'
# T1ImgNewSegment = '/mnt/47dd34f8-f6bf-4a5d-9a2c-fac4499356bc/preprocessing_fMRI/Preprocessed_ADHD200_OHSU/T1ImgNewSegment'
# RealignParameter = '/mnt/47dd34f8-f6bf-4a5d-9a2c-fac4499356bc/preprocessing_fMRI/Preprocessed_ADHD200_OHSU/RealignParameter'
# SegmentationMasks = '/mnt/47dd34f8-f6bf-4a5d-9a2c-fac4499356bc/preprocessing_fMRI/Preprocessed_ADHD200_OHSU/Masks/SegmentationMasks'
# WarpedMasks = '/mnt/47dd34f8-f6bf-4a5d-9a2c-fac4499356bc/preprocessing_fMRI/Preprocessed_ADHD200_OHSU/Masks/WarpedMasks'
#
#
# # 폴더 이동
# today = datetime.now()
# now = str(today)
# now = now.split(".")[0]
# now = now.replace("-","").replace(" ","_").replace(":","")
# FunImgAR_exc_folder = os.path.join('/mnt/47dd34f8-f6bf-4a5d-9a2c-fac4499356bc/preprocessing_fMRI/Preprocessed_ADHD200_OHSU/FunImgAR_exc', now)
# os.makedirs(FunImgAR_exc_folder)
#
#
# for path in sorted(glob(f"{FunImgAR}/*{exc_word}*")):
#     shutil.rmtree(path)
#     print(path)
#
# for path in sorted(glob(f"{FunImgAR}/*{subject}*")):
#     shutil.move(path, FunImgAR_exc_folder)
#     print(path)
#
# print()








# # 폴더 이동
# for path in glob(f"{T1ImgNewSegment}/{subject}*"):
#     shutil.move(path, '/mnt/47dd34f8-f6bf-4a5d-9a2c-fac4499356bc/preprocessing_fMRI/Preprocessed_ADHD200_OHSU/T1ImgNewSegment_exc')
#     print(path)
#
# print()
#
# # 폴더 이동
# for path in glob(f"{RealignParameter}/{subject}*"):
#     shutil.move(path, '/mnt/47dd34f8-f6bf-4a5d-9a2c-fac4499356bc/preprocessing_fMRI/Preprocessed_ADHD200_OHSU/RealignParameter_exc')
#     print(path)
#
# print()
#
#
# # 파일 이동
# for path in glob(f"{SegmentationMasks}/*{subject}*"):
#     shutil.move(path, os.path.join('/mnt/47dd34f8-f6bf-4a5d-9a2c-fac4499356bc/preprocessing_fMRI/Preprocessed_ADHD200_OHSU/Masks_exc/SegmentationMasks_exc', path.split('/')[-1]))
#     print(path)
#
# print()
#
#
# # 파일 이동
# for path in glob(f"{WarpedMasks}/*{subject}*"):
#     shutil.move(path, os.path.join('/mnt/47dd34f8-f6bf-4a5d-9a2c-fac4499356bc/preprocessing_fMRI/Preprocessed_ADHD200_OHSU/Masks_exc/WarpedMasks_exc', path.split('/')[-1]))
#     print(path)






''' rename file '''
# FunImgAR = '/mnt/47dd34f8-f6bf-4a5d-9a2c-fac4499356bc/preprocessing_fMRI/Preprocessed_ADHD200_Real/FunImgAR'
# exc_word = 'rrasub'
#
#
# for path in sorted(glob(f"{FunImgAR}/*/*{exc_word}*")):
#     sub_dir = path.split('/')[-2]
#     sub_file = path.split('/')[-1]
#
#     new_file_name = sub_file[1:]
#
#     old_file = os.path.join(FunImgAR, sub_dir, sub_file)
#     new_file = os.path.join(FunImgAR, sub_dir, new_file_name)
#     os.rename(old_file, new_file)
#     print(path)



''' size check '''
# FunImgAR = '/mnt/47dd34f8-f6bf-4a5d-9a2c-fac4499356bc/preprocessing_fMRI/Preprocessed_ADHD200_Real/FunImgAR'
# exc_word = 'rasub'
# import nibabel as nib
#
# for path in sorted(glob(f"{FunImgAR}/*/*{exc_word}*")):
#     a = nib.load(path)
#     print(path.split('/')[-1][:8],': ',a.shape)





''' SITE와 SUBxxx를 비교 즉 매칭이 어떻게 되는지 확인 '''
# file_path = '/mnt/47dd34f8-f6bf-4a5d-9a2c-fac4499356bc/preprocessing_fMRI/ADHD200/*'
# func_path = '/mnt/47dd34f8-f6bf-4a5d-9a2c-fac4499356bc/preprocessing_fMRI/Preprocessed_ADHD200_Real_v2/FunImg'
# T1_path = '/mnt/47dd34f8-f6bf-4a5d-9a2c-fac4499356bc/preprocessing_fMRI/Preprocessed_ADHD200_Real_v2/T1Img'
#
#
#
# for idx, sub in enumerate(sorted(glob(file_path))):
#     try:
#         anat_file = sub+'/anat_1/NIfTI/rest.nii.gz'
#         func_file = sub+'/rest_1/NIfTI/rest.nii.gz'
#
#
#         # print(idx, os.path.join(T1_path, f'sub_{str(idx).zfill(3)}'), anat_file)
#         print(idx, f'sub_{str(idx).zfill(3)}', func_file.split('/')[-4])
#
#
#
#     except:
#         pass



# RealignParameter_path = '/mnt/47dd34f8-f6bf-4a5d-9a2c-fac4499356bc/preprocessing_fMRI/Preprocessed_ADHD200_Peking_GSR/RealignParameter'
# exc_word = 'wmeana'
# for path in sorted(glob(f"{RealignParameter_path}/*/*{exc_word}*")):
#     os.remove(path)
#     print(path)