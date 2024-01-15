import scipy.io as sio
from glob import glob
import os
import os.path
import numpy as np
import shutil

path = '/mnt/hcp/HCP/*'
task_name = ['EMOTION','GAMBLING','LANGUAGE','MOTOR','RELATIONAL','SOCIAL','WM']
# lr = ['LR', 'RL']

'''디렉토리 생성'''
# for i in sorted(glob(path)):
#
#     try:
#         name = i.split('/')[-1]
#         os.makedirs('/mnt/hcp/homes/jwlee/HCP_extract/' + name, exist_ok=True)
#
#         for j in task_name:
#             os.makedirs('/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/' + j, exist_ok=True)
#             for p in lr:
#                 os.makedirs('/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/' + j + '/' + p, exist_ok=True)
#     except:
#         pass


# emotion_lr = ['tfMRI_EMOTION_LR','tfMRI_EMOTION_RL']
# gambling_lr = ['tfMRI_GAMBLING_LR','tfMRI_GAMBLING_RL']
# language_lr = ['tfMRI_LANGUAGE_LR','tfMRI_LANGUAGE_RL']
# motor_lr = ['tfMRI_MOTOR_LR','tfMRI_MOTOR_RL']
# relational_lr = ['tfMRI_RELATIONAL_LR','tfMRI_RELATIONAL_RL']
# social_lr = ['tfMRI_SOCIAL_LR','tfMRI_SOCIAL_RL']
# wm_lr = ['tfMRI_WM_LR','tfMRI_WM_RL']

'''emotion raw file copy'''
# for sub in sorted(glob(path)):
#     try:
#         name = sub.split('/')[-1]
#
#
#         for lr in emotion_lr:
#             try:
#                 if lr == 'tfMRI_EMOTION_LR':
#                     if os.path.isfile('/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/EMOTION/LR/tfMRI_EMOTION_LR.nii.gz'):
#                         print('/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/EMOTION/LR/tfMRI_EMOTION_LR.nii.gz', " 파일이 존재합니다.")
#                         continue
#                     source = '/mnt/hcp/HCP/' + name + '/MNINonLinear/Results/' + lr + '/tfMRI_EMOTION_LR.nii.gz'
#                     desgination = '/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/EMOTION/LR'
#                 else:
#                     if os.path.isfile('/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/EMOTION/RL/tfMRI_EMOTION_RL.nii.gz'):
#                         print('/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/EMOTION/RL/tfMRI_EMOTION_RL.nii.gz', " 파일이 존재합니다.")
#                         continue
#                     source = '/mnt/hcp/HCP/' + name + '/MNINonLinear/Results/' + lr + '/tfMRI_EMOTION_RL.nii.gz'
#                     desgination = '/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/EMOTION/RL'
#
#                 shutil.copy(source, desgination)
#                 print(name + ' ' + lr + ' done!!')
#             except:
#                 pass
#     except:
#         pass


'''gambling raw file copy'''
# for sub in sorted(glob(path)):
#     try:
#         name = sub.split('/')[-1]
#
#
#         for lr in gambling_lr:
#             try:
#                 if lr == 'tfMRI_GAMBLING_LR':
#                     if os.path.isfile('/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/GAMBLING/LR/tfMRI_GAMBLING_LR.nii.gz'):
#                         print('/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/GAMBLING/LR/tfMRI_GAMBLING_LR.nii.gz', " 파일이 존재합니다.")
#                         continue
#                     source = '/mnt/hcp/HCP/' + name + '/MNINonLinear/Results/' + lr + '/tfMRI_GAMBLING_LR.nii.gz'
#                     desgination = '/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/GAMBLING/LR'
#                 else:
#                     if os.path.isfile('/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/GAMBLING/RL/tfMRI_GAMBLING_RL.nii.gz'):
#                         print('/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/GAMBLING/RL/tfMRI_GAMBLING_RL.nii.gz', " 파일이 존재합니다.")
#                         continue
#                     source = '/mnt/hcp/HCP/' + name + '/MNINonLinear/Results/' + lr + '/tfMRI_GAMBLING_RL.nii.gz'
#                     desgination = '/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/GAMBLING/RL'
#
#                 shutil.copy(source, desgination)
#                 print(name + ' ' + lr + ' done!!')
#
#             except:
#                 pass
#     except:
#         pass

'''language raw file copy'''
# for sub in sorted(glob(path)):
#     try:
#         name = sub.split('/')[-1]
#
#
#         for lr in language_lr:
#             try:
#                 if lr == 'tfMRI_LANGUAGE_LR':
#                     if os.path.isfile('/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/LANGUAGE/LR/tfMRI_LANGUAGE_LR.nii.gz'):
#                         print('/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/LANGUAGE/LR/tfMRI_LANGUAGE_LR.nii.gz', " 파일이 존재합니다.")
#                         continue
#                     source = '/mnt/hcp/HCP/' + name + '/MNINonLinear/Results/' + lr + '/tfMRI_LANGUAGE_LR.nii.gz'
#                     desgination = '/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/LANGUAGE/LR'
#                 else:
#                     if os.path.isfile('/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/LANGUAGE/RL/tfMRI_LANGUAGE_RL.nii.gz'):
#                         print('/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/LANGUAGE/RL/tfMRI_LANGUAGE_RL.nii.gz', " 파일이 존재합니다.")
#                         continue
#                     source = '/mnt/hcp/HCP/' + name + '/MNINonLinear/Results/' + lr + '/tfMRI_LANGUAGE_RL.nii.gz'
#                     desgination = '/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/LANGUAGE/RL'
#
#                 shutil.copy(source, desgination)
#                 print(name + ' ' + lr + ' done!!')
#
#             except:
#                 pass
#     except:
#         pass



'''motor raw file copy'''
# for sub in sorted(glob(path)):
#     try:
#         name = sub.split('/')[-1]
#
#
#         for lr in motor_lr:
#             try:
#                 if lr == 'tfMRI_MOTOR_LR':
#                     if os.path.isfile('/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/MOTOR/LR/tfMRI_MOTOR_LR.nii.gz'):
#                         print('/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/MOTOR/LR/tfMRI_MOTOR_LR.nii.gz', " 파일이 존재합니다.")
#                         continue
#                     source = '/mnt/hcp/HCP/' + name + '/MNINonLinear/Results/' + lr + '/tfMRI_MOTOR_LR.nii.gz'
#                     desgination = '/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/MOTOR/LR'
#                 else:
#                     if os.path.isfile('/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/MOTOR/RL/tfMRI_MOTOR_RL.nii.gz'):
#                         print('/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/MOTOR/RL/tfMRI_MOTOR_RL.nii.gz', " 파일이 존재합니다.")
#                         continue
#                     source = '/mnt/hcp/HCP/' + name + '/MNINonLinear/Results/' + lr + '/tfMRI_MOTOR_RL.nii.gz'
#                     desgination = '/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/MOTOR/RL'
#
#                 shutil.copy(source, desgination)
#                 print(name + ' ' + lr + ' done!!')
#             except:
#                 pass
#     except:
#         pass


'''relational raw file copy'''
# for sub in sorted(glob(path)):
#     try:
#         name = sub.split('/')[-1]
#
#
#         for lr in relational_lr:
#             try:
#                 if lr == 'tfMRI_RELATIONAL_LR':
#                     if os.path.isfile('/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/RELATIONAL/LR/tfMRI_RELATIONAL_LR.nii.gz'):
#                         print('/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/RELATIONAL/LR/tfMRI_RELATIONAL_LR.nii.gz', " 파일이 존재합니다.")
#                         continue
#                     source = '/mnt/hcp/HCP/' + name + '/MNINonLinear/Results/' + lr + '/tfMRI_RELATIONAL_LR.nii.gz'
#                     desgination = '/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/RELATIONAL/LR'
#                 else:
#                     if os.path.isfile('/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/RELATIONAL/RL/tfMRI_RELATIONAL_RL.nii.gz'):
#                         print('/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/RELATIONAL/RL/tfMRI_RELATIONAL_RL.nii.gz', " 파일이 존재합니다.")
#                         continue
#                     source = '/mnt/hcp/HCP/' + name + '/MNINonLinear/Results/' + lr + '/tfMRI_RELATIONAL_RL.nii.gz'
#                     desgination = '/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/RELATIONAL/RL'
#
#                 shutil.copy(source, desgination)
#                 print(name + ' ' + lr + ' done!!')
#             except:
#                 pass
#     except:
#         pass

'''social raw file copy'''
# for sub in sorted(glob(path)):
#     try:
#         name = sub.split('/')[-1]
#
#
#         for lr in social_lr:
#             try:
#                 if lr == 'tfMRI_SOCIAL_LR':
#                     if os.path.isfile('/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/SOCIAL/LR/tfMRI_SOCIAL_LR.nii.gz'):
#                         print('/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/SOCIAL/LR/tfMRI_SOCIAL_LR.nii.gz', " 파일이 존재합니다.")
#                         continue
#
#                     source = '/mnt/hcp/HCP/' + name + '/MNINonLinear/Results/' + lr + '/tfMRI_SOCIAL_LR.nii.gz'
#                     desgination = '/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/SOCIAL/LR'
#                 else:
#                     if os.path.isfile('/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/SOCIAL/RL/tfMRI_SOCIAL_RL.nii.gz'):
#                         print('/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/SOCIAL/RL/tfMRI_SOCIAL_RL.nii.gz', " 파일이 존재합니다.")
#                         continue
#                     source = '/mnt/hcp/HCP/' + name + '/MNINonLinear/Results/' + lr + '/tfMRI_SOCIAL_RL.nii.gz'
#                     desgination = '/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/SOCIAL/RL'
#
#                 shutil.copy(source, desgination)
#                 print(name + ' ' + lr + ' done!!')
#             except:
#                 pass
#     except:
#         pass


'''wm raw file copy'''
# for sub in sorted(glob(path)):
#     try:
#         name = sub.split('/')[-1]
#
#
#         for lr in wm_lr:
#             try:
#                 if lr == 'tfMRI_WM_LR':
#                     if os.path.isfile('/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/WM/LR/tfMRI_WM_LR.nii.gz'):
#                         print('/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/WM/LR/tfMRI_WM_LR.nii.gz', " 파일이 존재합니다.")
#                         continue
#
#                     source = '/mnt/hcp/HCP/' + name + '/MNINonLinear/Results/' + lr + '/tfMRI_WM_LR.nii.gz'
#                     desgination = '/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/WM/LR'
#                 else:
#                     if os.path.isfile('/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/WM/RL/tfMRI_WM_RL.nii.gz'):
#                         print('/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/WM/RL/tfMRI_WM_RL.nii.gz', " 파일이 존재합니다.")
#                         continue
#                     source = '/mnt/hcp/HCP/' + name + '/MNINonLinear/Results/' + lr + '/tfMRI_WM_RL.nii.gz'
#                     desgination = '/mnt/hcp/homes/jwlee/HCP_extract/' + name + '/WM/RL'
#
#                 shutil.copy(source, desgination)
#                 print(name + ' ' + lr + ' done!!')
#             except:
#                 pass
#     except:
#         pass








for sub in sorted(glob(path)):
    try:
        name = sub.split('/')[-1]


        for task in task_name:
            try:
                file_name = 'tfMRI_' + task + '_LR'
                source = '/mnt/hcp/HCP/' + name + '/MNINonLinear/Results/' + file_name +'/' +file_name+'.nii.gz'
                desgination = '/home/djk/Desktop/TASK/'+task+'/'+name+'.nii.gz'

                shutil.copy(source, desgination)
                print(desgination + ' done!!')
            except:
                pass
    except:
        pass










