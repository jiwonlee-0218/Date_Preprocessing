import nibabel as nib
from nilearn.input_data import NiftiLabelsMasker
import scipy.io as sio
from glob import glob
import os.path

''' all subjects are 1113 '''

''' missing Results directory '''
#for sub in glob('/mnt/hcp/HCP/*'):
#    try:

#        file = sub + '/MNINonLinear/Results'

#        if os.path.isdir(file):
#            continue
#        else:
#            subject = sub.split('/')
#            subject = subject[-1]
#            print(subject + ' missing Results directory')

#    except:
#        pass


''' missing rfMRI_REST1_LR directory '''
#for sub in glob('/mnt/hcp/HCP/*'):
#    try:
#        file = sub + '/MNINonLinear/Results'

#        if os.path.isdir(file):
#            file2 = sub + '/MNINonLinear/Results/rfMRI_REST1_LR'
#
#            if os.path.isdir(file2):
#                continue
#            else:
#                subject = sub.split('/')
#                subject = subject[-1]
#                print(subject + ' missing rfMRI_REST1_LR directory')
#        else:
#            continue
#    except:
#        pass


''' missing rfMRI_REST1_RL directory '''
#for sub in glob('/mnt/hcp/HCP/*'):
#    try:
#        file = sub + '/MNINonLinear/Results'

#        if os.path.isdir(file):
#            file2 = sub + '/MNINonLinear/Results/rfMRI_REST1_RL'

#            if os.path.isdir(file2):
#                continue
#            else:
#                subject = sub.split('/')
#                subject = subject[-1]
#                print(subject + ' missing rfMRI_REST1_RL directory')
#        else:
#            continue
#    except:
#        pass


''' missing rfMRI_REST2_LR directory '''
#for sub in glob('/mnt/hcp/HCP/*'):
#    try:
#        file = sub + '/MNINonLinear/Results'

#        if os.path.isdir(file):
#            file2 = sub + '/MNINonLinear/Results/rfMRI_REST2_LR'

#            if os.path.isdir(file2):
#                continue
#            else:
#                subject = sub.split('/')
#                subject = subject[-1]
#                print(subject + ' missing rfMRI_REST2_LR directory')
#        else:
#            continue
#    except:
#        pass


''' missing tfMRI_MOTOR_LR directory '''
#for sub in glob('/mnt/hcp/HCP/*'):
#    try:
#        file = sub + '/MNINonLinear/Results'

#        if os.path.isdir(file):
#            file2 = sub + '/MNINonLinear/Results/tfMRI_MOTOR_LR'

#            if os.path.isdir(file2):
#                continue
#            else:
#                subject = sub.split('/')
#                subject = subject[-1]
#                print(subject + ' missing tfMRI_MOTOR_LR directory')
#        else:
#            continue
#    except:
#        pass

print()
print()
print()


''' missing tfMRI_MOTOR_RL directory '''
#for sub in glob('/mnt/hcp/HCP/*'):
#    try:
#        file = sub + '/MNINonLinear/Results'

#        if os.path.isdir(file):
#            file2 = sub + '/MNINonLinear/Results/tfMRI_MOTOR_RL'

#            if os.path.isdir(file2):
#                continue
#            else:
#                subject = sub.split('/')
#                subject = subject[-1]
#                print(subject + ' missing tfMRI_MOTOR_RL directory')
#        else:
#            continue
#    except:
#        pass



''' missing tfMRI_MOTOR_LR.nii.gz file '''
for sub in glob('/mnt/hcp/HCP/*'):
    try:
        file = sub + '/MNINonLinear/Results'

        if os.path.isdir(file):
            file2 = sub + '/MNINonLinear/Results/tfMRI_MOTOR_LR'

            if os.path.isdir(file2):
                file3 = sub + '/MNINonLinear/Results/tfMRI_MOTOR_LR/tfMRI_MOTOR_LR.nii.gz'

                if os.path.isfile(file3):
                    continue

                else:
                    subject = sub.split('/')
                    subject = subject[-1]
                    print(subject + ' missing tfMRI_MOTOR_LR.nii.gz file')


            else:
                continue
        else:
            continue
    except:
        pass