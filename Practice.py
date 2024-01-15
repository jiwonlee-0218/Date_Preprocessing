import nibabel as nib
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import torch
from random import randrange




''' ROI x TIME -> .mat file '''

# path = '/media/12T/practice/atlas/aal_roi_atlas.nii.gz' # Automated Anatomical Labeling (AAL)_116
# path = '/media/12T/practice/atlas/yeo_atlas_2mm_17.nii.gz' # Yeo17_114
# path = '/media/12T/practice/atlas/yeo_atlas_2mm.nii.gz' # Yeo7_51
# path = '/media/12T/practice/atlas/cc200_roi_atlas.nii.gz' # Craddock200_200
# path = '/media/12T/practice/atlas/cc400_roi_atlas.nii.gz' # Craddock400_392
# path = '/media/12T/practice/atlas/ho_roi_atlas.nii.gz' # Harvard-Oxford (HO)_111



# atlas = nib.load(path)
# filename = '/mnt/hcp/HCP/100206/MNINonLinear/Results/tfMRI_EMOTION_LR/tfMRI_EMOTION_LR.nii.gz'
# n1 = nib.load(filename)
# aa = n1.get_fdata()
#
# masker = NiftiLabelsMasker(labels_img=atlas, standardize=True)
# t1 = masker.fit_transform(n1)
# print(t1.shape) #ROI x TIME

#### save matfile ####
# save_path = '/media/12T/practice/practice1.mat'  #경로 + .mat file
# sio.savemat(save_path, {'ROI': t1})




''' load .mat file '''
# matfile_path = '/media/12T/practice/practice1.mat'
# sub = sio.loadmat(matfile_path)
# sub = sub['ROI']
# print(sub.shape) #ROI x TIME







''' ROI x ROI -> .npy '''
# def connectivity(input, type='correlation', vectorization=True, fisher_t=False):
#     '''kind{“covariance”, “correlation”, “partial correlation”, “tangent”, “precision”}, optional'''
#     measure = ConnectivityMeasure(kind=type, vectorize=vectorization, discard_diagonal=True)
#     mt = measure.fit_transform(input)
#
#     if fisher_t == True:
#         for i in range(len(mt)):
#             mt[i][:] = np.arctanh(mt[i][:])
#     return mt
#
#
# matfile_path = '/media/12T/practice/practice1.mat'
# sub = sio.loadmat(matfile_path)
# sub = sub['ROI']
#
# sub = np.reshape(sub, ((1,) + sub.shape))
#
# fc = connectivity(sub, type='correlation', vectorization=False, fisher_t=False)


#### save npyfile ####
# np.save('/media/12T/practice/practice1', fc)  ## -> practice1.npy파일로 저장
# print('done!!')



''' load .npy file '''
# fc = np.load('/media/12T/practice/practice1.npy')
# print(fc[0].shape)
# plt.imshow(fc[0])
# plt.colorbar()
# plt.show()
# plt.close()







''' dynamic FC '''
def get_fc(timeseries, sampling_point, window_size, self_loop):
    fc = corrcoef(timeseries[sampling_point:sampling_point+window_size].T)
    if not self_loop: fc-= torch.eye(fc.shape[0])
    return fc

def get_minibatch_fc(minibatch_timeseries, sampling_point, window_size, self_loop):
    fc_list = []
    for timeseries in minibatch_timeseries:
        fc = get_fc(timeseries, sampling_point, window_size, self_loop)
        fc_list.append(fc)
    return torch.stack(fc_list)


def process_dynamic_fc(minibatch_timeseries, window_size, window_stride, dynamic_length=None, sampling_init=None, self_loop=True):

    if dynamic_length is None:
        dynamic_length = minibatch_timeseries.shape[1]
        sampling_init = 0
    else:
        if isinstance(sampling_init, int):
            assert minibatch_timeseries.shape[1] > sampling_init + dynamic_length
    assert sampling_init is None or isinstance(sampling_init, int)
    assert minibatch_timeseries.ndim==3
    assert dynamic_length > window_size

    if sampling_init is None:
        sampling_init = randrange(minibatch_timeseries.shape[1]-dynamic_length+1) #150-151 sampling_init = 0
    sampling_points = list(range(sampling_init, sampling_init+dynamic_length-window_size, window_stride)) #(0, 100, 3)

    minibatch_fc_list = [get_minibatch_fc(minibatch_timeseries, sampling_point, window_size, self_loop) for sampling_point in sampling_points]
    dynamic_fc = torch.stack(minibatch_fc_list, dim=1)

    return dynamic_fc, sampling_points


def corrcoef(x):
    mean_x = torch.mean(x, 1, keepdim=True)
    xm = x.sub(mean_x.expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())
    c = torch.clamp(c, -1.0, 1.0)
    return c



if __name__ == '__main__':

    # sub_list = ['/DataCommon/jwlee/AAL_116_EMOTION/LR/AAL_116_100206.mat', '/DataCommon/jwlee/AAL_116_EMOTION/LR/AAL_116_100307.mat', '/DataCommon/jwlee/AAL_116_EMOTION/LR/AAL_116_100408.mat']
    # x = []
    # for i in sub_list:
    #     sub = sio.loadmat(i)
    #     sub = sub['ROI']
    #     x.append(sub)
    #
    #
    # x = np.array(x)
    # x = torch.Tensor(x)

    matfile_path = '/DataCommon/jwlee/AAL_116_EMOTION/LR/AAL_116_100206.mat'
    sub = sio.loadmat(matfile_path)
    sub = sub['ROI']
    sub = np.reshape(sub, ((1,) + sub.shape))
    sub = torch.Tensor(sub)

    window_size = 14
    window_stride = 3


    dyn_a, sampling_points = process_dynamic_fc(sub, window_size, window_stride)




