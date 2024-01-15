import numpy as np
import scipy.io as sio
from os.path import join
from nilearn.connectome import ConnectivityMeasure
from glob import glob
from numpy import zeros, arange, corrcoef


def connectivity(input, type='correlation', vectorization=True, fisher_t=False):
    '''kind{“covariance”, “correlation”, “partial correlation”, “tangent”, “precision”}, optional'''
    measure = ConnectivityMeasure(kind=type, vectorize=vectorization, discard_diagonal=True)
    mt = measure.fit_transform(input)

    if fisher_t == True:
        for i in range(len(mt)):
            mt[i][:] = np.arctanh(mt[i][:])
    return mt




# K = []
#
# for fn in sorted(glob('/DataCommon/jwlee/AAL_116_EMOTION/LR/*')):
#     if fn == '/DataCommon/jwlee/AAL_116_EMOTION/LR/AAL_116_136126.mat':
#         print('except 131126')
#         continue
#
#     sub = sio.loadmat(fn)
#
#     arr = sub['ROI'][4:17,:]
#     arr = np.expand_dims(arr, 0)
#     K = connectivity(arr, type='correlation', vectorization=False, fisher_t=False)
#
#
#     list = [16, 34, 46, 63, 75, 92, 104, 121, 133, 151, 163]
#
#
#     for i in list:
#         a = sub['ROI'][i:i+13,:]
#
#         a =np.expand_dims(a, 0)
#         fc = connectivity(a, type='correlation', vectorization=False, fisher_t=False)
#
#         K = np.concatenate((K, fc), axis=0)
#
#     if fn == '/DataCommon/jwlee/AAL_116_EMOTION/LR/AAL_116_100206.mat':
#         triu_indices = np.triu(K[0], 1).nonzero()
#         vector_upper_trian = K[0][triu_indices[0], triu_indices[1]]
#         vector_upper_trian = np.expand_dims(vector_upper_trian, 0)
#
#         for i in range(1, 12):
#             triu_indices_p = np.triu(K[i], 1).nonzero()
#             vector_upper_trian_p = K[i][triu_indices_p[0], triu_indices_p[1]]
#             vector_upper_trian_p = np.expand_dims(vector_upper_trian_p, 0)
#
#             vector_upper_trian = np.concatenate((vector_upper_trian, vector_upper_trian_p), axis=0)  # (12, 6670)
#         vector_upper_trian_T = np.expand_dims(vector_upper_trian, 0)  # (1, 12, 6670)
#
#     else:
#         triu_indices = np.triu(K[0], 1).nonzero()
#         vector_upper_trian = K[0][triu_indices[0], triu_indices[1]]
#         vector_upper_trian = np.expand_dims(vector_upper_trian, 0) # (1, 6670)
#
#         for i in range(1, 12):
#             triu_indices_p = np.triu(K[i], 1).nonzero()
#             vector_upper_trian_p = K[i][triu_indices_p[0], triu_indices_p[1]]
#             vector_upper_trian_p = np.expand_dims(vector_upper_trian_p, 0)
#
#             vector_upper_trian = np.concatenate((vector_upper_trian, vector_upper_trian_p), axis=0)  # (12, 6670)
#
#         vector_upper_trian_Q = np.expand_dims(vector_upper_trian, 0)  # (1, 12, 6670)
#
#         vector_upper_trian_T = np.concatenate((vector_upper_trian_T, vector_upper_trian_Q), axis=0)
#
#
# print(vector_upper_trian_T)
# np.savez_compressed('/DataCommon/jwlee/EMOTION_LR/cluster_2_hcp_emotion_Fc', tfMRI_EMOTION_LR=vector_upper_trian_T)






def corr(series, size):
    slide = size # window size = 3, 3 of 176

    idx = arange(0, series.shape[0], slide) # (0, 150, 25) 0부터 150까지 25 단위씩
    corr_mats = zeros((idx.shape[0], 28)) # (6, 28)


    for w in range(idx.shape[0]):
        p = np.expand_dims(series[0 + idx[w]: size + idx[w],:], 0) # (1, 2, 116) = (1, t, 116)
        # corr_mats[w, :, :] = connectivity(p, type='correlation', vectorization=True, fisher_t=False)
        corr_mats[w, :] = connectivity(p, type='correlation', vectorization=True, fisher_t=False)

    return corr_mats, idx
#
#
# for fn in sorted(glob('/DataCommon/jwlee/AAL_116_EMOTION/LR/*')):
#
#     if fn == '/DataCommon/jwlee/AAL_116_EMOTION/LR/AAL_116_136126.mat':
#         print('except 131126')
#         continue
#
#     if fn == '/DataCommon/jwlee/AAL_116_EMOTION/LR/AAL_116_100206.mat':
#         sub = sio.loadmat(fn)
#         arr = sub['ROI']
#         print(arr.shape) # (176, 116)
#
#         corr_matx, idx = corr(arr, 4) # (44, 116, 116)
#         corr_matx = np.expand_dims(corr_matx, 0)
#
#     else:
#         sub = sio.loadmat(fn)
#         arr = sub['ROI']
#         print(fn)  # (176, 116)
#
#         corr_matx_T, idx_T = corr(arr, 4)  # (44, 116, 116)
#         corr_matx_T = np.expand_dims(corr_matx_T, 0)
#
#         corr_matx = np.concatenate((corr_matx, corr_matx_T), axis=0)
#
#
#
# print(corr_matx.shape)
# np.savez_compressed('/DataCommon/jwlee/EMOTION_LR/cluster_3_hcp_emotion_FC', tfMRI_EMOTION_LR=corr_matx)





data = np.load('/DataCommon/jwlee/EMOTION_LR/cluster_2_hcp_emotion.npz')
sample = data['tfMRI_EMOTION_LR'] # (1041, 150, 116)
sample = sample[:,:,48:56] # (1041, 150, 8)

for i in range(sample.shape[0]):
    if i == 0:
        corr_matx, idx = corr(sample[i], 25)
        corr_matx_F = np.expand_dims(corr_matx, 0)
        print(corr_matx_F.shape) # (1, 6, 6670)
    else:
        corr_matx, idx = corr(sample[i], 25)
        corr_matx_P = np.expand_dims(corr_matx, 0)
        corr_matx_F = np.concatenate((corr_matx_F, corr_matx_P), axis=0)

print(corr_matx_F.shape)
np.savez_compressed('/DataCommon/jwlee/EMOTION_LR/cluster_2_hcp_emotion_FC_8areas', tfMRI_EMOTION_LR=corr_matx_F)