import numpy as np
import scipy.io as sio
from os.path import join
from nilearn.connectome import ConnectivityMeasure
import matplotlib.pyplot as plt
from glob import glob
import nibabel as nib


def connectivity(input, type='correlation', vectorization=True, fisher_t=False):
    '''kind{“covariance”, “correlation”, “partial correlation”, “tangent”, “precision”}, optional'''
    measure = ConnectivityMeasure(kind=type, vectorize=vectorization, discard_diagonal=True)
    mt = measure.fit_transform(input)

    if fisher_t == True:
        for i in range(len(mt)):
            mt[i][:] = np.arctanh(mt[i][:])
    return mt

# file_path = 'rfMRI_REST1_LR.nii.gz/practice'
#
# for i, n in enumerate(glob(join(file_path, 'rfMRI_REST1_LR.nii.gz.mat'))):
#     filename = n.split('/')
#     sub_name = filename[-1].split('.')
#
#     sub = sio.loadmat(n)
#     sub = sub['ROI']
#
#     sub = np.reshape(sub, ((1,) + sub.shape))
#
#     fc = connectivity(sub, type='correlation', vectorization=False, fisher_t=False)
#
#     np.save(join(file_path, 'FC_correlation', sub_name[-2]), fc)
#
# print('done!!')








# filename = '/DataCommon/jwlee/AAL_116_EMOTION/LR/AAL_116_870861.mat'
# sub = sio.loadmat(filename)
# sub = sub['ROI']
# sub = np.reshape(sub, ((1,) + sub.shape))
#
# list_atlas = [34, 63, 92, 121, 151]
#
# arr = sub[:, 4:4+25, :]
#
# for i in list_atlas:
#     b = sub[:, i:i+25 ,:]
#     arr = np.concatenate((arr, b), axis=1)
#
# cluster_2_hcp_emotion_LR = arr   # (1080, 284, 10)
# cluster_2_hcp_emotion_LR = cluster_2_hcp_emotion_LR.reshape(1, 116, -1)
#
# fc = connectivity(cluster_2_hcp_emotion_LR, type='correlation', vectorization=False, fisher_t=False)
# plt.imshow(fc[0])
# plt.colorbar()
# plt.show()


# filename = '/mnt/hcp/homes/jwlee/HCP_extract/100206/EMOTION/LR/tfMRI_EMOTION_LR.nii.gz'
#
# proxy = nib.load(filename)
# arr = proxy.get_fdada()




#
# list = ['100206', '203923', '379657', '469961', '520228', '644246', '970764']
#
# for i in list:
#     filename = '/mnt/hcp/homes/jwlee/HCP_extract/'+i+'/EMOTION/LR/Destrieux_148_'+i+'.mat'
#     # filename = '/DataCommon/jwlee/AAL_116_EMOTION/LR/AAL_116_783462.mat'             #AAL_116_857263
#
#     sub = sio.loadmat(filename)
#     sub = sub['ROI']
#     sub = np.reshape(sub, ((1,) + sub.shape))
#
#     list_atlas = [34, 63, 92, 121, 151]
#
#     arr = sub[:, 4:4+25, :]
#
#     for i in list_atlas:
#         b = sub[:, i:i+25 ,:]
#         arr = np.concatenate((arr, b), axis=1)
#
#     cluster_2_hcp_emotion_LR = arr   # (1080, 284, 10)
#     cluster_2_hcp_emotion_LR = np.expand_dims(cluster_2_hcp_emotion_LR[0].T, 0)
#
#     fc = connectivity(cluster_2_hcp_emotion_LR, type='correlation', vectorization=False, fisher_t=False)
#     plt.imshow(fc[0])
#     plt.colorbar()
#     plt.show()
#     plt.close()
#


############## time x time correlation ################
# filename = '/mnt/hcp/HCP/102715/MNINonLinear/Results/tfMRI_EMOTION_LR/tfMRI_EMOTION_LR_Atlas.dtseries.nii'
# proxy = nib.load(filename)
# aa = proxy.get_fdata() #################################### get_fdata() ############################# 중요
# b= np.array(aa)
# cluster_2_hcp_emotion_LR = np.expand_dims(b.T, 0)
#
#
# fc = connectivity(cluster_2_hcp_emotion_LR, type='correlation', vectorization=False, fisher_t=False)
# plt.imshow(fc[0])
# plt.colorbar()
# plt.show()
# plt.close()






filename = '/mnt/hcp/homes/jwlee/HCP_extract/683256/EMOTION/LR/AAL_116_683256.mat'

sub = sio.loadmat(filename)
sub = sub['ROI']


# sub = np.reshape(sub, ((1,) + sub.shape))
#
# fc = connectivity(sub, type='correlation', vectorization=False, fisher_t=False)
#
# plt.imshow(fc[0])
# plt.colorbar()
# plt.show()
# plt.close()




#########################################################################################################################

import os
import numpy as np
import multiprocessing as proc
from scipy import stats
from scipy import linalg



def get_lambda(windowed_func):
    '''
    get the subject-specific regularisation parameter, lambda,
    using leave-n-out cross validation
    '''

    nwindows = len(windowed_func)
    n_nodes = len(windowed_func[0])
    window_size = len(windowed_func[0, 0])

    leave_out = 5
    inds = np.arange(0, nwindows - leave_out, leave_out)

    # model = inverse_covariance.QuicGraphicalLasso(lam=0., max_iter=100)

    alpha_u = 1.
    alpha_l = 0.
    alpha_rounds = 5

    for s in range(alpha_rounds):

        alpha_step = 10 ** -(s + 1)
        alphas = np.arange(alpha_l, alpha_u, alpha_step)
        sum_ll = -np.inf * np.ones(len(alphas))

        for a in range(len(alphas)):
            # model.set_params(lam=alphas[a])

            log_likelihood = 0
            for i in range(len(inds)):
                Xtrain = np.concatenate([windowed_func[:inds[i]], windowed_func[inds[i] + leave_out:]], axis=0)
                Xtrain = np.reshape(Xtrain, [n_nodes, (nwindows - leave_out) * window_size])

                # model.fit(Xtrain.T)

                Xtest = windowed_func[inds[i]:inds[i] + leave_out]
                for t in range(leave_out):
                    log_likelihood += model.score(Xtest[t].T)

            sum_ll[a] = log_likelihood

            if a != 0 and sum_ll[a] <= sum_ll[a - 1]:
                break

        best_alpha_ind = np.argmax(sum_ll)
        alpha_l = alphas[max([best_alpha_ind - 1, 0])]
        alpha_u = alphas[min([best_alpha_ind + 1, len(sum_ll) - 1])]

    best_alpha = alphas[best_alpha_ind]

    return best_alpha


def get_fc(func, lambda_):
    '''
    Estimate functional connectivity from each BOLD timeseries window, using the
    subject-specific regularisation parameter, lambda.
    '''

    # model = inverse_covariance.QuicGraphicalLasso(lam=lambda_, max_iter=100)
    # model.fit(func.T)

    cov = np.array(model.covariance_)
    D = np.sqrt(np.diag(np.diag(cov)))
    DInv = np.linalg.inv(D);
    fc = np.matmul(DInv, np.matmul(cov, DInv))
    np.fill_diagonal(fc, 0)

    return np.arctanh(fc)


def get_dfc(in_):
    '''
    run SWC for a single subject. Window the BOLD timeseries data then get lambda
    and estimate the FC matrix for each window.
    '''

    func = in_[0]
    window_size = in_[1]
    window_shape = in_[2]
    n_nodes = in_[3]
    step = in_[4]

    n_nodes = len(func[:, 0])
    if window_shape == 'rectangle':
        window = np.ones(window_size)
    elif window_shape == 'hamming':
        window = np.hamming(window_size)
    elif window_shape == 'hanning':
        window = np.hanning(window_size)
    else:
        raise Exception('%s window shape not recognised. Choose rectangle, hamming or hanning.' % window_shape)

    inds = range(0, len(func[0]) - window_size, step)
    nwindows = len(inds)
    dfc = np.zeros([nwindows, n_nodes, n_nodes])
    windowed_func = np.zeros([nwindows, n_nodes, window_size])

    for i in range(nwindows):
        this_sec = func[:, inds[i]:inds[i] + window_size]
        windowed_func[i] = this_sec * window

    lambda_ = get_lambda(windowed_func)

    for i in range(nwindows):
        dfc[i, :, :] = get_fc(windowed_func[i], lambda_)

    return dfc


def load_data(func_path, zscore=True, hcp=False):
    '''
    Load raw timeseries data.
    SimTB data is in the form of one .csv file per subject, with one node per row
    and one timepoint per column (i.e. each separated by a comma). HCP data is in
    the form of one .txt file per subject, with one node per column (each separated
    by a space) and one timepoint per row. For other data, use the same file
    structure as SimTB and run without the -hcp flag, or use the same file structure
    as HCP and run with the -hcp flag.
    '''

    files = os.listdir(func_path)

    if hcp:
        files = sorted([file for file in files if file.endswith('.txt')])
        subjs = np.array([stats.zscore(np.loadtxt('%s/%s' % (func_path, file)).T, axis=1) for file in files])

    else:
        files = sorted([file for file in files if file.endswith('.csv')])
        if zscore:
            subjs = np.array(
                [stats.zscore(np.loadtxt('%s/%s' % (func_path, file), delimiter=','), axis=1) for file in files])
        else:
            # don't z-score when loading ground-truth state time courses for SimTB data
            subjs = np.array([np.loadtxt('%s/%s' % (func_path, file), delimiter=',') for file in files])

    return subjs




get_dfc((sub, 20, 'rectangle', 116, 20))