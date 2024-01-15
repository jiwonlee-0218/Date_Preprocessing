import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, rand_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import argparse
import torch.nn as nn


import os
from pathlib import Path

import torch
import glob
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':

    # 7 label data load
    data = np.load('/media/12T/practice/LR_DICT/hcp_7tasks.npz')
    emotion_samples = data['tfMRI_EMOTION_LR']  # (1041, 176, 116)
    gambling_samples = data['tfMRI_GAMBLING_LR']  # (1085, 253, 116)
    language_samples = data['tfMRI_LANGUAGE_LR']  # (1044, 316, 116)
    motor_samples = data['tfMRI_MOTOR_LR']  # (1080, 284, 116)
    relational_samples = data['tfMRI_RELATIONAL_LR']  # (1040, 232, 116)
    social_samples = data['tfMRI_SOCIAL_LR']  # (1045, 274, 116)
    wm_samples = data['tfMRI_WM_LR']  # (1082, 405, 116)


    # mean 0, std 1
    emo_results = []
    for ss in range(1041):
        w = emotion_samples[ss]
        emo_results.append(StandardScaler().fit_transform(w))
    sample1 = np.array(emo_results)
    a1 = sample1.reshape(-1, 116)


    gambling_results = []
    for pp in range(1085):
        k = gambling_samples[pp]
        gambling_results.append(StandardScaler().fit_transform(k))
    sample2 = np.array(gambling_results)
    a2 = sample2.reshape(-1, 116)


    language_results = []
    for pp in range(1044):
        k = language_samples[pp]
        language_results.append(StandardScaler().fit_transform(k))
    sample3 = np.array(language_results)
    a3 = sample3.reshape(-1, 116)


    motor_results = []
    for pp in range(1080):
        k = motor_samples[pp]
        motor_results.append(StandardScaler().fit_transform(k))
    sample4 = np.array(motor_results)
    a4 = sample4.reshape(-1, 116)



    relational_results = []
    for pp in range(1040):
        k = relational_samples[pp]
        relational_results.append(StandardScaler().fit_transform(k))
    sample5 = np.array(relational_results)
    a5 = sample5.reshape(-1, 116)


    social_results = []
    for pp in range(1045):
        k = social_samples[pp]
        social_results.append(StandardScaler().fit_transform(k))
    sample6 = np.array(social_results)
    a6 = sample6.reshape(-1, 116)



    wm_results = []
    for pp in range(1082):
        k = wm_samples[pp]
        wm_results.append(StandardScaler().fit_transform(k))
    sample7 = np.array(wm_results)
    a7 = sample7.reshape(-1, 116)




    pca = PCA(n_components=2)  # 주성분을 몇개로 할지 결정
    printcipalComponents = pca.fit_transform(a1)