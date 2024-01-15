import torch
import datetime
import torch.nn as nn
import os
import numpy as np

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings
from sklearn.manifold import TSNE



if __name__ == "__main__":



    # GPU Configuration
    gpu_id = '2'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    # data load
    data = np.load('/DataCommon/jwlee/EMOTION_LR/cluster_2_hcp_emotion.npz')
    samples = data['tfMRI_EMOTION_LR'] #(1041, 150, 116)

    # minmax
    mm = MinMaxScaler()
    results = []
    for ss in range(1041):
        results.append(mm.fit_transform(samples[ss]))
    sample = np.array(results)


    # train, validation, test
    data_label6 = np.load('/DataCommon/jwlee/EMOTION_LR/cluster_6_hcp_emotion_label.npz')
    label_6 = data_label6['label_list_LR']  #(1041, 150)


    # data_label2 = np.load('/DataCommon/jwlee/EMOTION_LR/cluster_2_hcp_emotion_label.npz')
    # label_2 = data_label2['label_list_LR']  # (1041, 150)
    # label_2 = label_2[:1041]


    tsne = TSNE(n_components=2, random_state=42)
    tsne_sample = tsne.fit_transform(sample[200])

    labeled_0 = tsne_sample[label_6[200] == 1]
    labeled_1 = tsne_sample[label_6[200] == 2]
    labeled_2 = tsne_sample[label_6[200] == 3]
    labeled_3 = tsne_sample[label_6[200] == 4]
    labeled_4 = tsne_sample[label_6[200] == 5]
    labeled_5 = tsne_sample[label_6[200] == 6]
    plt.scatter(labeled_0[:, 0], labeled_0[:, 1], label='shape_1', color='red', alpha=0.5)
    plt.scatter(labeled_1[:, 0], labeled_1[:, 1], label='face_2', color='orange', alpha=0.5)
    plt.scatter(labeled_2[:, 0], labeled_2[:, 1], label='shape_3', color='yellow', alpha=0.5)
    plt.scatter(labeled_3[:, 0], labeled_3[:, 1], label='face_4', color='green', alpha=0.5)
    plt.scatter(labeled_4[:, 0], labeled_4[:, 1], label='shape_5', color='blue', alpha=0.5)
    plt.scatter(labeled_5[:, 0], labeled_5[:, 1], label='face_6', color='purple', alpha=0.5)
    plt.legend()
    plt.show()
    plt.close()










