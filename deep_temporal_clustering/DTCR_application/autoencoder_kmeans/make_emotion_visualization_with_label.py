import torch
import datetime
import torch.nn as nn
import os
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import utils
import random
import glob
import matplotlib.pyplot as plt
import warnings
from sklearn.manifold import TSNE
import argparse
from tslearn.clustering import TimeSeriesKMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import normalized_mutual_info_score, rand_score


def writelog(file, line):
    file.write(line + '\n')
    print(line)

def get_arguments():
    parser = argparse.ArgumentParser()

    # training args
    parser.add_argument("--gpu_id", type=str, default="1", help="GPU id")
    parser.add_argument("--dir_root", default='/home/jwlee/HMM/deep_temporal_clustering/DTCR_application/autoencoder_kmeans',)



    return parser





if __name__ == "__main__":

    parser = get_arguments()
    args = parser.parse_args()


    # GPU Configuration
    gpu_id = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print(args.device)


    # data load
    data = np.load('/DataCommon/jwlee/EMOTION_LR/cluster_2_hcp_emotion.npz')
    samples = data['tfMRI_EMOTION_LR']
    # samples = samples[:, :, 48:56]  # (1041, 150, 116)



    # minmax
    mm = MinMaxScaler()
    results = []
    for ss in range(1041):
        results.append(mm.fit_transform(samples[ss]))
    sample = np.array(results)

    # train, validation, test
    data_label =  data = np.load('/DataCommon/jwlee/EMOTION_LR/cluster_6_hcp_emotion_label.npz')
    label = data['label_list_LR']
    label = label[:1041]  #(1041, 150)

    # number of clusters
    args.n_clusters = len(np.unique(label))

    if not os.path.exists(os.path.join(args.dir_root, 'visualization')):
        os.makedirs(os.path.join(args.dir_root, 'visualization'))

    sub0 = sample[0]
    sub0 = sub0.T
    sub0_label = label[0]
    sub0_label = np.expand_dims(sub0_label, 0)
    sub0_label = sub0_label * 0.1
    rep = np.repeat(sub0_label, 25, axis=0)
    total_0 = np.concatenate((sub0, rep), axis=0)
    plt.imshow(total_0)
    plt.xlabel('Time')
    plt.ylabel('ROI')
    plt.title('Subject 0')
    plt.savefig(os.path.join(args.dir_root, 'visualization/') + 'sub_0.png')
    plt.close()



    sub100 = sample[100]
    sub100 = sub100.T
    sub100_label = label[100]
    sub100_label = np.expand_dims(sub100_label, 0)
    sub100_label = sub100_label * 0.1
    rep = np.repeat(sub100_label, 25, axis=0)
    total_100 = np.concatenate((sub100, rep), axis=0)
    plt.imshow(total_100)
    plt.xlabel('Time')
    plt.ylabel('ROI')
    plt.title('Subject 100')
    plt.savefig(os.path.join(args.dir_root, 'visualization/') + 'sub_100.png')
    plt.close()



    sub500 = sample[500]
    sub500 = sub500.T
    sub500_label = label[500]
    sub500_label = np.expand_dims(sub500_label, 0)
    sub500_label = sub500_label * 0.1
    rep = np.repeat(sub500_label, 25, axis=0)
    total_500 = np.concatenate((sub500, rep), axis=0)
    plt.imshow(total_500)
    plt.xlabel('Time')
    plt.ylabel('ROI')
    plt.title('Subject 500')
    plt.savefig(os.path.join(args.dir_root, 'visualization/') + 'sub_500.png')
    plt.close()





