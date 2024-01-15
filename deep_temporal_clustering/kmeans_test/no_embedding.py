import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score, rand_score
import argparse
import torch.nn as nn
from functools import partial
from sklearn.manifold import TSNE
import os


from pathlib import Path

import torch
import glob
import matplotlib.pyplot as plt
import torch.nn.functional as F


from train import *
import ViT as vits
import FC_no_embedding


def cluster_using_kmeans(embeddings, K):

    kmc = KMeans(n_clusters=K).fit(embeddings)
    pmc = kmc.predict(embeddings)
    return pmc



def ri_score(y_true, y_pred):
    return rand_score(y_true, y_pred)

def nmi_score(y_true, y_pred):
    return normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')


def get_args_parser():
    parser = argparse.ArgumentParser('SiTv2', add_help=False)

    # Model parameters
    parser.add_argument("--dataset_name", default="MOTOR", help="dataset name")
    parser.add_argument('--model', default='vit_tiny', type=str, help="Name of architecture to train.")
    parser.add_argument("--model_name", default="R_T_Kmeans", help="model name")


    parser.add_argument('--img_size', default=284, type=int, help="Input size to the Transformer.")
    parser.add_argument('--patch_size', default=1, type=int, help="patch size to the Transformer.")
    parser.add_argument("--similarity", required=False, choices=["COR", "EUC", "CID", "COS"], default="COS", help="The similarity type")



    # Hyper-parameters
    parser.add_argument('--batch_size', default=16, type=int, help="Batch size per GPU.")
    parser.add_argument('--epochs', default=100, type=int, help="Number of epochs of training.")
    parser.add_argument("--max_epochs", type=int, default=250, help="Maximum epochs numer of the full model training", )

    parser.add_argument('--weight_decay', type=float, default=1e-6, help="weight decay")
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate.")



    # GPU
    parser.add_argument("--gpu_id", type=str, default="2", help="GPU id")


    # directory
    parser.add_argument("--dir_root", default='/home/jwlee/HMM/deep_temporal_clustering/kmeans_test/')


    # parser.add_argument("--weights", default=None, help='pre-trained autoencoder weights')
    parser.add_argument("--weights", default='models_weights/', help='pre-trained autoencoder weights')
    parser.add_argument("--SiT_models", default=None, help='full autoencoder weights')
    # parser.add_argument("--SiT_models", default='full_models/', help='full autoencoder weights')


    return parser









def writelog(file, line):
    file.write(line + '\n')
    print(line)









if __name__ == '__main__':
    parser = argparse.ArgumentParser('SiTv2', parents=[get_args_parser()])
    args = parser.parse_args()


    # GPU Configuration
    gpu_id = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print(args.device)

    # directory = os.path.join(args.dir_root, args.model_name, args.dataset_name,
    #                          'Epochs' + str(args.epochs) + '_BS_' + str(args.batch_size) + '_LR_' + str(
    #                              args.lr) + '_wdcay_' + str(args.weight_decay))


    ''' emotion '''
    # data load
    data = np.load('/DataCommon/jwlee/EMOTION_LR/cluster_2_hcp_emotion.npz')
    samples = data['tfMRI_EMOTION_LR']  # (1041, 150, 116)

    # minmax
    mm = MinMaxScaler()
    results = []
    for ss in range(1040):
        results.append(mm.fit_transform(samples[ss]))
    sample = np.array(results)

    # train, validation, test
    data_label6 = np.load('/DataCommon/jwlee/EMOTION_LR/cluster_6_hcp_emotion_label.npz')
    label_6 = data_label6['label_list_LR']  # (1041, 150)
    label_6 = label_6[:1040]

    data_label2 = np.load('/DataCommon/jwlee/EMOTION_LR/cluster_2_hcp_emotion_label.npz')
    label_2 = data_label2['label_list_LR']  # (1041, 150)
    label_2 = label_2[:1040]

    # number of clusters
    args.n_clusters = len(np.unique(label_2))
    args.timeseries = sample.shape[1]

    index = [10, 100, 200, 300, 400, 500, 600, 700, 800]


    for i in index:
        tsne = TSNE(n_components=2)
        kkk = tsne.fit_transform(sample[i])
        labeled_0 = kkk[label_2[0] == 1]
        labeled_1 = kkk[label_2[0] == 2]
        plt.scatter(labeled_0[:, 0], labeled_0[:, 1], label='shape_1', color='red', alpha=0.5)
        plt.scatter(labeled_1[:, 0], labeled_1[:, 1], label='face_2', color='blue', alpha=0.5)
        plt.legend()
        plt.savefig('/home/jwlee/train/sub{}_label2.png'.format((i)))
        plt.close()

        labeled_0 = kkk[label_6[0] == 1]
        labeled_1 = kkk[label_6[0] == 2]
        labeled_2 = kkk[label_6[0] == 3]
        labeled_3 = kkk[label_6[0] == 4]
        labeled_4 = kkk[label_6[0] == 5]
        labeled_5 = kkk[label_6[0] == 6]
        plt.scatter(labeled_0[:, 0], labeled_0[:, 1], label='shape_1', color='red', alpha=0.5)
        plt.scatter(labeled_1[:, 0], labeled_1[:, 1], label='face_2', color='blue', alpha=0.5)
        plt.scatter(labeled_2[:, 0], labeled_2[:, 1], label='shape_3', color='firebrick', alpha=0.5)
        plt.scatter(labeled_3[:, 0], labeled_3[:, 1], label='face_4', color='midnightblue', alpha=0.5)
        plt.scatter(labeled_4[:, 0], labeled_4[:, 1], label='shape_5', color='crimson', alpha=0.5)
        plt.scatter(labeled_5[:, 0], labeled_5[:, 1], label='face_6', color='royalblue', alpha=0.5)
        plt.legend()
        plt.savefig('/home/jwlee/train/sub{}_label6.png'.format(i))
        plt.close()






