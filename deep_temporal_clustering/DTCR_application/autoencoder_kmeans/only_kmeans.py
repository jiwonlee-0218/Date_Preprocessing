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

    # data args
    parser.add_argument("--dataset_name", default="EMOTION_6cluster", help="dataset name")
    parser.add_argument("--similarity", required=False, choices=["COR", "EUC", "CID"], default="EUC", help="The similarity type")

    # model args
    parser.add_argument("--model_name", default="DTCR_bidirection_dilated_RNN2",help="model name")

    # training args
    parser.add_argument("--gpu_id", type=str, default="1", help="GPU id")

    # parser.add_argument('--clip_grad', type=float, default=5.0, help="Gradient clipping: Maximal parameter gradient norm.")
    parser.add_argument("--batch_size", default=16, type=int, help="batch size")
    parser.add_argument("--epochs_ae", type=int, default=300, help="Epochs number of the autoencoder training",)
    parser.add_argument("--max_patience", type=int, default=15, help="The maximum patience for pre-training, above which we stop training.",)

    parser.add_argument("--lr_ae", type=float, default=0.01, help="Learning rate of the autoencoder training",)
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay for Adam optimizer",)
    parser.add_argument("--dir_root", default='/home/jwlee/HMM/deep_temporal_clustering/DTCR_application/autoencoder_kmeans',)
    parser.add_argument("--ae_weights", default='models_weights/', help='models_weights/')
    # parser.add_argument("--ae_models", default='full_models/', help='full autoencoder weights')
    # parser.add_argument("--ae_weights", default=None, help='pre-trained autoencoder weights')
    parser.add_argument("--ae_models", default=None, help='full autoencoder weights')
    parser.add_argument("--autoencoder_test", default=None, help='full autoencoder weights')


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


    directory = os.path.join(args.dir_root, args.model_name, args.dataset_name, 'Epochs' + str(args.epochs_ae) + '_BS_' + str(args.batch_size) + '_LR_' + str(args.lr_ae) + '_wdcay_' + str(args.weight_decay))

    total_NMI = 0

    f = open(os.path.join(directory, 'KMeans\'s ARI_no_embedding.log'), 'a')

    for i in range(sample.shape[0]):
        kmc = KMeans(n_clusters=args.n_clusters).fit(sample[i])
        pmc = kmc.predict(sample[i])
        NMI = normalized_mutual_info_score(pmc, label[i])
        writelog(f, 'NMI is: %.4f' % (np.round(NMI, 3)))

        total_NMI += np.round(NMI, 3)

    avg = (total_NMI / sample.shape[0])
    writelog(f, '======================')
    writelog(f, 'subject: %d' % (sample.shape[0]))
    writelog(f, 'total_NMI is: %.4f' % (total_NMI))
    writelog(f, 'avg_NMI is: %.4f' % (avg))
    writelog(f, 'number of clusters: %d' % (args.n_clusters))
    f.close()
    print(avg)

