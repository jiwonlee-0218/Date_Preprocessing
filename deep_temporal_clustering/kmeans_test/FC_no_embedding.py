import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score

import argparse
import torch.nn as nn
from functools import partial

import os


from pathlib import Path

import torch
import glob
import matplotlib.pyplot as plt
import torch.nn.functional as F


from train import *
import ViT as vits






def get_args_parser():
    parser = argparse.ArgumentParser('SiTv2', add_help=False)

    # Model parameters
    parser.add_argument("--dataset_name", default="EMOTION_6", help="dataset name")
    parser.add_argument('--model', default='vit_tiny', type=str, help="Name of architecture to train.")
    parser.add_argument("--model_name", default="FC_Kmeans", help="model name")


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


    parser.add_argument("--weights", default=None, help='pre-trained autoencoder weights')
    # parser.add_argument("--weights", default='models_weights/', help='pre-trained autoencoder weights')
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

    directory = os.path.join(args.dir_root, args.model_name, args.dataset_name,
                             'Epochs' + str(args.epochs) + '_BS_' + str(args.batch_size) + '_LR_' + str(
                                 args.lr) + '_wdcay_' + str(args.weight_decay))
    if not os.path.exists(directory):
        os.makedirs(directory)


    ''' EMOTION  '''
    ################ Preparing Dataset
    data = np.load('/DataCommon/jwlee/EMOTION_LR/cluster_2_hcp_emotion_FC_8areas.npz')
    samples = data['tfMRI_EMOTION_LR'] # (1041, 6, 6670)

    label = [0,1,0,1,0,1]
    label = np.array(label)
    label = np.expand_dims(label, 0)
    labels = np.repeat(label, 1041, 0)


    sample = samples.reshape(-1, 28) # -> (6246, 6670)



    # total_NMI = 0
    # f = open(os.path.join(directory, 'KMeans\'s NMI_no_embedding.log'), 'a')
    #
    #
    #
    # kmc = KMeans(n_clusters=2, init="k-means++").fit(sample)
    #
    #
    # for i in range(1041):
    #     p_kmc = kmc.predict(samples[i])
    #     NMI = normalized_mutual_info_score(p_kmc, labels[i])
    #     writelog(f, 'NMI is: %.4f' % (np.round(NMI,3)))
    #
    #     total_NMI += np.round(NMI,3)
    #
    #
    # avg = (total_NMI / samples.shape[0] )
    # writelog(f, '======================')
    # writelog(f, 'subject: %d' % ( samples.shape[0] ))
    # writelog(f, 'total_NMI is: %.4f' % (total_NMI))
    # writelog(f, 'avg_NMI is: %.4f' % (avg))
    # f.close()
    # print(avg)








    ''' EMOTION '''  # predict를 몇명씩 하느냐의 차이
    total_NMI = 0
    f = open(os.path.join(directory, 'KMeans\'s NMI_no_embedding_8areas.log'), 'a')

    kmc = KMeans(n_clusters=2, init="k-means++").fit(sample)
    p_kmc = kmc.predict(sample)
    # p_kmc = p_kmc.reshape(-1, 6) # (1041, 6)

    NMI = normalized_mutual_info_score(p_kmc, labels.flatten())
    print(NMI)
    print(np.sum(p_kmc == labels.flatten())/6246*100)
    print(np.sum(p_kmc == np.abs(labels.flatten()-1))/6246*100)

    # for i in range(1041):
    #     NMI = normalized_mutual_info_score(p_kmc[i], labels[i])
    #     print(np.sum(p_kmc[i] == labels[i])/6)
    #     # print(np.sum(p_kmc[i] == np.abs(labels[i]-1))/6)
    #     writelog(f, 'NMI is: %.4f' % (np.round(NMI, 3)))
    #
    #     total_NMI += np.round(NMI, 3)

    avg = (total_NMI / samples.shape[0])
    writelog(f, '======================')
    writelog(f, 'subject: %d' % (samples.shape[0]))
    writelog(f, 'total_NMI is: %.4f' % (total_NMI))
    writelog(f, 'avg_NMI is: %.4f' % (avg))
    f.close()
    print(avg)





    # ''' EMOTION '''  # individual
    # total_NMI = 0
    # f = open(os.path.join(directory, 'KMeans\'s NMI_no_embedding_8areas_individual.log'), 'a')
    #
    #
    # for i in range(1041):
    #     kmc = KMeans(n_clusters=2, init="k-means++").fit(samples[i])
    #     p_kmc = kmc.predict(samples[i])
    #     NMI = normalized_mutual_info_score(p_kmc, labels[i])
    #     writelog(f, 'NMI is: %.4f' % (np.round(NMI, 3)))
    #
    #     total_NMI += np.round(NMI, 3)
    #
    # avg = (total_NMI / samples.shape[0])
    # writelog(f, '======================')
    # writelog(f, 'subject: %d' % (samples.shape[0]))
    # writelog(f, 'total_NMI is: %.4f' % (total_NMI))
    # writelog(f, 'avg_NMI is: %.4f' % (avg))
    # f.close()
    # print(avg)









