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

from torch.utils.tensorboard import SummaryWriter
import mab_model
from sklearn.svm import SVC
from sklearn import metrics


def writelog(file, line):
    file.write(line + '\n')
    print(line)


def get_args_parser():
    parser = argparse.ArgumentParser('SiTv2', add_help=False)

    # Model parameters
    parser.add_argument("--dataset_name", default="RELATIONAL_6", help="dataset name")
    parser.add_argument('--model', default='vit_tiny', type=str, help="Name of architecture to train.")
    parser.add_argument("--model_name", default="FC_classification", help="model name")
    parser.add_argument("--methods", default="SVM",  type=str)


    parser.add_argument('--img_size', default=284, type=int, help="Input size to the Transformer.")
    parser.add_argument('--patch_size', default=1, type=int, help="patch size to the Transformer.")
    parser.add_argument("--similarity", required=False, choices=["COR", "EUC", "CID", "COS"], default="COS", help="The similarity type")



    # Hyper-parameters
    parser.add_argument('--batch_size', default=16, type=int, help="Batch size per GPU.")
    parser.add_argument('--epochs', default=50, type=int, help="Number of epochs of training.")
    parser.add_argument("--max_epochs", type=int, default=250, help="Maximum epochs numer of the full model training", )

    parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight decay")
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate.")



    # GPU
    parser.add_argument("--gpu_id", type=str, default="2", help="GPU id")


    # directory
    parser.add_argument("--dir_root", default='/home/jwlee/HMM/deep_temporal_clustering/task_clustering/')


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
    emotion_data = np.load('/DataCommon/jwlee/EMOTION_LR/cluster_2_hcp_emotion_FC.npz')
    emotion_samples = emotion_data['tfMRI_EMOTION_LR'] # (1041, 6, 6670)

    label = [0, 0, 0, 0, 0, 0]
    label = np.array(label)
    label = np.expand_dims(label, 0)
    labels = np.repeat(label, 1041, 0) # (1041, 6)


    ''' binary classification one-hot encoding '''
    # num = np.unique(labels, axis=0)
    # num = 2 # class 개수
    # emotion_labels = np.eye(num)[labels] # (1041, 6, 2)



    ''' RELATIONAL '''
    ################ Preparing Dataset
    relational_data = np.load('/DataCommon/jwlee/RELATIONAL_LR/cluster_2_hcp_relational_FC.npz')
    relational_samples = relational_data['tfMRI_RELATIONAL_LR']  # (1040, 6, 6670)

    R_label = [1, 1, 1, 1, 1, 1]
    R_label = np.array(R_label)
    R_label = np.expand_dims(R_label, 0)
    R_labels = np.repeat(R_label, 1040, 0) # (1040, 6)



    sum_samples = np.concatenate((emotion_samples, relational_samples), 0) #(2081, 6, 6670)
    sum_labels = np.concatenate((labels, R_labels), 0) # (2081, 6)


    X_train, X_test, y_train, y_test = train_test_split(sum_samples, sum_labels, random_state=42, shuffle=True, test_size=0.2)


    X_train = X_train.reshape(-1, 6670)    # (9984, 6670)
    y_train = y_train.reshape(-1)       # (9984,)
    X_test = X_test.reshape(-1, 6670)      # (2502, 6670)
    y_test = y_test.reshape(-1)


    model = SVC(kernel='rbf')
    model.fit(X_train, y_train)

    train_pmc = model.predict(X_train)
    test_pmc = model.predict(X_test)

    train_Acc = metrics.accuracy_score(train_pmc, y_train)
    test_Acc = metrics.accuracy_score(test_pmc, y_test)
    print('train_Accuracy: ', train_Acc)
    print('test_Accuracy: ', test_Acc)


    f = open(os.path.join(directory, 'SVM Classification\'s test Acc.log'), 'a')
    writelog(f, '======================')
    writelog(f, 'Model Name: %s' % args.methods)
    writelog(f, 'Train Accuracy: %.4f' % train_Acc)
    writelog(f, 'Test Accuracy: %.4f' % test_Acc)
    f.close()















