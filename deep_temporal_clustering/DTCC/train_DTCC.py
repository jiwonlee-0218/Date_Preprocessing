import argparse
import copy
import os
import random
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from load_data import *
from torch.utils.data import TensorDataset, DataLoader






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_name", default="UCR_UEA_Kmeans", help="model name")
    parser.add_argument('--dataset', default='SyntheticControl', help='UCR/UEA univariate or multivariate dataset')
    parser.add_argument("--gpu_id", type=str, default="1", help="GPU id")
    parser.add_argument("--visualization", default=True, help='visualization True or None')
    parser.add_argument('--input_dim', default=1, type=int, help="input dimension")

    # directory
    parser.add_argument("--dir_root", default='/home/jwlee/HMM/deep_temporal_clustering/task_clustering/')
    parser.add_argument("--dataset_name", default="UCR_SyntheticControl", help="dataset name")

    args = parser.parse_args()

    # GPU Configuration
    gpu_id = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print(args.device)

    ################ Preparing UCR Dataset
    train_data, train_label = load_data('Coffee')       #(56, 286, 1)=(S, T, D)  #(56,)=(0 or 1)


    args.n_clusters = len(np.unique(train_label))       #2
    args.num_steps = train_data.shape[1]
    args.embedding_size = train_data.shape[2]


    train_real_fake_data, train_real_fake_label = construct_classification_dataset(train_data)


    cls_data = np.expand_dims(train_real_fake_data, axis=2)
    cls_label_ = np.zeros(shape=(train_real_fake_label.shape[0], len(np.unique(train_real_fake_label))))
    cls_label_[np.arange(cls_label_.shape[0]), train_real_fake_label] = 1

    train_real_fake_data, train_real_fake_labels = torch.FloatTensor(train_real_fake_data), torch.FloatTensor(cls_label_)
    train_real_fake_data = train_real_fake_data.to(args.device)
    train_real_fake_labels = train_real_fake_labels.to(args.device)

    train_ds = TensorDataset(train_real_fake_data, train_real_fake_labels)
    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)  ###