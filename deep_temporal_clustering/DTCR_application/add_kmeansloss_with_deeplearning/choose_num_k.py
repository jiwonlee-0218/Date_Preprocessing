import torch
import datetime
import torch.nn as nn
import os
import numpy as np
from deep_main_config import get_arguments
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import deep_utils
from deep_KM_model import Seq2Seq, TCNModel, TCM_DRNN
import random
import glob
import matplotlib.pyplot as plt
import warnings
from sklearn.manifold import TSNE









if __name__ == "__main__":

    parser = get_arguments()
    args = parser.parse_args()


    # GPU Configuration
    gpu_id = '2'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print(args.device)


    # data load
    data = np.load('/DataCommon/jwlee/RESTING_LR/RESTING_LR.npz')
    samples = data['HCP_RESTING_LR'] #(1075, 1200, 116)

    # minmax
    mm = MinMaxScaler()
    results = []
    for ss in range(1075):
        results.append(mm.fit_transform(samples[ss]))
    sample = np.array(results)


    real_train_data, real_test_data = train_test_split(samples, random_state=42, test_size=0.2)

    real_train_data = real_train_data[:100]
    inputs = real_train_data.reshape(-1, 116)

    '''
    clusters_range = range(2, 14)
    results = []

    for i in clusters_range:
        clusterer = KMeans(n_clusters=i, init='random', n_init=10, max_iter=300, random_state=0)
        cluster_labels = clusterer.fit_predict(inputs)
        silhouette_avg = silhouette_score(inputs, cluster_labels)
        results.append([i, silhouette_avg])

    result = pd.DataFrame(results, columns=["n_clusters", "silhouette_score"])
    pivot_km = pd.pivot_table(result, index="n_clusters", values="silhouette_score")

    plt.figure()
    sns.heatmap(pivot_km, annot=True, linewidths=.5, fmt='.3f', cmap=sns.cm._rocket_lut)
    plt.tight_layout()
    plt.show()
    plt.close()
    '''

    distortions = []
    for i in range(1, 30):
        km = KMeans(n_clusters=i)
        km.fit(inputs)
        distortions.append(km.inertia_)

    plt.plot(range(1, 30), distortions, marker='o')
    plt.xlabel('Number of Cluster')
    plt.ylabel('Distortion')
    plt.show()
    plt.close()















