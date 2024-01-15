import numpy as np
import argparse
from tslearn.datasets import UCR_UEA_datasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import normalized_mutual_info_score, rand_score

import os
import torch
import matplotlib.pyplot as plt







ucr = UCR_UEA_datasets()
# UCR/UEA univariate and multivariate datasets.
all_ucr_datasets = ucr.list_datasets()

def writelog(file, line):
    file.write(line + '\n')
    print(line)


def load_ucr(dataset='CBF'):
    X_train, y_train, X_test, y_test = ucr.load_dataset(dataset)
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    if dataset == 'HandMovementDirection':  # this one has special labels
        y = [yy[0] for yy in y]
    y = LabelEncoder().fit_transform(y)  # sometimes labels are strings or start from 1
    assert(y.min() == 0)  # assert labels are integers and start from 0
    # preprocess data (standardization)
    X_scaled = TimeSeriesScalerMeanVariance().fit_transform(X)
    return X_scaled, y


def load_data(dataset_name):
    if dataset_name in all_ucr_datasets:
        return load_ucr(dataset_name)
    else:
        print('Dataset {} not available! Available datasets are UCR/UEA univariate and multivariate datasets.'.format(dataset_name))
        exit(0)



def cluster_using_kmeans(embeddings, K):
    return KMeans(n_clusters=K).fit(embeddings)

def cluster_using_gmm(embeddings, K):
    return GaussianMixture(n_components=K).fit(embeddings)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_name", default="UCR_UEA_Kmeans", help="model name")
    parser.add_argument('--dataset', default='SyntheticControl', help='UCR/UEA univariate or multivariate dataset')
    parser.add_argument("--gpu_id", type=str, default="2", help="GPU id")
    parser.add_argument("--visualization", default=True, help='visualization True or None')

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

    # directory
    directory = os.path.join(args.dir_root, args.model_name, args.dataset_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Load data
    (X_train, y_train) = load_data(args.dataset)
    X_train = X_train.squeeze()  #(884, 136)
    pca = PCA(2)
    X_train_pca = pca.fit_transform(X_train) #(884, 2)

    # log file
    k = open(os.path.join(directory, 'KMeans\'s PCA_train NMI.log'), 'a')
    g = open(os.path.join(directory, 'GMM\'s PCA_train RI.log'), 'a')
    writelog(k, '=======================================================================')
    writelog(k, 'Task-wise UCR -> KMeans clustering (PCA) -> (6670 dimension to 2 dimension)')
    writelog(g, '=======================================================================')
    writelog(g, 'Task-wise UCR -> GMM clustering (PCA) -> (6670 dimension to 2 dimension)')

    # cluetring
    kmc = cluster_using_kmeans(X_train_pca, 6)
    gmm = cluster_using_gmm(X_train_pca, 6)

    # kmeans
    L_kmc = kmc.predict(X_train_pca)
    k_NMI = normalized_mutual_info_score(L_kmc, y_train)
    k_RI = rand_score(L_kmc, y_train)
    writelog(k, '======================')
    writelog(k, 'number of data: %d' % (X_train_pca.shape[0]))
    writelog(k, 'train_NMI is: %.4f' % (k_NMI))
    writelog(k, 'train_RI is: %.4f' % (k_RI))
    k.close()

    # gmm
    L_gmm = gmm.predict(X_train_pca)
    g_NMI = normalized_mutual_info_score(L_gmm, y_train)
    g_RI = rand_score(L_gmm, y_train)
    writelog(g, '======================')
    writelog(g, 'number of data: %d' % (X_train_pca.shape[0]))
    writelog(g, 'train_NMI is: %.4f' % (g_NMI))
    writelog(g, 'train_RI is: %.4f' % (g_RI))
    g.close()



    ''' visualization '''
    if args.visualization is not None:
        if not os.path.exists(os.path.join(directory, 'train_clustering_visulization/')):
            os.makedirs(os.path.join(directory, 'train_clustering_visulization/'))


        # kmeans
        centroid = kmc.cluster_centers_

        ''' predict '''
        predicted_0 = X_train_pca[L_kmc == 0]
        predicted_1 = X_train_pca[L_kmc == 1]
        predicted_2 = X_train_pca[L_kmc == 2]
        predicted_3 = X_train_pca[L_kmc == 3]
        predicted_4 = X_train_pca[L_kmc == 4]
        predicted_5 = X_train_pca[L_kmc == 5]

        plt.scatter(predicted_0[:, 0], predicted_0[:, 1], label='predicted_0', color='aqua', alpha=0.5)
        plt.scatter(predicted_1[:, 0], predicted_1[:, 1], label='predicted_1', color='tan', alpha=0.5)
        plt.scatter(predicted_2[:, 0], predicted_2[:, 1], label='predicted_2', color='lightcoral', alpha=0.5)
        plt.scatter(predicted_3[:, 0], predicted_3[:, 1], label='predicted_3', color='plum', alpha=0.5)
        plt.scatter(predicted_4[:, 0], predicted_4[:, 1], label='predicted_4', color='royalblue', alpha=0.5)
        plt.scatter(predicted_5[:, 0], predicted_5[:, 1], label='predicted_5', color='slategray', alpha=0.5)
        plt.scatter(centroid[:, 0], centroid[:, 1], s=70, color='black')
        plt.legend()
        plt.savefig(os.path.join(directory, 'train_clustering_visulization/') + 'predicted_Kmeans.png')
        plt.close()

        # plt.scatter(predicted_0[:, 0], predicted_0[:, 1], color='green', alpha=0.5)
        # plt.savefig(os.path.join(directory, 'train_clustering_visulization/') + 'predicted_Kmeans_0.png')
        # plt.close()
        #
        # plt.scatter(predicted_1[:, 0], predicted_1[:, 1], color='olive', alpha=0.5)
        # plt.savefig(os.path.join(directory, 'train_clustering_visulization/') + 'predicted_Kmeans_1.png')
        # plt.close()


        ''' label '''
        labeled_0 = X_train_pca[y_train == 0]
        labeled_1 = X_train_pca[y_train == 1]
        labeled_2 = X_train_pca[y_train == 2]
        labeled_3 = X_train_pca[y_train == 3]
        labeled_4 = X_train_pca[y_train == 4]
        labeled_5 = X_train_pca[y_train == 5]

        plt.scatter(labeled_0[:, 0], labeled_0[:, 1], label='label_0', color='red', alpha=0.5)
        plt.scatter(labeled_1[:, 0], labeled_1[:, 1], label='label_1', color='orange', alpha=0.5)
        plt.scatter(labeled_2[:, 0], labeled_2[:, 1], label='label_2', color='gold', alpha=0.5)
        plt.scatter(labeled_3[:, 0], labeled_3[:, 1], label='label_3', color='green', alpha=0.5)
        plt.scatter(labeled_4[:, 0], labeled_4[:, 1], label='label_4', color='blue', alpha=0.5)
        plt.scatter(labeled_5[:, 0], labeled_5[:, 1], label='label_5', color='purple', alpha=0.5)
        plt.legend()
        plt.savefig(os.path.join(directory, 'train_clustering_visulization/') + 'label_Kmeans.png')
        plt.close()

        # plt.scatter(labeled_0[:, 0], labeled_0[:, 1], label='label_0', color='red', alpha=0.5)
        # plt.savefig(os.path.join(directory, 'train_clustering_visulization/') + 'label_0.png')
        # plt.close()
        #
        # plt.scatter(labeled_1[:, 0], labeled_1[:, 1], label='relational_label', color='blue', alpha=0.5)
        # plt.savefig(os.path.join(directory, 'train_clustering_visulization/') + 'label_1.png')
        # plt.close()

        ''' original '''
        plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], color='orange', alpha=0.5)
        plt.savefig(os.path.join(directory, 'train_clustering_visulization/') + 'orig_data.png')
        plt.close()


        # gmm
        if not os.path.exists(os.path.join(directory, 'train_clustering_visulization/GMM')):
            os.makedirs(os.path.join(directory, 'train_clustering_visulization/GMM'))

        ''' predict '''
        predicted_0 = X_train_pca[L_gmm == 0]
        predicted_1 = X_train_pca[L_gmm == 1]
        predicted_2 = X_train_pca[L_gmm == 2]
        predicted_3 = X_train_pca[L_gmm == 3]
        predicted_4 = X_train_pca[L_gmm == 4]
        predicted_5 = X_train_pca[L_gmm == 5]

        plt.scatter(predicted_0[:, 0], predicted_0[:, 1], label='predicted_0', color='aqua', alpha=0.5)
        plt.scatter(predicted_1[:, 0], predicted_1[:, 1], label='predicted_1', color='tan', alpha=0.5)
        plt.scatter(predicted_2[:, 0], predicted_2[:, 1], label='predicted_2', color='lightcoral', alpha=0.5)
        plt.scatter(predicted_3[:, 0], predicted_3[:, 1], label='predicted_3', color='plum', alpha=0.5)
        plt.scatter(predicted_4[:, 0], predicted_4[:, 1], label='predicted_4', color='royalblue', alpha=0.5)
        plt.scatter(predicted_5[:, 0], predicted_5[:, 1], label='predicted_5', color='slategray', alpha=0.5)
        plt.legend()
        plt.savefig(os.path.join(directory, 'train_clustering_visulization/GMM/') + 'predicted_GMM.png')
        plt.close()

        # plt.scatter(predicted_0[:, 0], predicted_0[:, 1], color='green', alpha=0.5)
        # plt.savefig(os.path.join(directory, 'train_clustering_visulization/GMM/') + 'predicted_GMM_0.png')
        # plt.close()
        #
        # plt.scatter(predicted_1[:, 0], predicted_1[:, 1], color='olive', alpha=0.5)
        # plt.savefig(os.path.join(directory, 'train_clustering_visulization/GMM/') + 'predicted_GMM_1.png')
        # plt.close()


        ''' label '''
        labeled_0 = X_train_pca[y_train == 0]
        labeled_1 = X_train_pca[y_train == 1]
        labeled_2 = X_train_pca[y_train == 2]
        labeled_3 = X_train_pca[y_train == 3]
        labeled_4 = X_train_pca[y_train == 4]
        labeled_5 = X_train_pca[y_train == 5]

        plt.scatter(labeled_0[:, 0], labeled_0[:, 1], label='label_0', color='red', alpha=0.5)
        plt.scatter(labeled_1[:, 0], labeled_1[:, 1], label='label_1', color='orange', alpha=0.5)
        plt.scatter(labeled_2[:, 0], labeled_2[:, 1], label='label_2', color='gold', alpha=0.5)
        plt.scatter(labeled_3[:, 0], labeled_3[:, 1], label='label_3', color='green', alpha=0.5)
        plt.scatter(labeled_4[:, 0], labeled_4[:, 1], label='label_4', color='blue', alpha=0.5)
        plt.scatter(labeled_5[:, 0], labeled_5[:, 1], label='label_5', color='purple', alpha=0.5)
        plt.legend()
        plt.savefig(os.path.join(directory, 'train_clustering_visulization/GMM/') + 'label_GMM.png')
        plt.close()

        # plt.scatter(labeled_0[:, 0], labeled_0[:, 1], label='label_0', color='red', alpha=0.5)
        # plt.savefig(os.path.join(directory, 'train_clustering_visulization/GMM/') + 'label_0.png')
        # plt.close()
        #
        # plt.scatter(labeled_1[:, 0], labeled_1[:, 1], label='relational_label', color='blue', alpha=0.5)
        # plt.savefig(os.path.join(directory, 'train_clustering_visulization/GMM/') + 'label_1.png')
        # plt.close()

        ''' original '''
        plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], color='orange', alpha=0.5)
        plt.savefig(os.path.join(directory, 'train_clustering_visulization/GMM/') + 'orig_data.png')
        plt.close()


        print('finish')