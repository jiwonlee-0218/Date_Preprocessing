import argparse
import copy
import os
import torch
import numpy as np
import load_ucr_data
import scikit_wrappers
from sklearn.preprocessing import MinMaxScaler

def fit_hyperparameters(args, train, train_labels, save_memory=False):
    """
    Creates a classifier from the given set of hyperparameters in the input
    file, fits it and return it.
    @param file Path of a file containing a set of hyperparemeters.
    @param train Training set.
    @param train_labels Labels for the training set.
    @param cuda If True, enables computations on the GPU.
    @param gpu GPU to use if CUDA is enabled.
    @param save_memory If True, save GPU memory by propagating gradients after
           each loss term, instead of doing it after computing the whole loss.
    """
    classifier = scikit_wrappers.CausalCNNEncoderClassifier(args)

    return classifier.fit(
        train, train_labels, save_memory=save_memory, verbose=True
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='SyntheticControl', help='UCR/UEA univariate or multivariate dataset')
    parser.add_argument("--gpu_id", type=str, default="1", help="GPU id")
    parser.add_argument("--visualization", default=True, help='visualization True or None')
    parser.add_argument('--input_dim', default=1, type=int, help="input dimension")
    parser.add_argument("--batch_size", default=4, type=int)



    args = parser.parse_args()

    # GPU Configuration
    gpu_id = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print(args.device)


    # args.path = '/home/jwlee/tensorflow/deep_temporal_clustering/KMdsy_DTCR/Multivariate_ts'
    # args.dataset = 'Heartbeat'
    # args.dataset = 'Coffee'


    ################ Preparing UCR Dataset
    # train_data, train_label = load_ucr_data.load_data('Coffee')       #(56, 286, 1)=(S, T, D)  #(56,)=(0 or 1)
    # train, train_labels, test, test_labels = load_ucr_data.load_UEA_dataset(
    #     args.path, args.dataset
    # )

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

    args.n_clusters = len(np.unique(label_6))
    args.timeseries = sample.shape[1]

    classifier = fit_hyperparameters(args, sample, label_6)
    classifier.fit_classifier(classifier.encode(train), train_labels)

    print("Test accuracy: " + str(classifier.score(test, test_labels)))