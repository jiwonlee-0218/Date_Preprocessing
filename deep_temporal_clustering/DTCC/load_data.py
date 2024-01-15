import numpy as np
from sklearn.preprocessing import LabelEncoder
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.preprocessing import StandardScaler
from tslearn.datasets import UCR_UEA_datasets
import copy


ucr = UCR_UEA_datasets()
all_ucr_datasets = ucr.list_datasets()

def load_ucr(dataset='CBF'):
    X_train, y_train, X_test, y_test = ucr.load_dataset(dataset)
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    if dataset == 'HandMovementDirection':  # this one has special labels
        y = [yy[0] for yy in y]
    y = LabelEncoder().fit_transform(y)  # sometimes labels are strings or start from 1
    assert(y.min() == 0)  # assert labels are integers and start from 0
    # preprocess data (standardization)
    scr = StandardScaler()
    results = []
    for ss in range(X.shape[0]):
        results.append(scr.fit_transform(X[ss]))
    X_scaled = np.array(results)
    return X_scaled, y


def load_data(dataset_name):
    if dataset_name in all_ucr_datasets:
        return load_ucr(dataset_name)
    else:
        print('Dataset {} not available! Available datasets are UCR/UEA univariate and multivariate datasets.'.format(dataset_name))
        exit(0)



def shuffle_timeseries(data, rate=0.2):

    ordered_index = np.arange(len(data))
    ordered_index.astype(int)

    np.random.seed(42)
    shuffled_index = np.random.choice(ordered_index, size=int(np.floor(rate * len(data))), replace=False)
    ordered_index[shuffled_index] = -1

    shuffled_index = np.random.permutation(shuffled_index)
    ordered_index[ordered_index == -1] = shuffled_index
    data = data[ordered_index]

    return data


def construct_classification_dataset(dataset):
    real_dataset = copy.deepcopy(dataset)
    fake_dataset = []
    for seq in real_dataset:
        fake_dataset.append(shuffle_timeseries(seq))

    fake_dataset = np.array(fake_dataset)

    fake_label = np.array([1] * fake_dataset.shape[0])
    true_label = np.array([0] * real_dataset.shape[0])
    dataset = np.concatenate([fake_dataset, real_dataset], axis=0)  # (112, 286, 1)

    print('true dataset shape: ', real_dataset.shape)
    print('true label shape:', true_label.shape)

    return real_dataset, fake_dataset, true_label, fake_label

