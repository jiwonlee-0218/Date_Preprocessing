import numpy as np
from sklearn.preprocessing import LabelEncoder
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.preprocessing import StandardScaler
from tslearn.datasets import UCR_UEA_datasets


# ucr = UCR_UEA_datasets()
# all_ucr_datasets = ucr.list_datasets()
#
# def load_ucr(dataset='CBF'):
#     X_train, y_train, X_test, y_test = ucr.load_dataset(dataset)
#     X = np.concatenate((X_train, X_test))
#     y = np.concatenate((y_train, y_test))
#     if dataset == 'HandMovementDirection':  # this one has special labels
#         y = [yy[0] for yy in y]
#     y = LabelEncoder().fit_transform(y)  # sometimes labels are strings or start from 1
#     assert(y.min() == 0)  # assert labels are integers and start from 0
#     # preprocess data (standardization)
#     scr = StandardScaler()
#     results = []
#     for ss in range(X.shape[0]):
#         results.append(scr.fit_transform(X[ss]))
#     X_scaled = np.array(results)
#     return X_scaled, y
#
#
# def load_data(dataset_name):
#     if dataset_name in all_ucr_datasets:
#         return load_ucr(dataset_name)
#     else:
#         print('Dataset {} not available! Available datasets are UCR/UEA univariate and multivariate datasets.'.format(dataset_name))
#         exit(0)




import os
import json
import math
import torch
import numpy
import pandas
import argparse

import scikit_wrappers


def load_UCR_dataset(path, dataset):
    """
    Loads the UCR dataset given in input in numpy arrays.
    @param path Path where the UCR dataset is located.
    @param dataset Name of the UCR dataset.
    @return Quadruplet containing the training set, the corresponding training
            labels, the testing set and the corresponding testing labels.
    """
    train_file = os.path.join(path, dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join(path, dataset, dataset + "_TEST.tsv")
    train_df = pandas.read_csv(train_file, sep='\t', header=None)
    test_df = pandas.read_csv(test_file, sep='\t', header=None)
    train_array = numpy.array(train_df)
    test_array = numpy.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = numpy.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = numpy.expand_dims(train_array[:, 1:], 1).astype(numpy.float64)
    train_labels = numpy.vectorize(transform.get)(train_array[:, 0])
    test = numpy.expand_dims(test_array[:, 1:], 1).astype(numpy.float64)
    test_labels = numpy.vectorize(transform.get)(test_array[:, 0])

    # Normalization for non-normalized datasets
    # To keep the amplitude information, we do not normalize values over
    # individual time series, but on the whole dataset
    if dataset not in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        return train, train_labels, test, test_labels
    # Post-publication note:
    # Using the testing set to normalize might bias the learned network,
    # but with a limited impact on the reported results on few datasets.
    # See the related discussion here: https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries/pull/13.
    mean = numpy.nanmean(numpy.concatenate([train, test]))
    var = numpy.nanvar(numpy.concatenate([train, test]))
    train = (train - mean) / math.sqrt(var)
    test = (test - mean) / math.sqrt(var)
    return train, train_labels, test, test_labels



def fill_out_with_Nan(data,max_length):
    #via this it can works on more dimensional array
    pad_length = max_length-data.shape[-1]
    if pad_length == 0:
        return data
    else:
        pad_shape = list(data.shape[:-1])
        pad_shape.append(pad_length)
        Nan_pad = np.empty(pad_shape)*np.nan
        return np.concatenate((data, Nan_pad), axis=-1)


def set_nan_to_zero(a):
    where_are_NaNs = np.isnan(a)
    a[where_are_NaNs] = 0
    return a


def get_data_and_label_from_ts_file(file_path, label_dict):
    with open(file_path) as file:
        lines = file.readlines()
        Start_reading_data = False
        Label_list = []
        Data_list = []
        max_length = 0
        for line in lines:
            if Start_reading_data == False:
                if '@data' in line:
                    Start_reading_data = True
            else:
                temp = line.split(':')
                Label_list.append(label_dict[temp[-1].replace('\n', '')])
                data_tuple = [np.expand_dims(np.fromstring(channel, sep=','), axis=0) for channel in temp[:-1]]
                max_channel_length = 0
                for channel_data in data_tuple:
                    if channel_data.shape[-1] > max_channel_length:
                        max_channel_length = channel_data.shape[-1]
                data_tuple = [fill_out_with_Nan(data, max_channel_length) for data in data_tuple]
                data = np.expand_dims(np.concatenate(data_tuple, axis=0), axis=0)
                Data_list.append(data)
                if max_channel_length > max_length:
                    max_length = max_channel_length

        Data_list = [fill_out_with_Nan(data, max_length) for data in Data_list]
        X = np.concatenate(Data_list, axis=0)
        Y = np.asarray(Label_list)

        return np.float32(X), Y



def get_label_dict(file_path):
    label_dict = {}
    with open(file_path) as file:
        lines = file.readlines()
        for line in lines:
            if '@classLabel' in line:
                label_list = line.replace('\n', '').split(' ')[2:]
                for i in range(len(label_list)):
                    label_dict[label_list[i]] = i

                break
    return label_dict

def load_UEA_dataset(path, dataset):
    """
    Loads the UEA dataset given in input in numpy arrays.
    @param path Path where the UCR dataset is located.
    @param dataset Name of the UCR dataset.
    @return Quadruplet containing the training set, the corresponding training
            labels, the testing set and the corresponding testing labels.
    """

    Train_dataset_path = os.path.join(path, dataset, dataset + "_TRAIN.ts")
    Test_dataset_path = os.path.join(path, dataset, dataset + "_TEST.ts")
    label_dict = get_label_dict(Train_dataset_path)
    X_train, y_train = get_data_and_label_from_ts_file(Train_dataset_path, label_dict)
    X_test, y_test = get_data_and_label_from_ts_file(Test_dataset_path, label_dict)


    return set_nan_to_zero(X_train), y_train, set_nan_to_zero(X_test), y_test