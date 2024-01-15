import numpy as np
import copy



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

    label = np.array([1] * fake_dataset.shape[0] + [0] * real_dataset.shape[0])
    dataset = np.concatenate([fake_dataset, real_dataset], axis=0)  # (112, 286, 1)

    print('dataset shape: ', dataset.shape)
    print('label shape:', label.shape)

    return dataset, label