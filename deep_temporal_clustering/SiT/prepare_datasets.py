import os


from deep_temporal_clustering.SiT.masking.CIFAR import CIFAR100


import torchvision


def build_dataset(args, is_train, trnsfrm=None, training_mode='SSL'):
    if args.data_set == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10(os.path.join(args.data_location, 'CIFAR10_dataset'),
                          train=is_train, transform=trnsfrm,download=True)

        nb_classes = 10


    elif args.data_set == 'MNIST':
        dataset = torchvision.datasets.MNIST(os.path.join(args.data_location, 'MNIST_dataset'),
                                             train=is_train, transform=trnsfrm, download=True)

        nb_classes = 10

    elif args.data_set == 'CIFAR100':
        dataset = CIFAR100(os.path.join(args.data_location, 'CIFAR100_dataset'),
                           train=is_train, transform=trnsfrm,download=True,
                           training_mode=training_mode)

        nb_classes = 100



    return dataset, nb_classes