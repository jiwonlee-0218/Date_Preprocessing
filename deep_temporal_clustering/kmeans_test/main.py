import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.cluster import adjusted_rand_score

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
import FC_no_embedding





def get_args_parser():
    parser = argparse.ArgumentParser('SiTv2', add_help=False)

    # Model parameters
    parser.add_argument("--dataset_name", default="MOTOR", help="dataset name")
    parser.add_argument('--model', default='vit_tiny', type=str, help="Name of architecture to train.")
    parser.add_argument("--model_name", default="R_T_Kmeans", help="model name")


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


    # parser.add_argument("--weights", default=None, help='pre-trained autoencoder weights')
    parser.add_argument("--weights", default='models_weights/', help='pre-trained autoencoder weights')
    parser.add_argument("--SiT_models", default=None, help='full autoencoder weights')
    # parser.add_argument("--SiT_models", default='full_models/', help='full autoencoder weights')
    parser.add_argument("--SiT_test", default=True, help='full autoencoder weights')

    return parser









def writelog(file, line):
    file.write(line + '\n')
    print(line)






def test_function(args, X_test_data):


    print("Test pretraining model ...")
    if not os.path.exists(os.path.join(directory, 'SiT_models_pictures')):
        os.makedirs(os.path.join(directory, 'SiT_models_pictures'))

    path = os.path.join(directory, args.weights)
    full_path = sorted(glob.glob(path + '*'), key=os.path.getctime)
    full_path = full_path[-1]
    print("I got: " + full_path + " weights")



    full_SiT_model = vits.FullpiplineSiT(args)
    full_SiT_model = full_SiT_model.cuda()



    checkpoint = torch.load(full_path, map_location=args.device)
    full_SiT_model.load_state_dict(checkpoint['model_state_dict'])

    loss_ae = torch.nn.MSELoss()

    full_SiT_model.eval()
    with torch.no_grad():
        X_test_data = X_test_data.to(args.device)
        test_represent, test_recons = full_SiT_model(X_test_data)

        test_loss = loss_ae(X_test_data, test_recons)
        print("Test SiT loss is : %.4f" % test_loss)

    f = open(os.path.join(os.path.join(directory, 'SiT_models_pictures/'), 'SiT\'s test loss.log'),'a')
    writelog(f, '======================')
    writelog(f, 'Model Name: %s' % args.model_name)
    writelog(f, '----------------------')
    writelog(f, 'Epoch: %d' % args.epochs)
    writelog(f, 'Batch Size: %d' % args.batch_size)
    writelog(f, 'Learning Rate: %s' % str(args.lr))
    writelog(f, 'Weight Decay: %s' % str(args.weight_decay))
    writelog(f, '======================')
    writelog(f, 'Test SiT loss loss is : %.4f' % test_loss)
    f.close()

    ########### TEST data's first subject ##############
    plt.xlabel('time')
    plt.ylabel('ROIs')
    plt.imshow(X_test_data[1].cpu().T)
    plt.title("Original_X")
    plt.colorbar()
    plt.savefig(os.path.join(directory, 'SiT_models_pictures/') + 'orig.png')

    plt.xlabel('time')
    plt.ylabel('ROIs')
    plt.imshow(test_recons[1].cpu().detach().T)
    plt.title("Reconstruction")
    plt.savefig(os.path.join(directory, 'SiT_models_pictures/') + 'recon.png')

    plt.xlabel('time')
    plt.ylabel('features')
    plt.imshow(test_represent[1].cpu().detach().T)
    plt.title("Representation")
    plt.savefig(os.path.join(directory, 'SiT_models_pictures/') + 'represented.png')

    plt.xlabel('time')
    plt.ylabel('features')
    plt.imshow(test_represent[1][:150, :].cpu().detach().T)
    plt.title("Representation")
    plt.savefig(os.path.join(directory, 'SiT_models_pictures/') + 'represented_shorts.png')













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

    ################ Preparing Dataset
    data = np.load('/DataCommon/jwlee/MOTOR_LR/hcp_motor.npz')
    samples = data['tfMRI_MOTOR_LR']
    samples = samples[:1080]  # (1041, 284, 116)

    label = data['label_MOTOR_LR']
    label = label[:1080]  # (1041, 284)

    # minmax
    mm = MinMaxScaler()
    results = []
    for ss in range(1080):
        results.append(mm.fit_transform(samples[ss]))
    sample = np.array(results)

    # train, validation, test
    X_train, X_test, y_train, y_test = train_test_split(sample, label, random_state=42, test_size=0.2)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, random_state=42, test_size=0.5)

    X_train, y_train = torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    X_val, y_val = torch.FloatTensor(X_val), torch.FloatTensor(y_val)
    X_test, y_test = torch.FloatTensor(X_test), torch.FloatTensor(y_test)

    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size)  ###
    valid_ds = TensorDataset(X_val, y_val)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size)  ###

    # number of clusters
    args.n_clusters = len(np.unique(y_train))

    sub, labels = next(iter(train_dl))
    print('SUB: ', sub.shape, 'LABEL: ', labels.shape)

    print(f"-------> The dataset consists of {len(train_dl) * args.batch_size} subjects.")  ## 864 subject
    print(f"-------> The dataset size: {sub[0, :, :].shape}")






    if args.weights is None and args.epochs > 0:
        train_SiTv2(args, train_dl, valid_dl, directory)



    if args.weights is not None and args.SiT_models is None and args.SiT_test is None:  #### weight ok, but model is none



        path = os.path.join(directory, args.weights)
        full_path = sorted(glob.glob(path + '*'), key=os.path.getctime)
        full_path = full_path[-1]

        # full_path = '/home/jwlee/HMM/deep_temporal_clustering/mean_for_centroid/transformer/SiT_tiny/MOTOR/Epochs100_BS_16_LR_0.01_wdcay_1e-06/models_weights/checkpoint_epoch_100_loss_0.00459.pt'
        print("I got: " + full_path + " weights")


        Full_SiT_model = vits.FullpiplineSiT(args)


        checkpoint = torch.load(full_path, map_location=args.device)
        Full_SiT_model.load_state_dict(checkpoint['model_state_dict'])


        Full_SiT_model = Full_SiT_model.to(args.device)
        print(Full_SiT_model)

        sample = torch.FloatTensor(sample)
        sample = sample.type(torch.FloatTensor).to(args.device)
        represent, recon = Full_SiT_model(sample)

        print(represent.shape)  # torch.Size([1080, 284, 64])

        represent_np = represent.detach().cpu()

        total_ARI = 0

        f = open(os.path.join(directory, 'KMeans\'s ARI.log'), 'a')
        for i in range(sample.size(0)):

            assignments = KMeans(n_clusters=args.n_clusters).fit(represent_np[i])
            # print(assignments.labels_)
            ARI = adjusted_rand_score(assignments.labels_, label[i])
            # print(np.round(ARI,3))
            writelog(f, 'ARI is: %.4f' % (np.round(ARI,3)))

            total_ARI += np.round(ARI,3)



        avg = (total_ARI / sample.size(0))
        writelog(f, '======================')
        writelog(f, 'subject: %d' % (sample.size(0)))
        writelog(f, 'total_ARI is: %.4f' % (total_ARI))
        writelog(f, 'avg_ARI is: %.4f' % (avg))
        f.close()
        print(avg)



    if args.weights is not None and args.SiT_models is None and args.SiT_test is not None:

        print("Test pretraining model ...")
        if not os.path.exists(os.path.join(directory, 'SiT_models_pictures')):
            os.makedirs(os.path.join(directory, 'SiT_models_pictures'))

        path = os.path.join(directory, args.weights)
        full_path = sorted(glob.glob(path + '*'), key=os.path.getctime)
        full_path = full_path[-1]
        print("I got: " + full_path + " weights")

        full_SiT_model = vits.FullpiplineSiT(args)
        full_SiT_model = full_SiT_model.cuda()

        checkpoint = torch.load(full_path, map_location=args.device)
        full_SiT_model.load_state_dict(checkpoint['model_state_dict'])

        loss_ae = torch.nn.MSELoss()

        full_SiT_model.eval()
        with torch.no_grad():
            X_test_data = X_test.to(args.device)
            test_represent, test_recons = full_SiT_model(X_test_data)

            test_loss = loss_ae(X_test_data, test_recons)
            print("Test SiT loss is : %.4f" % test_loss)

        f = open(os.path.join(os.path.join(directory, 'SiT_models_pictures/'), 'SiT\'s test loss.log'), 'a')
        writelog(f, '======================')
        writelog(f, 'Model Name: %s' % args.model_name)
        writelog(f, '----------------------')
        writelog(f, 'Epoch: %d' % args.epochs)
        writelog(f, 'Batch Size: %d' % args.batch_size)
        writelog(f, 'Learning Rate: %s' % str(args.lr))
        writelog(f, 'Weight Decay: %s' % str(args.weight_decay))
        writelog(f, '======================')
        writelog(f, 'Test SiT loss loss is : %.4f' % test_loss)
        f.close()

        ########### TEST data's first subject ##############
        plt.xlabel('time')
        plt.ylabel('ROIs')
        plt.imshow(X_test_data[1].cpu().T)
        plt.title("Original_X")
        plt.colorbar()
        plt.savefig(os.path.join(directory, 'SiT_models_pictures/') + 'orig.png')

        plt.xlabel('time')
        plt.ylabel('ROIs')
        plt.imshow(test_recons[1].cpu().detach().T)
        plt.title("Reconstruction")
        plt.savefig(os.path.join(directory, 'SiT_models_pictures/') + 'recon.png')

        plt.xlabel('time')
        plt.ylabel('features')
        plt.imshow(test_represent[1].cpu().detach().T)
        plt.title("Representation")
        plt.savefig(os.path.join(directory, 'SiT_models_pictures/') + 'represented.png')

        plt.xlabel('time')
        plt.ylabel('features')
        plt.imshow(test_represent[1][:150, :].cpu().detach().T)
        plt.title("Representation")
        plt.savefig(os.path.join(directory, 'SiT_models_pictures/') + 'represented_shorts.png')












