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


from train import *
from FC_model import Net






def get_args_parser():
    parser = argparse.ArgumentParser('SiTv2', add_help=False)

    # Model parameters
    parser.add_argument("--dataset_name", default="EMOTION_6", help="dataset name")
    parser.add_argument('--model', default='vit_tiny', type=str, help="Name of architecture to train.")
    parser.add_argument("--model_name", default="FC_classification", help="model name")


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
    parser.add_argument("--dir_root", default='/home/jwlee/HMM/deep_temporal_clustering/kmeans_test/')


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
    data = np.load('/DataCommon/jwlee/EMOTION_LR/cluster_2_hcp_emotion_FC.npz')
    samples = data['tfMRI_EMOTION_LR'] # (1041, 6, 6670)

    label = [0,1,0,1,0,1]
    label = np.array(label)
    label = np.expand_dims(label, 0)
    labels = np.repeat(label, 1041, 0)



    num = np.unique(labels.flatten(), axis=0)
    num = num.shape[0]
    encoding = np.eye(num)[labels]

    X_train, X_test, y_train, y_test = train_test_split(samples, encoding, shuffle=False,random_state=42, test_size=0.2)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, shuffle=False, random_state=42, test_size=0.5)
    X_train, y_train = torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    X_val, y_val = torch.FloatTensor(X_val), torch.FloatTensor(y_val)
    X_test, y_test = torch.FloatTensor(X_test), torch.FloatTensor(y_test)


    X_train = X_train.reshape(-1, 6670)  # torch.Size([4992, 6670])
    y_train = y_train.reshape(-1, 2)  # ttorch.Size([4992, 2])
    X_val = X_val.reshape(-1, 6670)  # torch.Size([630, 6670])
    y_val = y_val.reshape(-1, 2)
    X_test = X_test.reshape(-1, 6670)  # torch.Size([624, 6670])
    y_test = y_test.reshape(-1, 2)



    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size)  ###
    valid_ds = TensorDataset(X_val, y_val)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size)  ###


    if args.weights is None:
        if not os.path.exists(os.path.join(directory, 'models_logs')):
            os.makedirs(os.path.join(directory, 'models_logs'))

        if not os.path.exists(os.path.join(directory, 'models_weights')):
            os.makedirs(os.path.join(directory, 'models_weights'))

        writer = SummaryWriter(log_dir=os.path.join(directory, 'models_logs'))


        net = Net(args)
        net = net.to(args.device)
        print(net)

        ## MSE loss
        loss_ae = nn.CrossEntropyLoss()
        ## Optimizer
        # optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)


        for epoch in range(args.epochs):

            # training
            net.train()
            all_loss = 0

            for batch_idx, (inputs, y) in enumerate(train_dl):
                inputs = inputs.type(torch.FloatTensor).to(args.device)
                y = y.type(torch.FloatTensor).to(args.device)

                optimizer.zero_grad()  # 기울기에 대한 정보 초기화
                outputs = net(inputs)
                loss_mse = loss_ae(outputs, y)  # x_reconstr(decoded) & 원본(input) 사이의 평균제곱오차
                loss_mse.backward()  # 기울기 구함
                optimizer.step()  # 최적화 진행

                all_loss += loss_mse.item()

            train_loss = all_loss / (batch_idx + 1)


            writer.add_scalar("training loss", train_loss, epoch + 1)
            print("training autoencoder loss for epoch {} is : {}".format(epoch + 1, train_loss))

            # validation
            net.eval()
            with torch.no_grad():
                all_val_loss = 0
                for j, (val_x, val_y) in enumerate(valid_dl):
                    val_x = val_x.type(torch.FloatTensor).to(args.device)
                    val_y = val_y.type(torch.FloatTensor).to(args.device)
                    val_outputs = net(val_x)
                    val_loss = loss_ae(val_outputs, val_y)

                    all_val_loss += val_loss.item()

                validation_loss = all_val_loss / (j + 1)

                writer.add_scalar("validation loss", validation_loss, epoch + 1)
                print("val_loss for epoch {} is : {}".format(epoch + 1, validation_loss))


            if epoch == 0:
                min_val_loss = validation_loss

            if validation_loss < min_val_loss:
                torch.save({
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'loss': validation_loss
                }, os.path.join(directory, 'models_weights') + '/checkpoint_epoch_{}_loss_{:.5f}.pt'.format(epoch + 1,
                                                                                                            validation_loss))
                min_val_loss = validation_loss
                print("save weights !!")

        writer.close()
        print("Ending classification. \n")


        # test
        net.eval()
        y_pred = []
        y_true = []

        with torch.no_grad():
            X_test = X_test.to(args.device)

            outputs = net(X_test)
            _, preds = torch.max(outputs, 1)
            preds = preds.detach().cpu().numpy()
            # preds = preds[1:].reshape(-1, 6)


            d, preds_d = torch.max(y_test, 1)
            preds_d = preds_d.detach().cpu().numpy()

            accuracy = np.mean(preds==preds_d)
            print('accuracy: ', accuracy)







