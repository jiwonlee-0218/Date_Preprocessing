import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.cluster import adjusted_rand_score
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
from FC_model import TAE, FullpiplineSiT





def get_args_parser():
    parser = argparse.ArgumentParser('SiTv2', add_help=False)

    # Model parameters
    parser.add_argument("--dataset_name", default="EMOTION_6_ViT+masking", help="dataset name")
    parser.add_argument("--model_name", default="FC_Kmeans", help="model name")



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
    parser.add_argument("--SiT_test", default=None, help='full autoencoder weights')

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
    samples = data['tfMRI_EMOTION_LR']  # (1041, 6, 6670)

    label = [0,1,0,1,0,1]
    label = np.array(label)
    label = np.expand_dims(label, 0)
    labels = np.repeat(label, 1041, 0) # (1041, 6)




    X_train, X_test, y_train, y_test = train_test_split(samples, labels, random_state=42, shuffle=False, test_size=0.2)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, random_state=42, shuffle=False, test_size=0.5)
    X_train, y_train = torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    X_val, y_val = torch.FloatTensor(X_val), torch.FloatTensor(y_val)
    X_test, y_test = torch.FloatTensor(X_test), torch.FloatTensor(y_test)

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

        ## define TAE architecture
        Full_SiT_model = FullpiplineSiT(args)
        Full_SiT_model = Full_SiT_model.to(args.device)
        print(Full_SiT_model)

        ## MSE loss
        loss_ae = nn.MSELoss()
        ## Optimizer
        optimizer = torch.optim.Adam(Full_SiT_model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

        for epoch in range(args.epochs):

            # training
            Full_SiT_model.train()
            all_loss = 0


            for batch_idx, (inputs, _) in enumerate(train_dl):
                inputs = inputs.type(torch.FloatTensor).to(args.device)

                optimizer.zero_grad()  # 기울기에 대한 정보 초기화
                features, x_reconstr = Full_SiT_model(inputs)
                loss_mse = loss_ae(inputs, x_reconstr)  # x_reconstr(decoded) & 원본(input) 사이의 평균제곱오차
                loss_mse.backward()  # 기울기 구함


                optimizer.step()  # 최적화 진행

                all_loss += loss_mse.item()

            train_loss = all_loss / (batch_idx + 1)


            writer.add_scalar("training loss", train_loss, epoch+1)
            print("training autoencoder loss for epoch {} is : {}".format(epoch + 1, train_loss))

            # validation
            Full_SiT_model.eval()
            with torch.no_grad():
                all_val_loss = 0
                for j, (val_x, val_y) in enumerate(valid_dl):
                    val_x = val_x.type(torch.FloatTensor).to(args.device)
                    v_features, val_reconstr = Full_SiT_model(val_x)
                    val_loss = loss_ae(val_x, val_reconstr)

                    all_val_loss += val_loss.item()

                validation_loss = all_val_loss / (j + 1)

                writer.add_scalar("validation loss", validation_loss, epoch+1)
                print("val_loss for epoch {} is : {}".format(epoch + 1, validation_loss))

            if epoch == 0:
                min_val_loss = validation_loss

            if validation_loss < min_val_loss:
                torch.save({
                    'model_state_dict': Full_SiT_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'loss': validation_loss
                }, os.path.join(directory, 'models_weights') + '/checkpoint_epoch_{}_loss_{:.5f}.pt'.format(epoch + 1,
                                                                                                            validation_loss))
                min_val_loss = validation_loss
                print("save weights !!")

        writer.close()
        print("Ending pretraining autoencoder. \n")



    if args.weights is not None and args.SiT_test is not None:  #### weight ok, but model is none

        print("Test pretraining model ...")
        if not os.path.exists(os.path.join(directory, 'SiT_models_pictures')):
            os.makedirs(os.path.join(directory, 'SiT_models_pictures'))

        path = os.path.join(directory, args.weights)
        full_path = sorted(glob.glob(path + '*'), key=os.path.getctime)
        full_path = full_path[-1]
        print("I got: " + full_path + " weights")

        Full_SiT_model = FullpiplineSiT(args)
        Full_SiT_model = Full_SiT_model.to(args.device)

        checkpoint = torch.load(full_path, map_location=args.device)
        Full_SiT_model.load_state_dict(checkpoint['model_state_dict'])

        loss_ae = torch.nn.MSELoss()
        Full_SiT_model.eval()
        with torch.no_grad():
            X_test_data = X_test.to(args.device)
            test_represent, test_recons = Full_SiT_model(X_test_data)

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
        plt.imshow(X_test_data[100].cpu().T.reshape(116, -1))
        plt.title("Original_X")
        plt.colorbar()
        plt.savefig(os.path.join(directory, 'SiT_models_pictures/') + 'orig_100.png')

        plt.xlabel('time')
        plt.ylabel('ROIs')
        plt.imshow(test_recons[100].cpu().detach().T.reshape(116, -1))
        plt.title("Reconstruction")
        plt.savefig(os.path.join(directory, 'SiT_models_pictures/') + 'recon_100.png')

        plt.xlabel('time')
        plt.ylabel('features')
        plt.imshow(test_represent[100].cpu().detach().T.reshape(32, -1))
        plt.title("Representation")
        plt.savefig(os.path.join(directory, 'SiT_models_pictures/') + 'represented_100.png')

        # plt.xlabel('time')
        # plt.ylabel('features')
        # plt.imshow(test_represent[1][:150, :].cpu().detach().T)
        # plt.title("Representation")
        # plt.savefig(os.path.join(directory, 'SiT_models_pictures/') + 'represented_shorts.png')












    if args.weights is not None and args.SiT_test is None:

        path = os.path.join(directory, args.weights)
        full_path = sorted(glob.glob(path + '*'), key=os.path.getctime)
        full_path = full_path[-1]
        print("I got: " + full_path + " weights")

        Full_SiT_model = FullpiplineSiT(args)
        Full_SiT_model = Full_SiT_model.to(args.device)

        checkpoint = torch.load(full_path, map_location=args.device)
        Full_SiT_model.load_state_dict(checkpoint['model_state_dict'])

        # samples = torch.FloatTensor(samples)
        # samples = samples.type(torch.FloatTensor).to(args.device)
        X_test_data = X_test.to(args.device)

        features, output = Full_SiT_model(X_test_data)
        print(features.shape)

        features_np = features.detach().cpu()
        ff = features_np.reshape(-1, 1024)
        total_NMI = 0
        # f = open(os.path.join(directory, 'KMeans\'s NMI.log'), 'a')



        kmc = KMeans(n_clusters=2, init="k-means++", max_iter=1000)
        kmc.fit(ff)

        for i in range(features.size(0)):

            p_kmc = kmc.predict(features_np[i])
            NMI = normalized_mutual_info_score(p_kmc, labels[i])
            # print(np.round(ARI,3))
            # writelog(f, 'NMI is: %.4f' % (np.round(NMI,3)))

            total_NMI += np.round(NMI,3)



        avg = (total_NMI / samples.shape[0] )
        # writelog(f, '======================')
        # writelog(f, 'subject: %d' % ( samples.shape[0] ))
        # writelog(f, 'total_NMI is: %.4f' % (total_NMI))
        # writelog(f, 'avg_NMI is: %.4f' % (avg))
        # f.close()
        print(avg)












