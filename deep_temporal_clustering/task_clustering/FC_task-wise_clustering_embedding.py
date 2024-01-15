import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import normalized_mutual_info_score, rand_score

import argparse
import torch.nn as nn
from functools import partial

import os


from pathlib import Path

import torch
import glob
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
import mab_model



def writelog(file, line):
    file.write(line + '\n')
    print(line)


def get_args_parser():
    parser = argparse.ArgumentParser('SiTv2', add_help=False)

    # Model parameters
    parser.add_argument("--dataset_name", default="RELATIONAL_6", help="dataset name")
    parser.add_argument('--model', default='vit_tiny', type=str, help="Name of architecture to train.")
    parser.add_argument("--model_name", default="FC_Kmeans", help="model name")
    parser.add_argument("--methods", default="MLP",  type=str)


    parser.add_argument('--img_size', default=284, type=int, help="Input size to the Transformer.")
    parser.add_argument('--patch_size', default=1, type=int, help="patch size to the Transformer.")
    parser.add_argument("--similarity", required=False, choices=["COR", "EUC", "CID", "COS"], default="COS", help="The similarity type")



    # Hyper-parameters
    parser.add_argument('--batch_size', default=16, type=int, help="Batch size per GPU.")
    parser.add_argument('--epochs', default=30, type=int, help="Number of epochs of training.")
    parser.add_argument("--max_epochs", type=int, default=250, help="Maximum epochs numer of the full model training", )

    parser.add_argument('--weight_decay', type=float, default=1e-6, help="weight decay")
    parser.add_argument("--lr", default=0.0005, type=float, help="Learning rate.")



    # GPU
    parser.add_argument("--gpu_id", type=str, default="2", help="GPU id")


    # directory
    parser.add_argument("--dir_root", default='/home/jwlee/HMM/deep_temporal_clustering/task_clustering/')


    parser.add_argument("--weights", default=None, help='pre-trained autoencoder weights')
    # parser.add_argument("--weights", default='models_weights/', help='pre-trained autoencoder weights')
    parser.add_argument("--SiT_models", default=None, help='full autoencoder weights')
    # parser.add_argument("--SiT_models", default='full_models/', help='full autoencoder weights')
    parser.add_argument("--SiT_test", default=None, help='full autoencoder weights')


    return parser









def writelog(file, line):
    file.write(line + '\n')
    print(line)


def cluster_using_kmeans(embeddings, K):
    return KMeans(n_clusters=K).fit(embeddings)

def cluster_using_gmm(embeddings, K):
    return GaussianMixture(n_components=K).fit(embeddings)




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
    emotion_data = np.load('/DataCommon/jwlee/EMOTION_LR/cluster_2_hcp_emotion_FC.npz')
    emotion_samples = emotion_data['tfMRI_EMOTION_LR'] # (1041, 6, 6670)

    label = [0, 0, 0, 0, 0, 0]
    label = np.array(label)
    label = np.expand_dims(label, 0)
    labels = np.repeat(label, 1041, 0) # (1041, 6)


    ''' binary classification one-hot encoding '''
    num = 2 # class 개수
    emotion_labels = np.eye(num)[labels] # (1041, 6, 2)



    ''' RELATIONAL '''
    ################ Preparing Dataset
    relational_data = np.load('/DataCommon/jwlee/RELATIONAL_LR/cluster_2_hcp_relational_FC.npz')
    relational_samples = relational_data['tfMRI_RELATIONAL_LR']  # (1040, 6, 6670)

    R_label = [1, 1, 1, 1, 1, 1]
    R_label = np.array(R_label)
    R_label = np.expand_dims(R_label, 0)
    R_labels = np.repeat(R_label, 1040, 0) # (1040, 6)


    ''' binary classification one-hot encoding '''
    num = 2  # class 개수
    relational_labels = np.eye(num)[R_labels]  # (1041, 6, 2)





    sum_samples = np.concatenate((emotion_samples, relational_samples), 0) #(2081, 6, 6670)
    sum_labels = np.concatenate((emotion_labels, relational_labels), 0) # (2081, 6, 2)


    X_train, X_test, y_train, y_test = train_test_split(sum_samples, sum_labels, random_state=42, shuffle=True, test_size=0.2)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, shuffle=False, random_state=42, test_size=0.5)
    X_train, y_train = torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    X_val, y_val = torch.FloatTensor(X_val), torch.FloatTensor(y_val)
    X_test, y_test = torch.FloatTensor(X_test), torch.FloatTensor(y_test)


    X_train = X_train.reshape(-1, 6670)    # (9984, 6670)
    y_train = y_train.reshape(-1, 2)       # (9984,)
    X_test = X_test.reshape(-1, 6670)      # (1248, 6670)
    y_test = y_test.reshape(-1, 2)
    X_val = X_val.reshape(-1, 6670)        # (1254, 6670)
    y_val = y_val.reshape(-1, 2)

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

        net = mab_model.Net(args)
        net = net.to(args.device)
        print(net)

        ## Loss
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

        # test data -> kmeans
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

            accuracy = np.mean(preds == preds_d)
            ACC = metrics.accuracy_score(preds, preds_d)
            print('accuracy: ', accuracy)

            f = open(os.path.join(directory, 'MLP Classification\'s test Acc.log'), 'a')
            writelog(f, '======================')
            writelog(f, 'Model Name: %s' % args.methods)
            writelog(f, 'Test Accuracy: %.4f' % accuracy)
            writelog(f, 'Test Accuracy: %.4f' % ACC)

            f.close()





    if args.weights is not None and args.SiT_test is None:

        # path = os.path.join(directory, args.weights)
        path = '/home/jwlee/HMM/deep_temporal_clustering/task_clustering/FC_classification/RELATIONAL_6/Epochs30_BS_16_LR_0.0005_wdcay_1e-06/models_weights/'
        full_path = sorted(glob.glob(path + '*'), key=os.path.getctime)
        full_path = full_path[-1]
        print("I got: " + full_path + " weights")

        net = mab_model.Net(args)
        net = net.to(args.device)

        checkpoint = torch.load(full_path, map_location=args.device)
        net.load_state_dict(checkpoint['model_state_dict'])


        sum_labels = np.concatenate((labels, R_labels), 0)
        X_train, X_test, y_train, y_test = train_test_split(sum_samples, sum_labels, random_state=42, shuffle=True, test_size=0.5) # X_test -> (1041, 6, 6670)
        X_test_net = torch.FloatTensor(X_test).to(args.device)
        X_test_net = X_test_net.reshape(-1, 6670)     # X_test_net -> torch.Size([6246, 6670])

        ############# modify
        X_train_net = torch.FloatTensor(X_train).to(args.device)
        X_train_net = X_train_net.reshape(-1, 6670)  # torch.Size([6240, 6670])
        output_X = net(X_train_net)
        output_X = output_X.detach().cpu().numpy() # (6240, 2)
        kmc = cluster_using_kmeans(output_X, 2)
        gmm = cluster_using_gmm(output_X, 2)
        output_train = output_X.reshape(-1, 6, 2)  # (1040, 6, 2)


        # output = net(X_test_net) # torch.Size([1248, 2])
        # print(output.shape)  # output -> torch.Size([6246, 2])
        #
        # output = output.detach().cpu().numpy()   # output -> (6246, 2)



        k_total_NMI = 0
        k_total_RI = 0
        g_total_NMI = 0
        g_total_RI = 0

        # f = open(os.path.join(directory, 'KMeans\'s train NMI.log'), 'a')
        # k = open(os.path.join(directory, 'KMeans\'s train RI.log'), 'a')
        # writelog(f, '=======================================================================')
        # writelog(f, 'Task-wise FC -> KMeans clustering (MLP) -> (6670 dimension to 2 dimension)')
        # writelog(k, 'Task-wise FC -> KMeans clustering (MLP) -> (6670 dimension to 2 dimension)')



        for i in range(output_train.shape[0]):
            L_kmc = kmc.predict(output_train[i])
            L_gmm = gmm.predict(output_train[i])

            k_NMI = normalized_mutual_info_score(L_kmc, y_train[i])
            k_RI = rand_score(L_kmc, y_train[i])

            g_NMI = normalized_mutual_info_score(L_gmm, y_train[i])
            g_RI = rand_score(L_gmm, y_train[i])



            # writelog(f, 'train_NMI is: %.4f' % (np.round(NMI, 3)))
            # writelog(k, 'train_RI is: %.4f' % (np.round(RI, 3)))

            k_total_NMI += k_NMI
            k_total_RI += k_RI
            g_total_NMI += g_NMI
            g_total_RI += g_RI



        avg_k_NMI = (k_total_NMI / output_train.shape[0])
        avg_k_RI = (k_total_RI / output_train.shape[0])
        avg_g_NMI = (g_total_NMI / output_train.shape[0])
        avg_g_RI = (g_total_RI / output_train.shape[0])

        # writelog(f, '======================')
        # writelog(f, 'subject: %d' % (output_train.shape[0]))
        # writelog(f, 'total_train_NMI is: %.4f' % (total_NMI))
        # writelog(f, 'avg_train_NMI is: %.4f' % (avg_NMI))
        # f.close()
        #
        # writelog(k, '======================')
        # writelog(k, 'subject: %d' % (output_train.shape[0]))
        # writelog(k, 'total_train_RI is: %.4f' % (total_RI))
        # writelog(k, 'avg_train_RI is: %.4f' % (avg_RI))
        # k.close()





        ''' visualization '''

        if not os.path.exists(os.path.join(directory, 'MLP_train_clustering_visulization/')):
            os.makedirs(os.path.join(directory, 'MLP_train_clustering_visulization/'))


        # predict = np.argmax(output, 1)
        pmc = kmc.predict(output_X)  # (6246,)

        centroid = kmc.cluster_centers_

        ''' predict '''
        predicted_0 = output_X[pmc == 0]
        predicted_1 = output_X[pmc == 1]

        plt.scatter(predicted_0[:, 0], predicted_0[:, 1], label = 'predicted_0' , color='green', alpha=0.5)
        plt.scatter(predicted_1[:, 0], predicted_1[:, 1], label = 'predicted_1', color='olive', alpha=0.5)
        # plt.scatter(centroid[:, 0], centroid[:1], s=70, color='black')
        plt.legend()
        # plt.show()
        plt.savefig(os.path.join(directory, 'MLP_train_clustering_visulization/') + 'predicted_Kmeans.png')
        plt.close()


        plt.scatter(predicted_0[:, 0], predicted_0[:, 1], color='green', alpha=0.5)
        plt.savefig(os.path.join(directory, 'MLP_train_clustering_visulization/') + 'predicted_Kmeans_0.png')
        plt.close()


        plt.scatter(predicted_1[:, 0], predicted_1[:, 1], color='olive', alpha=0.5)
        plt.savefig(os.path.join(directory, 'MLP_train_clustering_visulization/') + 'predicted_Kmeans_1.png')
        plt.close()



        ''' label '''
        y_train = y_train.flatten()
        # y_test = y_test.flatten()
        labeled_0 = output_X[y_train == 0]
        labeled_1 = output_X[y_train == 1]

        plt.scatter(labeled_0[:, 0], labeled_0[:, 1], label='emotion_label', color='red', alpha=0.5)
        plt.scatter(labeled_1[:, 0], labeled_1[:, 1], label='relational_label', color='blue', alpha=0.5)
        plt.legend()
        # plt.show()
        plt.savefig(os.path.join(directory, 'MLP_train_clustering_visulization/') + 'label_Kmeans.png')
        plt.close()

        ''' original '''
        plt.scatter(output_X[:, 0], output_X[:, 1], color='orange', alpha=0.5)
        # plt.show()
        plt.savefig(os.path.join(directory, 'MLP_train_clustering_visulization/') + 'orig_data.png')
        plt.close()

        ''' emotion '''
        plt.scatter(labeled_0[:, 0], labeled_0[:, 1], label='emotion_label', color='red', alpha=0.5)
        plt.savefig(os.path.join(directory, 'MLP_train_clustering_visulization/') + 'label_emotion.png')
        plt.close()


        ''' relational '''
        plt.scatter(labeled_1[:, 0], labeled_1[:, 1], label='relational_label', color='blue', alpha=0.5)
        plt.savefig(os.path.join(directory, 'MLP_train_clustering_visulization/') + 'label_relational.png')
        plt.close()


        # ''' classification '''
        # predict = np.argmax(output_X, 1)
        # cluster_predicted_0 = output_X[predict == 0]
        # cluster_predicted_1 = output_X[predict == 1]
        #
        # plt.scatter(cluster_predicted_0[:, 0], cluster_predicted_0[:, 1], label = 'classified_emotion', color='green', alpha=0.5)
        # plt.scatter(cluster_predicted_1[:, 0], cluster_predicted_1[:, 1], label = 'classified_relational', color='olive', alpha=0.5)
        # plt.legend()
        # plt.savefig(os.path.join(directory, 'MLP_train_clustering_visulization/') + 'predicted_GMM.png')
        # plt.close()
        #
        #
        # plt.scatter(cluster_predicted_0[:, 0], cluster_predicted_0[:, 1], label = 'classified_emotion', color='green', alpha=0.5)
        # plt.savefig(os.path.join(directory, 'MLP_train_clustering_visulization/GMM/') + 'classified_emotion.png')
        # plt.close()
        #
        # plt.scatter(cluster_predicted_1[:, 0], cluster_predicted_1[:, 1], label = 'classified_relational', color='olive', alpha=0.5)
        # plt.savefig(os.path.join(directory, 'MLP_train_clustering_visulization/') + 'classified_relational.png')
        # plt.close()


        print('finish')