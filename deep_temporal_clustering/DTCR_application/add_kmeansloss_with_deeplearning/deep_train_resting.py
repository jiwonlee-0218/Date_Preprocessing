import torch
import datetime
import torch.nn as nn
import os
import numpy as np
from deep_main_config import get_arguments
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import deep_utils
from deep_KM_model import Seq2Seq, TCNModel, TCM_DRNN
import random
import glob
import matplotlib.pyplot as plt
import warnings
from sklearn.manifold import TSNE

from tslearn.clustering import TimeSeriesKMeans


def writelog(file, line):
    file.write(line + '\n')
    print(line)



def pretrain_autoencoder(args, train_dl, test_dl, kmeans_train, kmeans_test, directory='.'):
    """
    function for the autoencoder pretraining
    """


    if not os.path.exists(directory):
        os.makedirs(directory)

    if not os.path.exists(os.path.join(directory, 'models_logs')):
        os.makedirs(os.path.join(directory, 'models_logs'))

    if not os.path.exists(os.path.join(directory, 'models_weights')):
        os.makedirs(os.path.join(directory, 'models_weights'))

    if not os.path.exists(os.path.join(directory, 'visualization')):
        os.makedirs(os.path.join(directory, 'visualization/EPOCH'))

    # Text Logging
    f = open(os.path.join(directory, 'setting.log'), 'a')
    writelog(f, '======================')
    writelog(f, 'GPU ID: %s' % (args.gpu_id))
    writelog(f, 'Dataset: %s' % (args.dataset_name))
    writelog(f, 'number of clusters: %d' % (args.n_clusters))
    writelog(f, '----------------------')
    writelog(f, 'Model Name: %s' % args.model_name)
    writelog(f, '----------------------')
    writelog(f, 'Epoch: %d' % args.epochs_ae)
    writelog(f, 'Batch Size: %d' % args.batch_size)
    writelog(f, 'Learning Rate: %s' % str(args.lr_ae))
    writelog(f, 'Weight Decay: %s' % str(args.weight_decay))
    writelog(f, '======================')
    f.close()

    print("Pretraining autoencoder... \n")
    writer = SummaryWriter(log_dir=os.path.join(directory, 'models_logs'))



    input_channels = 116
    kernel_size = 15
    dropout = 0.25
    n_hidden=[64, 50, 10]
    print(n_hidden)
    model = TCM_DRNN(args, input_size=input_channels, num_channels=[64] * 5, kernel_size=kernel_size, dropout=dropout, n_hidden=n_hidden)
    model.to(args.device)

    ## MSE loss
    loss_ae = nn.MSELoss()

    ## Optimizer
    # optimizer = torch.optim.Adam(seq2seq.parameters(), lr=args.lr_ae, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=1e-3, weight_decay=0)




    for epoch in tqdm(range(args.epochs_ae)):

        # training
        model.train()
        all_loss = 0
        recon_loss = 0
        km_loss = 0

        for batch_idx, inputs in enumerate(train_dl):
            inputs = inputs.type(torch.FloatTensor).to(args.device)  #torch.Size([16batch, 150, 116])

            optimizer.zero_grad()  # 기울기에 대한 정보 초기화
            h_real, real_reconstr, kmeans_loss = model(inputs)
            loss_mse = loss_ae(inputs, real_reconstr)  # x_reconstr(decoded) & 원본(input) 사이의 평균제곱오차

            total_loss = loss_mse +  (0.7) * kmeans_loss
            total_loss.backward()  # 기울기 구함
            optimizer.step()  # 최적화 진행

            all_loss += total_loss.item()
            recon_loss += loss_mse.item()
            km_loss += kmeans_loss.item()

        if epoch % 5 == 0 and epoch != 0:
            U, S, Vh = torch.linalg.svd(h_real.T)
            sorted_indices = torch.argsort(S)
            topk_evecs = Vh[sorted_indices.flip(dims=(0,))[:args.n_clusters], :]
            F_new = topk_evecs.T
            model.kmeans.F.data = F_new

        train_loss = all_loss / (batch_idx + 1)
        recon_loss = recon_loss / (batch_idx + 1)
        km_loss = km_loss / (batch_idx + 1)

        writer.add_scalar("training loss", train_loss, epoch)
        writer.add_scalar("reconstruction loss", recon_loss, epoch)
        writer.add_scalar("kmeans loss", km_loss, epoch)
        print()
        print("total loss for epoch {} is : {}, recon loss: {}, kmeans loss: {}".format(epoch, train_loss, recon_loss, km_loss))
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': train_loss
        }, os.path.join(directory, 'models_weights') + '/checkpoint_epoch_{}_loss_{:.5f}.pt'.format(epoch, train_loss))


        # kmeans
        model.eval()
        with torch.no_grad():

            # kmeans with train data
            for t_idx, train in enumerate(kmeans_train):
                train = train.type(torch.FloatTensor).to(args.device)  # torch.Size([16batch, 150, 116])
                features = model.tcn(train)
                features = features.detach().cpu().numpy().squeeze()

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pred_train = deep_utils.cluster_using_kmeans(features, args.n_clusters) ## CLUSTERING



                if epoch == 0 or epoch == 5 or epoch == 10 or epoch == 25 or (epoch) == 50 or (epoch) == 100 or (epoch) == 150 or (epoch) == 200 or (epoch) == 250 or (epoch) == 300 or (epoch) == 350:
                    if (t_idx+1) == 100:

                        f = open(os.path.join(directory, 'visualization/clustering_result.log'), 'a')
                        writelog(f, 'Epochs: %d' % (epoch))
                        for i in range(args.n_clusters):
                            writelog(f, 'number of cluster %d: %d' % (i, np.array(np.where(pred_train == i)).shape[-1]))
                        writelog(f, '======================')
                        f.close()

                        tsne = TSNE(n_components=2)
                        features_t = tsne.fit_transform(features)
                        aa = np.expand_dims(np.arange(1200), 1)
                        con = np.concatenate((features_t, aa), 1) #(1200, 3)

                        f = open(os.path.join(directory, 'visualization/clustering_result.log'), 'a')
                        writelog(f, 'Epochs: %d' % (epoch))
                        for i in range(args.n_clusters):
                            writelog(f, 'predicted cluster %d: %s' % (i, np.asarray(con[pred_train == i][:,2], dtype=int).tolist()))
                        writelog(f, '======================')
                        writelog(f, ' ')
                        f.close()

                        predicted_0 = con[pred_train == 0]
                        predicted_1 = con[pred_train == 1]
                        predicted_2 = con[pred_train == 2]
                        # predicted_3 = con[pred_train == 3]
                        # predicted_4 = con[pred_train == 4]
                        # predicted_5 = con[pred_train == 5]
                        # predicted_6 = con[pred_train == 6]
                        # predicted_7 = con[pred_train == 7]

                        for i in range(predicted_0.shape[0]):
                            plt.scatter(predicted_0[i][0], predicted_0[i][1], color='red', alpha=0.5, s=40)
                            if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                                plt.text(predicted_0[i][0], predicted_0[i][1] + .03, int(predicted_0[i][2]), fontsize=5)

                        for i in range(predicted_1.shape[0]):
                            plt.scatter(predicted_1[i][0], predicted_1[i][1], color='orange', alpha=0.5, s=40)
                            if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                                plt.text(predicted_1[i][0], predicted_1[i][1] + .03, int(predicted_1[i][2]), fontsize=5)

                        for i in range(predicted_2.shape[0]):
                            plt.scatter(predicted_2[i][0], predicted_2[i][1], color='yellow', alpha=0.5, s=40)
                            if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                                plt.text(predicted_2[i][0], predicted_2[i][1] + .03, int(predicted_2[i][2]), fontsize=5)

                        # for i in range(predicted_3.shape[0]):
                        #     plt.scatter(predicted_3[i][0], predicted_3[i][1], color='green', alpha=0.5, s=40)
                        #     if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                        #         plt.text(predicted_3[i][0], predicted_3[i][1] + .03, int(predicted_3[i][2]), fontsize=5)
                        #
                        # for i in range(predicted_4.shape[0]):
                        #     plt.scatter(predicted_4[i][0], predicted_4[i][1], color='blue', alpha=0.5, s=40)
                        #     if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                        #         plt.text(predicted_4[i][0], predicted_4[i][1] + .03, int(predicted_4[i][2]), fontsize=5)
                        #
                        # for i in range(predicted_5.shape[0]):
                        #     plt.scatter(predicted_5[i][0], predicted_5[i][1], color='purple', alpha=0.5, s=40)
                        #     if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                        #         plt.text(predicted_5[i][0], predicted_5[i][1] + .03, int(predicted_5[i][2]), fontsize=5)
                        #
                        # for i in range(predicted_6.shape[0]):
                        #     plt.scatter(predicted_6[i][0], predicted_6[i][1], color='pink', alpha=0.5, s=40)
                        #     if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                        #         plt.text(predicted_6[i][0], predicted_6[i][1] + .03, int(predicted_6[i][2]), fontsize=5)
                        #
                        # for i in range(predicted_7.shape[0]):
                        #     plt.scatter(predicted_7[i][0], predicted_7[i][1], color='deeppink', alpha=0.5, s=40)
                        #     if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                        #         plt.text(predicted_7[i][0], predicted_7[i][1] + .03, int(predicted_7[i][2]), fontsize=5)


                        plt.savefig(os.path.join(directory, 'visualization/EPOCH/') + 'sub100_predicted_epoch_{}.png'.format(epoch))
                        plt.close()



            # kmeans with test data
            for tt_idx, test in enumerate(kmeans_test):
                test = test.type(torch.FloatTensor).to(args.device)
                t_features = model.tcn(test)
                t_features = t_features.detach().cpu().numpy().squeeze()

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pred_test = deep_utils.cluster_using_kmeans(t_features, args.n_clusters)





    print("Ending pretraining autoencoder. \n")




















if __name__ == "__main__":

    parser = get_arguments()
    args = parser.parse_args()


    # GPU Configuration
    gpu_id = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print(args.device)


    # data load
    data = np.load('/DataCommon/jwlee/RESTING_LR/RESTING_LR.npz')
    samples = data['HCP_RESTING_LR'] #(1075, 1200, 116)

    # minmax
    mm = MinMaxScaler()
    results = []
    for ss in range(1075):
        results.append(mm.fit_transform(samples[ss]))
    sample = np.array(results)


    # train, validation, test
    data_label6 = np.load('/DataCommon/jwlee/EMOTION_LR/cluster_6_hcp_emotion_label.npz')
    label = data_label6['label_list_LR']  #(1041, 150)


    # number of clusters
    args.n_clusters = 3
    args.timeseries = samples.shape[1]


    real_train_data, real_test_data = train_test_split(sample, random_state=42, test_size=0.2)


    # make tensor
    real_train_data = torch.FloatTensor(real_train_data)
    real_test_data = torch.FloatTensor(real_test_data)

    # make dataloader with batch_size
    train_dl = DataLoader(real_train_data, batch_size=4)
    real_train_dl = DataLoader(real_train_data, batch_size=1)

    test_dl = DataLoader(real_test_data, batch_size=4)
    real_test_dl = DataLoader(real_test_data, batch_size=1)




    directory = os.path.join(args.dir_root, args.model_name, args.dataset_name, 'Epochs' + str(args.epochs_ae) + '_BS_' + str(args.batch_size) + '_LR_' + str(args.lr_ae) + '_wdcay_' + str(args.weight_decay))






    if args.ae_weights is None and args.epochs_ae > 0:  ########### pretrain


        pretrain_autoencoder(args, train_dl=train_dl, test_dl=test_dl, kmeans_train=real_train_dl, kmeans_test=real_test_dl, directory=directory)













