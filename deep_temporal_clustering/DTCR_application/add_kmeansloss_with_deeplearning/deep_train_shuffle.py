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
from deep_KM_model import Transformer
from deep_KM_model_KLloss import ClusterNet4
import deep_KM_model
from deep_KM_model_contrastive import ContrastiveTCN
from M_TCN import M_TCNetwork
import random
import glob
import copy
import matplotlib.pyplot as plt
import warnings
from sklearn.manifold import TSNE

from tslearn.clustering import TimeSeriesKMeans





''' shuffle timepoints '''
def shuffle_timeseries(data, label, rate=1.0):

    ordered_index = np.arange(len(data))
    ordered_index.astype(int)


    shuffled_index = np.random.choice(ordered_index, size=int(np.floor(rate * len(data))), replace=False)
    ordered_index[shuffled_index] = -1


    shuffled_index = np.random.permutation(shuffled_index)
    ordered_index[ordered_index == -1] = shuffled_index
    data = data[ordered_index]
    label = label[ordered_index]

    return data, label


def construct_classification_dataset(dataset):
    real_dataset = copy.deepcopy(dataset)
    shuffled_dataset = []
    shuffled_label = []
    for seq in real_dataset:
        data, label = shuffle_timeseries(seq[0], seq[1])
        shuffled_dataset.append(data)
        shuffled_label.append(label)

    shuffled_dataset = torch.stack(shuffled_dataset)
    shuffled_label = torch.stack(shuffled_label)

    print('dataset shape: ', shuffled_dataset.shape)
    print('label shape:', shuffled_label.shape)

    return shuffled_dataset, shuffled_label


def writelog(file, line):
    file.write(line + '\n')
    print(line)


# batch 12, 12, 1, 1
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
    writelog(f, 'Max Patience: %d (10 percent of the epoch size)' % args.max_patience)
    writelog(f, 'Batch Size: %d' % args.batch_size)
    writelog(f, 'Learning Rate: %s' % str(args.lr_ae))
    writelog(f, 'Weight Decay: %s' % str(args.weight_decay))
    writelog(f, '======================')
    f.close()

    print("Pretraining autoencoder... \n")
    writer = SummaryWriter(log_dir=os.path.join(directory, 'models_logs'))


    # seq2seq = Seq2Seq(args, n_input=116, filters=64, n_hidden=[100, 50])
    # seq2seq = Seq2Seq(args, n_input=116, filters=64, n_hidden=[50, 10])
    # seq2seq = Seq2Seq(args, n_input=116, n_hidden=[50, 10])
    # seq2seq = Seq2Seq(args, n_input=116, filters=64, n_hidden=[32, 10])`
    # seq2seq = seq2seq.to(args.device)
    # print(seq2seq)


    # model = TCM_DRNN(args, input_size=116, num_channels=[64] * 5, kernel_size=15, dropout=0.25, n_hidden=[100, 64, 50, 10])
    # model.to(args.device)



    # model = DRNN_Model(args, input_size=116, n_hidden=[100, 50, 10])
    # model.to(args.device)



    model = Transformer(args)
    model.to(args.device)




    # input_channels = 1
    # kernel_size = 3
    # dropout = 0.25
    # dilation = [1, 4, 16]
    # n_hidden = [64, 50, 10]
    # model = M_TCNetwork(args, input_size=1, num_channels=2, dilation=[1, 4, 16], kernel_size=3, dropout=0.25, n_hidden=[64, 50, 10])
    # model.to(args.device)


    ## MSE loss
    loss_ae = nn.MSELoss()

    ## Optimizer
    # optimizer = torch.optim.Adam(seq2seq.parameters(), lr=args.lr_ae, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=1e-3, weight_decay=0)


    lamda = 0.1
    best_indicator = 0
    best_epoch = -1
    nmi_best_indicator = 0
    nmi_best_epoch = -1


    for epoch in tqdm(range(args.epochs_ae)):

        # training
        model.train()
        all_loss = 0
        recon_loss = 0
        km_loss = 0

        for batch_idx, (inputs, _) in enumerate(train_dl):
            inputs = inputs.type(torch.FloatTensor).to(args.device)  #torch.Sh, 150, 116])


            optimizer.zero_grad()  # 기울기에 대한 정보 초기화
            # h_real, real_reconstr, kmeans_loss = model(inputs)
            h_real, real_reconstr = model(inputs)
            loss_mse = loss_ae(inputs, real_reconstr)  # x_reconstr(decoded) & 원본(input) 사이의 평균제곱오차

            # total_loss = loss_mse +  (0.7 * kmeans_loss)
            # total_loss.backward()  # 기울기 구함

            loss_mse.backward()  # 기울기 구함
            optimizer.step()  # 최적화 진행

            # all_loss += total_loss.item()
            recon_loss += loss_mse.item()
            # km_loss += kmeans_loss.item()

        # if epoch % 5 == 0 and epoch != 0:
        #     U, S, Vh = torch.linalg.svd(h_real.T)
        #     sorted_indices = torch.argsort(S)
        #     topk_evecs = Vh[sorted_indices.flip(dims=(0,))[:args.n_clusters], :]
        #     F_new = topk_evecs.T
        #     model.kmeans.F.data = F_new

        # train_loss = all_loss / (batch_idx + 1)
        recon_loss = recon_loss / (batch_idx + 1)
        # km_loss = km_loss / (batch_idx + 1)

        # writer.add_scalar("training loss", train_loss, epoch)
        writer.add_scalar("reconstruction loss", recon_loss, epoch)
        # writer.add_scalar("kmeans loss", km_loss, epoch)
        print()
        # print("total loss for epoch {} is : {}, recon loss: {}, kmeans loss: {}".format(epoch, train_loss, recon_loss, km_loss))
        print("recon loss for epoch {} is : {}".format(epoch, recon_loss))
        # torch.save({
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'cluster_indicator_matrix': model.kmeans.F,
        #     'epoch': epoch,
        #     'loss': train_loss
        # }, os.path.join(directory, 'models_weights') + '/checkpoint_epoch_{}_loss_{:.5f}.pt'.format(epoch, train_loss))
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': recon_loss
        }, os.path.join(directory, 'models_weights') + '/checkpoint_epoch_{}_loss_{:.5f}.pt'.format(epoch, recon_loss))


        # kmeans
        model.eval()
        with torch.no_grad():
            total_RI = 0
            total_NMI = 0
            test_total_RI = 0
            test_total_NMI = 0

            # kmeans with train data
            for t_idx, (train, train_label) in enumerate(kmeans_train):
                train = train.type(torch.FloatTensor).to(args.device)  # torch.Size([16batch, 150, 116])
                features = model.backbone(train)  #encoded_out
                features = features.detach().cpu().numpy().squeeze()

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pred_train = deep_utils.cluster_using_kmeans(features, args.n_clusters) #Clustering
                    pred_cluster6_train = deep_utils.cluster_using_kmeans(features, 6)  # Clustering with 2
                RI_score_train = deep_utils.ri_score(train_label.detach().cpu().numpy().squeeze(), pred_train)
                NMI_score_train = deep_utils.nmi_score(train_label.detach().cpu().numpy().squeeze(), pred_train)

                total_RI += RI_score_train
                total_NMI += NMI_score_train

                if epoch == 0 or epoch == 5 or epoch == 10 or  epoch == 17 or epoch == 25 or epoch == 50 or epoch == 100 or epoch == 150 or epoch == 170 or epoch == 200 or epoch == 225 or epoch == 250 or epoch == 300 or epoch == 320 or epoch == 350 or epoch == 400 or epoch == 420:
                    if (t_idx+1) == 300:

                        f = open(os.path.join(directory, 'visualization/clustering_result.log'), 'a')
                        writelog(f, 'Epochs: %d' % (epoch))
                        for i in range(args.n_clusters):
                            writelog(f, 'number of cluster %d: %d' % (i, np.array(np.where(pred_train == i)).shape[-1]))
                        writelog(f, '======================')
                        f.close()

                        tsne = TSNE(n_components=2)
                        features_t = tsne.fit_transform(features)
                        aa = np.expand_dims(np.arange(args.timeseries), 1)
                        con = np.concatenate((features_t, aa), 1)

                        f = open(os.path.join(directory, 'visualization/clustering_result.log'), 'a')
                        writelog(f, 'Epochs: %d' % (epoch))
                        for i in range(args.n_clusters):
                            writelog(f, 'predicted cluster %d: %s' % (i, np.asarray(con[pred_train == i][:, 2], dtype=int).tolist()))
                        writelog(f, '======================')
                        writelog(f, ' ')
                        f.close()


                        predicted_0 = features_t[pred_cluster6_train == 0]  # shape
                        predicted_1 = features_t[pred_cluster6_train == 1]  # face
                        predicted_2 = features_t[pred_cluster6_train == 2]  # shape
                        predicted_3 = features_t[pred_cluster6_train == 3]  # face
                        predicted_4 = features_t[pred_cluster6_train == 4]  # shape
                        predicted_5 = features_t[pred_cluster6_train == 5]  # face
                        plt.scatter(predicted_0[:, 0], predicted_0[:, 1], color='coral', alpha=0.5)
                        plt.scatter(predicted_1[:, 0], predicted_1[:, 1], color='saddlebrown', alpha=0.5)
                        plt.scatter(predicted_2[:, 0], predicted_2[:, 1], color='goldenrod', alpha=0.5)
                        plt.scatter(predicted_3[:, 0], predicted_3[:, 1], color='orange', alpha=0.5)
                        plt.scatter(predicted_4[:, 0], predicted_4[:, 1], color='forestgreen', alpha=0.5)
                        plt.scatter(predicted_5[:, 0], predicted_5[:, 1], color='cadetblue', alpha=0.5)
                        plt.savefig(os.path.join(directory, 'visualization/EPOCH/') + 'sub300_predicted6_epoch_{}.png'.format(epoch))
                        plt.close()

                        predicted_0 = features_t[pred_train == 0]
                        predicted_1 = features_t[pred_train == 1]
                        plt.scatter(predicted_0[:, 0], predicted_0[:, 1], color='coral', alpha=0.5)
                        plt.scatter(predicted_1[:, 0], predicted_1[:, 1], color='saddlebrown', alpha=0.5)
                        plt.savefig(os.path.join(directory, 'visualization/EPOCH/') + 'sub300_predicted2_epoch_{}.png'.format(############ title
                                epoch))
                        plt.close()


                        labeled_0 = features_t[train_label.squeeze() == 1]
                        labeled_1 = features_t[train_label.squeeze() == 2]
                        # labeled_2 = features_t[train_label.squeeze() == 3]
                        # labeled_3 = features_t[train_label.squeeze() == 4]
                        # labeled_4 = features_t[train_label.squeeze() == 5]
                        # labeled_5 = features_t[train_label.squeeze() == 6]
                        plt.scatter(labeled_0[:, 0], labeled_0[:, 1], label='shape_1', color='red', alpha=0.5)
                        plt.scatter(labeled_1[:, 0], labeled_1[:, 1], label='face_2', color='darkslategray', alpha=0.5)
                        # plt.scatter(labeled_2[:, 0], labeled_2[:, 1], label='shape_3', color='firebrick', alpha=0.5)
                        # plt.scatter(labeled_3[:, 0], labeled_3[:, 1], label='face_4', color='seagreen', alpha=0.5)
                        # plt.scatter(labeled_4[:, 0], labeled_4[:, 1], label='shape_5', color='lightcoral', alpha=0.5)
                        # plt.scatter(labeled_5[:, 0], labeled_5[:, 1], label='face_6', color='mediumturquoise', alpha=0.5)
                        plt.legend()
                        plt.savefig(os.path.join(directory, 'visualization/EPOCH/') + 'sub300_label2_epoch_{}.png'.format(epoch))
                        plt.close()


                        # labeled_0 = features_t[label_6[0] == 1]  # label6는 안섞였기 때문에 feature_t는
                        # labeled_1 = features_t[label_6[0] == 2]
                        # labeled_2 = features_t[label_6[0] == 3]
                        # labeled_3 = features_t[label_6[0] == 4]
                        # labeled_4 = features_t[label_6[0] == 5]
                        # labeled_5 = features_t[label_6[0] == 6]
                        # plt.scatter(labeled_0[:, 0], labeled_0[:, 1], label='shape_1', color='red', alpha=0.5)
                        # plt.scatter(labeled_1[:, 0], labeled_1[:, 1], label='face_2', color='darkslategray', alpha=0.5)
                        # plt.scatter(labeled_2[:, 0], labeled_2[:, 1], label='shape_3', color='firebrick', alpha=0.5)
                        # plt.scatter(labeled_3[:, 0], labeled_3[:, 1], label='face_4', color='seagreen', alpha=0.5)
                        # plt.scatter(labeled_4[:, 0], labeled_4[:, 1], label='shape_5', color='lightcoral', alpha=0.5)
                        # plt.scatter(labeled_5[:, 0], labeled_5[:, 1], label='face_6', color='mediumturquoise', alpha=0.5)
                        # plt.legend()
                        # plt.savefig(os.path.join(directory, 'visualization/EPOCH/') + 'sub300_label6_epoch_{}.png'.format(epoch))
                        # plt.close()




            total_RI = total_RI / (t_idx + 1)
            total_NMI = total_NMI / (t_idx + 1)


            # kmeans with test data
            for tt_idx, (test, test_label) in enumerate(kmeans_test):
                test = test.type(torch.FloatTensor).to(args.device)
                t_features =  model.backbone(test)
                t_features = t_features.detach().cpu().numpy().squeeze()

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pred_test = deep_utils.cluster_using_kmeans(t_features, args.n_clusters)
                RI_score_test = deep_utils.ri_score(test_label.detach().cpu().numpy().squeeze(), pred_test)
                NMI_score_test = deep_utils.nmi_score(test_label.detach().cpu().numpy().squeeze(), pred_test)

                test_total_RI += RI_score_test
                test_total_NMI += NMI_score_test
            test_total_RI = test_total_RI / (tt_idx + 1)
            test_total_NMI = test_total_NMI / (tt_idx + 1)


            if test_total_RI > best_indicator:
                best_indicator = test_total_RI
                best_epoch = epoch

            if test_total_NMI > nmi_best_indicator:
                nmi_best_indicator = test_total_NMI
                nmi_best_epoch = epoch
            print('Epoch {} - {} train:{}\ttest:{}, {} train:{}\ttest:{}'.format(epoch, 'RI', total_RI, test_total_RI, 'NMI', total_NMI, test_total_NMI))


    f = open(os.path.join(directory, 'Rand_Index.log'), 'a')
    writelog(f, 'number of clusters: %d' % args.n_clusters)
    writelog(f, 'best\tRI = {}, epoch = {}\n\n'.format(best_indicator, best_epoch))
    writer.close()

    f = open(os.path.join(directory, 'NMI.log'), 'a')
    writelog(f, 'number of clusters: %d' % args.n_clusters)
    writelog(f, 'best\tNMI = {}, epoch = {}\n\n'.format(nmi_best_indicator, nmi_best_epoch))
    writer.close()
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
    data = np.load('/DataCommon/jwlee/EMOTION_LR/cluster_2_hcp_emotion.npz')
    samples = data['tfMRI_EMOTION_LR'] #(1041, 150, 116)

    # minmax
    mm = MinMaxScaler()
    results = []
    for ss in range(1040):
        results.append(mm.fit_transform(samples[ss]))
    sample = np.array(results)

    # train, validation, test
    data_label6 = np.load('/DataCommon/jwlee/EMOTION_LR/cluster_6_hcp_emotion_label.npz')
    label_6 = data_label6['label_list_LR']  # (1041, 150)
    label_6 = label_6[:1040]

    data_label2 = np.load('/DataCommon/jwlee/EMOTION_LR/cluster_2_hcp_emotion_label.npz')
    label_2 = data_label2['label_list_LR']  # (1041, 150)
    label_2 = label_2[:1040]

    sample, label_2 = torch.FloatTensor(sample), torch.FloatTensor(label_2)   # change 3
    train = TensorDataset(sample, label_2)

    train_shuffle_data, train_shuffle_label = construct_classification_dataset(train)

    # number of clusters
    args.n_clusters = len(np.unique(train_shuffle_label))
    args.timeseries = train_shuffle_data.shape[1]


    real_train_data, real_test_data, real_train_label, real_test_label = train_test_split(train_shuffle_data, train_shuffle_label, random_state=42, test_size=0.2)




    # make dataloader with batch_size
    real_train_ds = TensorDataset(real_train_data, real_train_label)
    train_dl = DataLoader(real_train_ds, batch_size=args.batch_size)
    real_train_dl = DataLoader(real_train_ds, batch_size=1)

    real_test_ds = TensorDataset(real_test_data, real_test_label)
    test_dl = DataLoader(real_test_ds, batch_size=args.batch_size)
    real_test_dl = DataLoader(real_test_ds, batch_size=1)




    directory = os.path.join(args.dir_root, args.model_name, args.dataset_name, 'Epochs' + str(args.epochs_ae) + '_BS_' + str(args.batch_size) + '_LR_' + str(args.lr_ae) + '_wdcay_' + str(args.weight_decay))



    if args.ae_weights is None and args.epochs_ae > 0:  ########### pretrain

        pretrain_autoencoder(args, train_dl=train_dl, test_dl=test_dl, kmeans_train=real_train_dl, kmeans_test=real_test_dl, directory=directory)









    if args.ae_weights is not None and args.autoencoder_test is not None:

        # full_path = '/home/jwlee/HMM/deep_temporal_clustering/DTCR_application/add_kmeansloss_with_deeplearning/Transformer/EMOTION_no_positionalencoding_no_kmeansloss_new/Epochs421_BS_8_LR_0.01_wdcay_1e-06/models_weights/checkpoint_epoch_384_loss_0.01242.pt'
        full_path = '/home/jwlee/HMM/deep_temporal_clustering/DTCR_application/add_kmeansloss_with_deeplearning/Transformer_Shuffle/EMOTION_no_positionalencoding_no_kmeansloss_new/Epochs421_BS_8_LR_0.01_wdcay_1e-06/models_weights/checkpoint_epoch_404_loss_0.01099.pt'



        model = Transformer(args)
        checkpoint = torch.load(full_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(args.device)

        total_RI = 0
        total_NMI = 0

        for t_idx, (train, train_label) in enumerate(real_test_dl):
            train = train.type(torch.FloatTensor).to(args.device)  # torch.Size([16batch, 150, 116])
            features = model.backbone(train)  # encoded_out
            features = features.detach().cpu().numpy().squeeze()
            pred_train = deep_utils.cluster_using_kmeans(features, 2)  # Clustering



            RI_score_train = deep_utils.ri_score(train_label.detach().cpu().numpy().squeeze(), pred_train)
            NMI_score_train = deep_utils.nmi_score(train_label.detach().cpu().numpy().squeeze(), pred_train)
            total_RI += RI_score_train
            total_NMI += NMI_score_train

            # if (t_idx + 1) == 10 or (t_idx + 1) == 100 or (t_idx + 1) == 200 or (t_idx + 1) == 300 or (t_idx + 1) == 400 or (t_idx + 1) == 500 or (t_idx + 1) == 600 or (t_idx + 1) == 700 or (t_idx + 1) == 800:
            if (t_idx + 1) == 10or (t_idx + 1) == 20 or (t_idx + 1) == 30 or (t_idx + 1) == 40 or (t_idx + 1) == 50 or (t_idx + 1) == 60 or (t_idx + 1) == 70 or (t_idx + 1) == 100 or (t_idx + 1) == 200:

                tsne = TSNE(n_components=2)
                features_t = tsne.fit_transform(features)

                labeled_0 = features_t[train_label.squeeze() == 1]
                labeled_1 = features_t[train_label.squeeze() == 2]
                labeled_2 = features_t[train_label.squeeze() == 3]
                labeled_3 = features_t[train_label.squeeze() == 4]
                labeled_4 = features_t[train_label.squeeze() == 5]
                labeled_5 = features_t[train_label.squeeze() == 6]
                plt.scatter(labeled_0[:, 0], labeled_0[:, 1], label='shape_1', color='red', alpha=0.5)
                plt.scatter(labeled_1[:, 0], labeled_1[:, 1], label='face_2', color='darkslategray', alpha=0.5)
                plt.scatter(labeled_2[:, 0], labeled_2[:, 1], label='shape_3', color='firebrick', alpha=0.5)
                plt.scatter(labeled_3[:, 0], labeled_3[:, 1], label='face_4', color='skyblue', alpha=0.5)
                plt.scatter(labeled_4[:, 0], labeled_4[:, 1], label='shape_5', color='lightcoral', alpha=0.5)
                plt.scatter(labeled_5[:, 0], labeled_5[:, 1], label='face_6', color='mediumturquoise', alpha=0.5)
                plt.legend()
                plt.show()
                plt.close()

        total_RI = total_RI / (t_idx + 1)
        total_NMI = total_NMI / (t_idx + 1)

        print(total_RI, t_idx)
        print(total_NMI)












        # if not os.path.exists(os.path.join(directory, 'visualization3')):
        #     os.makedirs(os.path.join(directory, 'visualization3/EPOCH'))
        #
        # model = ClusterNet4(args)
        # model = model.to(args.device)
        # print(model)
        #
        #
        # loss_ae = nn.MSELoss()
        #
        # # optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=args.lr_ae, weight_decay=args.weight_decay)
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_ae)
        #
        #
        # for epoch in tqdm(range(args.epochs_ae)):
        #
        #     # training
        #     model.train()
        #     total_train_loss = 0
        #     mse_train_loss = 0
        #     kl_train_loss = 0
        #
        #     for batch_idx, (inputs, _) in enumerate(train_dl):
        #         inputs = inputs.type(torch.FloatTensor).to(args.device)  # torch.Sh, 150, 116])
        #
        #         optimizer.zero_grad()  # 기울기에 대한 정보 초기화
        #
        #         h_real, real_reconstr, kkk = model(inputs)
        #
        #         loss_mse = loss_ae(inputs, real_reconstr)  # x_reconstr(decoded) & 원본(input) 사이의 평균제곱오차
        #         # loss_kl = kld(Q.log(), P)  ## (KL_LOSS(input, target))
        #         warnings.filterwarnings("ignore")
        #
        #         total_loss = loss_mse + (0.7 * loss_kl)
        #
        #         total_loss.backward()  # 기울기 구함
        #         torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=5)
        #         optimizer.step()  # 최적화 진행
        #
        #         total_train_loss += total_loss.item()
        #         mse_train_loss += loss_mse.item()
        #         kl_train_loss += loss_kl.item()
        #
        #     total_train_loss = total_train_loss / (batch_idx + 1)
        #     mse_train_loss = mse_train_loss / (batch_idx + 1)
        #     kl_train_loss = kl_train_loss / (batch_idx + 1)
        #
        #     print("total loss for epoch {} is : {}, recon loss: {}, kl loss: {}".format(epoch, total_train_loss,
        #                                                                                 mse_train_loss, kl_train_loss))
        #
        #     # kmeans
        #     model.eval()
        #     with torch.no_grad():
        #
        #         # kmeans with train data
        #         for t_idx, (train, train_label) in enumerate(real_train_dl):
        #             train = train.type(torch.FloatTensor).to(args.device)
        #             features = model.tcm_drnn.tcn(train)
        #             features = features.detach().cpu().numpy().squeeze()
        #
        #             with warnings.catch_warnings():
        #                 warnings.simplefilter("ignore")
        #                 pred_train = deep_utils.cluster_using_kmeans(features, args.n_clusters)  # Clustering
        #
        #             if epoch == 0 or epoch == 5 or epoch == 10 or epoch == 17 or epoch == 25 or epoch == 32 or epoch == 42 or epoch == 50 or epoch == 75 or epoch == 85 or epoch == 100:
        #                 if (t_idx + 1) == 100:
        #                     tsne = TSNE(n_components=2)
        #                     features_t = tsne.fit_transform(features)
        #
        #                     labeled_0 = features_t[label_6[0] == 1]
        #                     labeled_1 = features_t[label_6[0] == 2]
        #                     labeled_2 = features_t[label_6[0] == 3]
        #                     labeled_3 = features_t[label_6[0] == 4]
        #                     labeled_4 = features_t[label_6[0] == 5]
        #                     labeled_5 = features_t[label_6[0] == 6]
        #                     plt.scatter(labeled_0[:, 0], labeled_0[:, 1], label='shape_1', color='red', alpha=0.5)
        #                     plt.scatter(labeled_1[:, 0], labeled_1[:, 1], label='face_2', color='orange', alpha=0.5)
        #                     plt.scatter(labeled_2[:, 0], labeled_2[:, 1], label='shape_3', color='yellow', alpha=0.5)
        #                     plt.scatter(labeled_3[:, 0], labeled_3[:, 1], label='face_4', color='green', alpha=0.5)
        #                     plt.scatter(labeled_4[:, 0], labeled_4[:, 1], label='shape_5', color='blue', alpha=0.5)
        #                     plt.scatter(labeled_5[:, 0], labeled_5[:, 1], label='face_6', color='purple', alpha=0.5)
        #                     plt.legend()
        #                     plt.savefig(
        #                         os.path.join(directory, 'visualization3/EPOCH/') + 'sub100_labeld6_epoch_{}.png'.format(
        #                             epoch))
        #                     plt.close()










