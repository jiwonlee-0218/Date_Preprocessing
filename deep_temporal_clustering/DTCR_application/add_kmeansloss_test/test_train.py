import torch
import datetime
import torch.nn as nn
import os
import numpy as np
from test_main_config import get_arguments
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import test_load_data
import test_utils
from test_KM_model import Seq2Seq
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
    writelog(f, 'Max Patience: %d (10 percent of the epoch size)' % args.max_patience)
    writelog(f, 'Batch Size: %d' % args.batch_size)
    writelog(f, 'Learning Rate: %s' % str(args.lr_ae))
    writelog(f, 'Weight Decay: %s' % str(args.weight_decay))
    writelog(f, '======================')
    f.close()

    print("Pretraining autoencoder... \n")
    writer = SummaryWriter(log_dir=os.path.join(directory, 'models_logs'))


    # seq2seq = Seq2Seq(args, n_input=116, filter_1=64, n_hidden=[64, 32, 20])
    seq2seq = Seq2Seq(args, n_input=116, n_hidden=[100, 50, 10])
    seq2seq = seq2seq.to(args.device)
    print(seq2seq)

    ## MSE loss
    loss_ae = nn.MSELoss()

    ## Optimizer
    optimizer = torch.optim.Adam(seq2seq.parameters(), lr=args.lr_ae, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    lamda = 0.1
    best_indicator = 0
    best_epoch = -1
    nmi_best_indicator = 0
    nmi_best_epoch = -1
    for epoch in tqdm(range(args.epochs_ae)):


        # training
        seq2seq.train()
        all_loss = 0
        recon_loss = 0
        km_loss = 0

        for batch_idx, (inputs, _) in enumerate(train_dl):
            inputs = inputs.type(torch.FloatTensor).to(args.device)  #torch.Size([16batch, 150, 116])

            optimizer.zero_grad()  # 기울기에 대한 정보 초기화
            h_real, real_reconstr, kmeans_loss = seq2seq(inputs)
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
            topk_evecs = Vh[sorted_indices.flip(dims=(0,))[:7], :] ####################################################
            F_new = topk_evecs.T
            seq2seq.kmeans.F.data = F_new

        train_loss = all_loss / (batch_idx + 1)
        recon_loss = recon_loss / (batch_idx + 1)
        km_loss = km_loss / (batch_idx + 1)

        writer.add_scalar("training loss", train_loss, epoch)
        writer.add_scalar("reconstruction loss", recon_loss, epoch)
        writer.add_scalar("kmeans loss", km_loss, epoch)
        print()
        print("total loss for epoch {} is : {}, recon loss: {}, kmeans loss: {}".format(epoch, train_loss, recon_loss, km_loss))
        torch.save({
            'model_state_dict': seq2seq.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': train_loss
        }, os.path.join(directory, 'models_weights') + '/checkpoint_epoch_{}_loss_{:.5f}.pt'.format(epoch, train_loss))


        # kmeans
        seq2seq.eval()
        with torch.no_grad():
            total_RI = 0
            test_total_RI = 0
            total_NMI = 0
            test_total_NMI = 0

            # kmeans with train data
            for t_idx, (train, train_label) in enumerate(kmeans_train):
                train = train.type(torch.FloatTensor).to(args.device)  # torch.Size([1batch, 176, 116])
                features, encoder_state_fw = seq2seq.f_enc(train)
                features = features.detach().cpu().numpy().squeeze() #(176, 10)

                pred_train = test_utils.cluster_using_kmeans(features, args.n_clusters)
                # RI_score_train = test_utils.ri_score(train_label.detach().cpu().numpy().squeeze(), pred_train)
                # NMI_score_train = test_utils.nmi_score(train_label.detach().cpu().numpy().squeeze(), pred_train)
                #
                # total_RI += RI_score_train
                # total_NMI += NMI_score_train

                if epoch == 0 or (epoch) == 50 or (epoch) == 100 or (epoch) == 150 or (epoch) == 200 or (epoch) == 250 or (epoch) == 300 or (epoch) == 350:
                    if (t_idx+1) == 100:

                        f = open(os.path.join(directory, 'visualization/clustering_result.log'), 'a')
                        writelog(f, 'Epochs: %d' % (epoch))
                        for i in range(7):
                            writelog(f, 'number of cluster %d: %d' % (i, np.array(np.where(pred_train == i)).shape[-1]))
                        writelog(f, '======================')
                        f.close()


                        tsne = TSNE(n_components=2)
                        features_t = tsne.fit_transform(features)

                        predicted_0 = features_t[pred_train == 0]
                        predicted_1 = features_t[pred_train == 1]
                        predicted_2 = features_t[pred_train == 2]
                        predicted_3 = features_t[pred_train == 3]
                        predicted_4 = features_t[pred_train == 4]
                        predicted_5 = features_t[pred_train == 5]
                        predicted_6 = features_t[pred_train == 6]


                        plt.scatter(predicted_0[:, 0], predicted_0[:, 1], color='lightsalmon', alpha=0.5)
                        plt.scatter(predicted_1[:, 0], predicted_1[:, 1], color='saddlebrown', alpha=0.5)
                        plt.scatter(predicted_2[:, 0], predicted_2[:, 1], color='goldenrod', alpha=0.5)
                        plt.scatter(predicted_3[:, 0], predicted_3[:, 1], color='khaki', alpha=0.5)
                        plt.scatter(predicted_4[:, 0], predicted_4[:, 1], color='aquamarine', alpha=0.5)
                        plt.scatter(predicted_5[:, 0], predicted_5[:, 1], color='slategray', alpha=0.5)
                        plt.scatter(predicted_6[:, 0], predicted_6[:, 1], color='teal', alpha=0.5)
                        plt.savefig(os.path.join(directory, 'visualization/EPOCH/') + 'sub100_predicted_epoch_{}.png'.format(epoch))
                        plt.close()


                        labeled_0 = features_t[train_label.squeeze() == 0] #cue
                        labeled_1 = features_t[train_label.squeeze() == 1]
                        labeled_2 = features_t[train_label.squeeze() == 2]
                        labeled_3 = features_t[train_label.squeeze() == 3]
                        labeled_4 = features_t[train_label.squeeze() == 4]
                        labeled_5 = features_t[train_label.squeeze() == 5]
                        labeled_6 = features_t[train_label.squeeze() == 6]

                        plt.scatter(labeled_0[:, 0], labeled_0[:, 1], label='cue', color='black', alpha=0.5)
                        plt.scatter(labeled_1[:, 0], labeled_1[:, 1], label='shape_1', color='red', alpha=0.5)
                        plt.scatter(labeled_2[:, 0], labeled_2[:, 1], label='face_2', color='orange', alpha=0.5)
                        plt.scatter(labeled_3[:, 0], labeled_3[:, 1], label='shape_3', color='yellow', alpha=0.5)
                        plt.scatter(labeled_4[:, 0], labeled_4[:, 1], label='face_4', color='green', alpha=0.5)
                        plt.scatter(labeled_5[:, 0], labeled_5[:, 1], label='shape_5', color='pink', alpha=0.5)
                        plt.scatter(labeled_6[:, 0], labeled_6[:, 1], label='face_6', color='purple', alpha=0.5)
                        plt.legend()
                        plt.savefig(os.path.join(directory, 'visualization/EPOCH/') + 'sub100_labeld_epoch_{}.png'.format(epoch))
                        plt.close()


            # total_RI = total_RI / (t_idx + 1)
            # total_NMI = total_NMI / (t_idx + 1)


            # kmeans with test data
            # for tt_idx, (test, test_label) in enumerate(kmeans_test):
            #     test = test.type(torch.FloatTensor).to(args.device)
            #     tt_features, tt_encoder_state_fw = seq2seq.f_enc(test)
            #     tt_features = tt_features.detach().cpu().numpy().squeeze()
            #
            #     pred_test = test_utils.cluster_using_kmeans(tt_features, args.n_clusters)
            #     RI_score_test = test_utils.ri_score(test_label.detach().cpu().numpy().squeeze(), pred_test)
            #     NMI_score_test = test_utils.nmi_score(test_label.detach().cpu().numpy().squeeze(), pred_test)
            #
            #     test_total_RI += RI_score_test
            #     test_total_NMI += NMI_score_test
            # test_total_RI = test_total_RI / (tt_idx + 1)
            # test_total_NMI = test_total_NMI / (tt_idx + 1)
            #
            #
            # if test_total_RI > best_indicator:
            #     best_indicator = test_total_RI
            #     best_epoch = epoch+1
            #
            # if test_total_NMI > nmi_best_indicator:
            #     nmi_best_indicator = test_total_NMI
            #     nmi_best_epoch = epoch+1
            # print('Epoch {}'.format(epoch + 1))


    # f = open(os.path.join(directory, 'Rand_Index.log'), 'a')
    # writelog(f, 'number of clusters: %d' % args.n_clusters)
    # writelog(f, 'best\tRI = {}, epoch = {}\n\n'.format(best_indicator, best_epoch))
    # writer.close()
    #
    # f = open(os.path.join(directory, 'NMI.log'), 'a')
    # writelog(f, 'number of clusters: %d' % args.n_clusters)
    # writelog(f, 'best\tNMI = {}, epoch = {}\n\n'.format(nmi_best_indicator, nmi_best_epoch))
    # writer.close()
    # print("Ending pretraining autoencoder. \n")




















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
    data = np.load('/DataCommon/jwlee/EMOTION_LR/hcp_emotion.npz')
    samples = data['tfMRI_EMOTION_LR'] #(1041, 176, 116)
    # samples = samples[:, :, 48:56]  # (1041, 150, 8)



    # minmax
    mm = MinMaxScaler()
    results = []
    for ss in range(1041):
        results.append(mm.fit_transform(samples[ss]))
    sample = np.array(results)

    # train, validation, test
    data_label = np.load('/DataCommon/jwlee/EMOTION_LR/cluster_7_hcp_emotion_label.npz')
    label = data_label['label_list_LR'] #(1041, 176)


    # number of clusters
    args.n_clusters = 7
    args.timeseries = label.shape[1]

    real_train_data, real_test_data, real_train_label, real_test_label = train_test_split(sample, label, random_state=42, test_size=0.2)


    # make tensor
    real_train_data, real_train_label = torch.FloatTensor(real_train_data), torch.FloatTensor(real_train_label)
    real_test_data, real_test_label = torch.FloatTensor(real_test_data), torch.FloatTensor(real_test_label)

    # make dataloader with batch_size
    real_train_ds = TensorDataset(real_train_data, real_train_label)
    train_dl = DataLoader(real_train_ds, batch_size=16)
    real_train_dl = DataLoader(real_train_ds, batch_size=1)

    real_test_ds = TensorDataset(real_test_data, real_test_label)
    test_dl = DataLoader(real_test_ds, batch_size=16)
    real_test_dl = DataLoader(real_test_ds, batch_size=1)







    directory = os.path.join(args.dir_root, args.model_name, args.dataset_name, 'Epochs' + str(args.epochs_ae) + '_BS_' + str(args.batch_size) + '_LR_' + str(args.lr_ae) + '_wdcay_' + str(args.weight_decay))






    if args.ae_weights is None and args.epochs_ae > 0:  ########### pretrain


        pretrain_autoencoder(args, train_dl=train_dl, test_dl=test_dl, kmeans_train=real_train_dl, kmeans_test=real_test_dl, directory=directory)






    if args.ae_weights is not None and args.epochs_ae > 0:

        if not os.path.exists(os.path.join(directory, 'visualization')):
            os.makedirs(os.path.join(directory, 'visualization'))

        seq2seq_enc = Seq2Seq(args, n_input=116, n_hidden=[100, 50, 10])
        seq2seq_enc = seq2seq_enc.to(args.device)


        ''' just using encoder to kmeans '''
        sub100_label = real_train_label[99].detach().cpu().numpy()  # (150, )
        sub100 = real_train_data[99]  # 100subject
        sub100 = sub100.unsqueeze(0)
        sub100 = sub100.type(torch.FloatTensor).to(args.device)
        features, encoder_state_fw = seq2seq_enc.f_enc(sub100)
        features = features.detach().cpu().numpy().squeeze()  # (150, 20)
        pred_train = test_utils.cluster_using_kmeans(features, args.n_clusters)
        RI = test_utils.ri_score(sub100_label, pred_train)
        NMI = test_utils.nmi_score(features, pred_train)
        tsne = TSNE(n_components=2)

        features_t = tsne.fit_transform(features)  # (150, 2)
        predicted_0 = features_t[pred_train == 0]  # shape
        predicted_1 = features_t[pred_train == 1]  # face
        predicted_2 = features_t[pred_train == 2]  # shape
        predicted_3 = features_t[pred_train == 3]  # face
        predicted_4 = features_t[pred_train == 4]  # shape
        predicted_5 = features_t[pred_train == 5]  # face
        plt.scatter(predicted_0[:, 0], predicted_0[:, 1], color='lightsalmon', alpha=0.5)
        plt.scatter(predicted_1[:, 0], predicted_1[:, 1], color='sandybrown', alpha=0.5)
        plt.scatter(predicted_2[:, 0], predicted_2[:, 1], color='goldenrod', alpha=0.5)
        plt.scatter(predicted_3[:, 0], predicted_3[:, 1], color='khaki', alpha=0.5)
        plt.scatter(predicted_4[:, 0], predicted_4[:, 1], color='aquamarine', alpha=0.5)
        plt.scatter(predicted_5[:, 0], predicted_5[:, 1], color='slategray', alpha=0.5)
        plt.savefig(os.path.join(directory, 'visualization/EPOCH/') + 'sub100_predicted_epoch_0_raw.png')
        plt.close()

        labeled_0 = features_t[sub100_label == 1]
        labeled_1 = features_t[sub100_label == 2]
        labeled_2 = features_t[sub100_label == 3]
        labeled_3 = features_t[sub100_label == 4]
        labeled_4 = features_t[sub100_label == 5]
        labeled_5 = features_t[sub100_label == 6]
        plt.scatter(labeled_0[:, 0], labeled_0[:, 1], label='shape_1', color='red', alpha=0.5)
        plt.scatter(labeled_1[:, 0], labeled_1[:, 1], label='face_2', color='orange', alpha=0.5)
        plt.scatter(labeled_2[:, 0], labeled_2[:, 1], label='shape_3', color='yellow', alpha=0.5)
        plt.scatter(labeled_3[:, 0], labeled_3[:, 1], label='face_4', color='green', alpha=0.5)
        plt.scatter(labeled_4[:, 0], labeled_4[:, 1], label='shape_5', color='blue', alpha=0.5)
        plt.scatter(labeled_5[:, 0], labeled_5[:, 1], label='face_6', color='purple', alpha=0.5)
        plt.legend()
        plt.savefig(os.path.join(directory, 'visualization/EPOCH/') + 'sub100_labeld_epoch_0_raw.png')
        plt.close()







        path = os.path.join(directory, args.ae_weights)
        full_path = sorted(glob.glob(path + '*'), key=os.path.getctime)
        full_path = full_path[109]
        print("I got: " + full_path + " weights")


        seq2seq = Seq2Seq(args, n_input=116, n_hidden=[100, 50, 10])
        checkpoint = torch.load(full_path, map_location=args.device)
        seq2seq.load_state_dict(checkpoint['model_state_dict'])
        seq2seq = seq2seq.to(args.device)


        real_train_data = real_train_data.type(torch.FloatTensor).to(args.device)
        features, encoder_state_fw = seq2seq.f_enc(real_train_data)
        features = features.detach().cpu().numpy()












        ''' 0 subject '''
        kmc = KMeans(n_clusters=2, init="k-means++").fit(features[0])
        pmc = kmc.predict(features[0])
        centroid = kmc.cluster_centers_
        predicted_0 = features[0][pmc == 0] #shape
        predicted_1 = features[0][pmc == 1] #face
        plt.scatter(predicted_0[:, 0], predicted_0[:, 1], color='red', alpha=0.5)
        plt.scatter(predicted_1[:, 0], predicted_1[:, 1], color='blue', alpha=0.5)
        plt.scatter(centroid[:, 0], centroid[:, 1], s=70, color='black')
        plt.savefig(os.path.join(directory, 'visualization/') + 'sub0_predicted.png')
        plt.close()

        labeled_0 = features[0][real_train_label[0] == 1] #shape
        labeled_1 = features[0][real_train_label[0] == 2] #face
        plt.scatter(labeled_0[:, 0], labeled_0[:, 1], label='shape', color='orange', alpha=0.5)
        plt.scatter(labeled_1[:, 0], labeled_1[:, 1], label='face', color='green', alpha=0.5)
        plt.legend()
        plt.savefig(os.path.join(directory, 'visualization/') + 'sub0_label.png')
        plt.close()

        ''' 100 subject '''
        kmc = KMeans(n_clusters=2, init="k-means++").fit(features[100])
        pmc = kmc.predict(features[100])
        centroid = kmc.cluster_centers_
        predicted_0 = features[100][pmc == 0]  # shape
        predicted_1 = features[100][pmc == 1]  # face
        plt.scatter(predicted_0[:, 0], predicted_0[:, 1], color='red', alpha=0.5)
        plt.scatter(predicted_1[:, 0], predicted_1[:, 1], color='blue', alpha=0.5)
        plt.scatter(centroid[:, 0], centroid[:, 1], s=70, color='black')
        plt.savefig(os.path.join(directory, 'visualization/') + 'sub100_predicted.png')
        plt.close()

        labeled_0 = features[100][real_train_label[100] == 1]  # shape
        labeled_1 = features[100][real_train_label[100] == 2]  # face
        plt.scatter(labeled_0[:, 0], labeled_0[:, 1], label='shape', color='orange', alpha=0.5)
        plt.scatter(labeled_1[:, 0], labeled_1[:, 1], label='face', color='green', alpha=0.5)
        plt.legend()
        plt.savefig(os.path.join(directory, 'visualization/') + 'sub100_label.png')
        plt.close()




        ''' 200 subject '''
        kmc = KMeans(n_clusters=2, init="k-means++").fit(features[200])
        pmc = kmc.predict(features[200])
        centroid = kmc.cluster_centers_
        predicted_0 = features[200][pmc == 0]  # shape
        predicted_1 = features[200][pmc == 1]  # face
        plt.scatter(predicted_0[:, 0], predicted_0[:, 1], color='red', alpha=0.5)
        plt.scatter(predicted_1[:, 0], predicted_1[:, 1], color='blue', alpha=0.5)
        plt.scatter(centroid[:, 0], centroid[:, 1], s=70, color='black')
        plt.savefig(os.path.join(directory, 'visualization/') + 'sub200_predicted.png')
        plt.close()

        labeled_0 = features[200][real_train_label[200] == 1]  # shape
        labeled_1 = features[200][real_train_label[200] == 2]  # face
        plt.scatter(labeled_0[:, 0], labeled_0[:, 1], label='shape', color='orange', alpha=0.5)
        plt.scatter(labeled_1[:, 0], labeled_1[:, 1], label='face', color='green', alpha=0.5)
        plt.legend()
        plt.savefig(os.path.join(directory, 'visualization/') + 'sub200_label.png')
        plt.close()











